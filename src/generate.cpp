// Emit scheduling (each rule was measured on an M4 Max):
//   * a small first chunk (3 frames, ~400 ms) for a low time to first audio,
//     then each chunk is half again as long, up to 32 frames, so a chunk's
//     generation stays inside the previous chunk's playback while the fixed
//     per-emit cost is amortised
//   * the emit test runs after every frame, not once per six-frame window,
//     otherwise the first chunk could never be smaller than a window
//   * flush on silence: once four latents in a row are silent and real
//     frames are pending, they are emitted, because the model sometimes pads
//     half a second of silence before its EOS fires. A latent is silent when
//     its L2 norm is under an absolute floor (silence sits near 2, speech at
//     5-12 for this model), raised to 1.6x the lead-in's norm when frames 1-3
//     are themselves quiet: some voices start speaking at once, and a purely
//     lead-relative threshold then classes everything as silent
//   * optional lead hold: keep chunks small until one has carried speech, for
//     hosts that cut the lead-in themselves at chunk boundaries
// The decoder is stateful, so each emit decodes only the new frames.

#include "generate.hpp"

#include "backend.hpp"
#include "bench.hpp"
#include "common.hpp"
#include "frame_graph.hpp"
#include "resample2x.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace lv {

namespace {

constexpr int   kTextWindow       = 5;
constexpr int   kSpeechWindow     = 6;
constexpr int   kTailSilentFrames = 4;
constexpr float kSilentNormRatio  = 1.6f;
constexpr float kSilentNormFloor  = 3.5f;
constexpr int   kWarmupSamples    = 2400;   // the decoder's zero-state transient, 100 ms at 24 kHz

// Runs its scope at utility QoS, which puts it on the efficiency cores of
// Apple chips. The post-filter costs about a fifth of the energy there and
// still runs ~40x faster than real time. A thread that is already at
// utility or below is left alone.
struct UtilityQos {
#if defined(__APPLE__)
    qos_class_t prev = QOS_CLASS_UNSPECIFIED;
    int         rel  = 0;
    bool        set  = false;
    UtilityQos() {
        pthread_get_qos_class_np(pthread_self(), &prev, &rel);
        if (prev == QOS_CLASS_UTILITY || prev == QOS_CLASS_BACKGROUND) return;
        set = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0) == 0;
    }
    ~UtilityQos() {
        if (set) pthread_set_qos_class_self_np(prev == QOS_CLASS_UNSPECIFIED ? QOS_CLASS_DEFAULT : prev, rel);
    }
#endif
};

// One forward pass of `n` tokens through a Qwen2 stack with its resident
// cache. Fills all hidden states, or only the last one.
bool run_stack(const Model& model, const std::vector<Qwen2LayerWeights>& layers, struct ggml_tensor* out_norm,
               int pos0, int n, const float* x_in, ResidentKV& kv, std::vector<float>* all, std::vector<float>* last) {
    const int hidden = model.cfg.hidden;
    Qwen2Hparams hp  = model.hparams();
    hp.flash_attn    = backend_supports_flash_attn();
    const int kv_len = kv.past_len + n;

    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    struct ggml_tensor* x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, n, 1);
    struct ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, n);

    struct ggml_tensor* h = x;
    std::vector<struct ggml_tensor*> writes;
    for (size_t li = 0; li < layers.size(); ++li) {
        const Qwen2LayerOut o = qwen2_layer(ctx, h, pos, mask, kv, static_cast<int>(li), layers[li], hp);
        h = o.y;
        writes.push_back(o.k_write);
        writes.push_back(o.v_write);
    }
    if (out_norm) h = rms_norm(ctx, h, out_norm, hp.rms_norm_eps);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    // Layer l's cache writes must precede layer l+1, which reads them; its
    // attention only enters the graph through a later expand.
    for (auto* w : writes) ggml_build_forward_expand(gf, w);
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t buf = allocate_ctx_tensors(ctx);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x, x_in, 0, sizeof(float) * hidden * n);
    std::vector<int32_t> pv(n);
    for (int i = 0; i < n; ++i) pv[i] = pos0 + i;
    ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int32_t) * n);
    std::vector<ggml_fp16_t> mv(static_cast<size_t>(kv_len) * n);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < kv_len; ++j) mv[static_cast<size_t>(i) * kv_len + j] = j > pos0 + i ? ninf : zero;
    ggml_backend_tensor_set(mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());

    const bool ok = compute_graph(gf);
    if (ok) {
        if (all) {
            all->resize(static_cast<size_t>(hidden) * n);
            ggml_backend_tensor_get(h, all->data(), 0, sizeof(float) * all->size());
        }
        if (last) {
            last->resize(hidden);
            ggml_backend_tensor_get(h, last->data(), sizeof(float) * hidden * (n - 1), sizeof(float) * hidden);
        }
        kv.past_len = kv_len;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

void upload(ResidentKV& dst, const Voice::Stack& src) {
    for (size_t l = 0; l < src.k.size(); ++l) {
        ggml_backend_tensor_set(dst.k[l], src.k[l].data(), 0, src.k[l].size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(dst.v[l], src.v[l].data(), 0, src.v[l].size() * sizeof(ggml_fp16_t));
    }
    dst.past_len = src.seq;
}

}  // namespace

Tokenizer::EncodeOptions encode_options(const lv_params& p) {
    Tokenizer::EncodeOptions o;
    o.max_merge_token_id           = std::max(0, p.merge_cap);
    o.canonical_first_word         = p.canonical_first_word != 0;
    o.last_word_max_merge_token_id = std::max(0, p.last_word_merge_cap);
    return o;
}

std::string terminate_text(const std::string& text) {
    // The voice prompt ends mid-conversation; the "\n" closes the user turn
    // and lets the model start its spoken reply.
    std::string t = text;
    while (!t.empty() && (t.back() == ' ' || t.back() == '\t' || t.back() == '\r' || t.back() == '\n')) t.pop_back();
    return t + "\n";
}

int output_sample_rate(const Model& model, const lv_params& p) {
    return (p.postfilter && model.dfn.ready) ? model.dfn.sample_rate : model.cfg.sample_rate;
}

int generate(const Model& model, const Voice& voice, const lv_params& p, const GenerateIO& io) {
    if (!io.pull_text || !io.emit_audio) return LV_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(compute_mutex());

    const auto& cfg   = model.cfg;
    const int  hidden = cfg.hidden, latent = cfg.latent, hd = cfg.head_dim, n_kv = cfg.n_kv_heads;
    const int  max_frames = std::max(1, p.max_frames);
    const bool guided     = voice.has_neg && p.cfg_scale > 1.0f;
    auto aborted_by_host  = [&] { return io.should_abort && io.should_abort(); };

    // ---- caches, seeded with the voice prompt ----
    const int max_text = std::max(16, p.max_text_tokens);
    ResidentKV kv_lm, kv_tlm, kv_neg;
    if (!kv_lm.init(cfg.n_layers_lm, hd, n_kv, voice.lm.seq + max_text + 32) ||
        !kv_tlm.init(cfg.n_layers_tlm, hd, n_kv, voice.tlm.seq + max_text + max_frames + 32) ||
        (guided && !kv_neg.init(cfg.n_layers_tlm, hd, n_kv, voice.neg_tlm.seq + max_frames + 32)))
        return LV_ERR_GENERIC;
    upload(kv_lm, voice.lm);
    upload(kv_tlm, voice.tlm);
    if (guided) upload(kv_neg, voice.neg_tlm);

    std::vector<float> hid_pos = voice.tlm_hidden;
    std::vector<float> hid_neg = guided ? voice.neg_tlm_hidden : std::vector<float>{};
    const std::vector<float> neg_anchor = hid_neg;
    const float anchor_w = std::min(1.0f, std::max(0.0f, p.neg_anchor));
    std::vector<float> neg_cond(hid_neg.size());

    // ---- diffusion ----
    DpmSchedule sched;
    sched.init(std::max(1, p.n_steps));
    const int freq = model.head_cfg.freq_size;
    std::vector<float> t_sin(static_cast<size_t>(freq) * sched.steps());
    for (int i = 0; i < sched.steps(); ++i)
        timestep_sinusoidal(static_cast<float>(sched.timesteps[i]), freq, t_sin.data() + static_cast<size_t>(i) * freq);
    std::mt19937 rng(p.seed ? p.seed : std::random_device{}());
    std::normal_distribution<float> normal(0.0f, 1.0f);

    AcousticDecoder::State dec;
    if (!model.decoder.init_state(&dec)) return LV_ERR_GENERIC;
    BenchTotals bench;
    LV_LOG_DEBUG("generate: guidance %s (%.2f, anchor %.2f), %d steps", guided ? "on" : "off",
                 static_cast<double>(p.cfg_scale), static_cast<double>(anchor_w), sched.steps());

    // ---- output: [2x upsample -> DFN3 at 48 kHz] -> lead trim -> host ----
    const bool use_pf   = p.postfilter && model.dfn.ready;
    const int  out_rate = output_sample_rate(model, p);
    std::unique_ptr<DfnStream> pf;
    Upsampler2x up2x;
    std::vector<float> pf_in, pf_out;
    if (use_pf) {
        BenchScope bs(&bench, "dfn warmup");
        pf = std::make_unique<DfnStream>(model.dfn);
        pf->warmup(100);
    }
    // The model puts digital silence in front of every reply (~150-800 ms),
    // or an artefact the post-filter turns into silence or a faint hiss
    // (around -60 dBFS, up to a second). It is dropped up to the onset,
    // keeping 60 ms ahead of it so a soft attack survives, then faded in
    // over 20 ms. The onset is found on what the listener would hear, after
    // the post-filter.
    const float kOnset     = 1e-2f;   // -40 dBFS; speech gets there within ~50 ms
    const int   kKeepAhead = 60 * out_rate / 1000;
    const int   kFadeIn    = 20 * out_rate / 1000;
    const int   kLeadMax   = 1500 * out_rate / 1000;   // never swallow more than this
    int  lead_dropped = 0;
    bool lead_done    = !p.trim_lead;
    int  fade_left    = 0;
    std::vector<float> lead_keep, trim_buf;
    auto ship = [&](const float* pcm, int n) -> bool {
        if (n <= 0) return true;
        if (lead_done && fade_left <= 0) return io.emit_audio(pcm, n);
        trim_buf.clear();
        if (!lead_done) {
            int first = 0;
            while (first < n && std::fabs(pcm[first]) < kOnset) ++first;
            if (first >= n && lead_dropped + n <= kLeadMax) {
                // Still the lead. Kept for the keep-ahead, and so a reply that
                // never rises above the onset is delivered as silence at the
                // end rather than as nothing.
                lead_dropped += n;
                lead_keep.insert(lead_keep.end(), pcm, pcm + n);
                return true;
            }
            if (first >= n) first = 0;   // cap reached: deliver from here on, whatever it is
            lead_done = true;
            fade_left = kFadeIn;
            const int from_chunk = std::min(first, kKeepAhead);
            const int from_lead  = std::min(static_cast<int>(lead_keep.size()), kKeepAhead - from_chunk);
            trim_buf.insert(trim_buf.end(), lead_keep.end() - from_lead, lead_keep.end());
            trim_buf.insert(trim_buf.end(), pcm + first - from_chunk, pcm + n);
            lead_keep.clear();
        } else {
            trim_buf.assign(pcm, pcm + n);
        }
        const int done = kFadeIn - fade_left;
        for (int i = 0; i < static_cast<int>(trim_buf.size()) && fade_left > 0; ++i, --fade_left) {
            const float t = static_cast<float>(done + i) / kFadeIn;
            trim_buf[i] *= t * t;
        }
        return io.emit_audio(trim_buf.data(), static_cast<int>(trim_buf.size()));
    };
    auto deliver = [&](const float* pcm24, int n, bool final_chunk) -> bool {
        if (!use_pf) return n > 0 ? ship(pcm24, n) : true;
        BenchScope bs(&bench, "postfilter");
        pf_in.clear();
        pf_out.clear();
        {
            UtilityQos qos;
            if (n > 0) up2x.process(pcm24, n, &pf_in);
            pf->process(pf_in.data(), static_cast<int>(pf_in.size()), &pf_out, final_chunk);
        }
        return pf_out.empty() || ship(pf_out.data(), static_cast<int>(pf_out.size()));
    };

    // ---- emit scheduling ----
    std::vector<float> latents;
    latents.reserve(static_cast<size_t>(std::min(max_frames, 4096)) * latent);
    const int max_chunk   = std::max(1, p.max_chunk_frames);
    const int first_chunk = std::min(max_chunk, std::max(1, p.first_chunk_frames));
    const int lead_chunk  = p.lead_chunk_frames > 0 ? std::min(p.lead_chunk_frames, first_chunk) : 0;
    int   emit_chunk     = lead_chunk > 0 ? lead_chunk : first_chunk;
    int   frames_emitted = 0, emits = 0;
    bool  first_emit     = true, sound_shipped = false;
    float silent_ref     = 0.0f;
    int   silent_run     = 0;
    int   pending_sound  = 0;   // non-silent frames since the last emit
    int   pending_speech = 0;   // the same, from frame 1 on
    auto frames = [&] { return static_cast<int>(latents.size()) / latent; };

    auto advance_chunk = [&] {
        if (lead_chunk > 0 && !sound_shipped) {
            if (pending_speech == 0) return;   // still the lead-in: hold the size
            sound_shipped = true;               // playback starts: configured size, then grow
            emit_chunk = first_chunk;
            return;
        }
        emit_chunk = std::min(max_chunk, std::max(emit_chunk + 1, emit_chunk * 3 / 2));
    };
    // Decodes and delivers every frame not emitted yet.
    auto emit_pending = [&]() -> bool {
        const int total = frames(), nf = total - frames_emitted;
        if (nf <= 0) return true;
        BenchScope bs(&bench, "decode+emit");
        std::vector<float> unscaled(static_cast<size_t>(nf) * latent);
        const float* src = latents.data() + static_cast<size_t>(frames_emitted) * latent;
        for (size_t i = 0; i < unscaled.size(); ++i) unscaled[i] = src[i] / cfg.speech_scaling - cfg.speech_bias;
        std::vector<float> audio;
        if (!model.decoder.decode(dec, unscaled.data(), nf, &audio)) return false;
        const size_t off = first_emit ? std::min(audio.size(), static_cast<size_t>(kWarmupSamples)) : 0;
        first_emit     = false;
        frames_emitted = total;
        ++emits;
        return deliver(audio.data() + off, static_cast<int>(audio.size() - off), false);
    };

    // ---- text ----
    std::vector<int32_t> text;
    int  text_pos = 0;
    bool text_eof = false;
    auto ensure_text = [&](int want) {
        while (!text_eof && static_cast<int>(text.size()) < want) {
            std::vector<int32_t> more;
            bool eof = false;
            io.pull_text(want - static_cast<int>(text.size()), &more, &eof);
            text.insert(text.end(), more.begin(), more.end());
            text_eof = eof;
            if (more.empty() && !eof) break;   // a non-blocking source with nothing yet
        }
    };

    int  rc = LV_OK, lm_pos = voice.lm.seq, tlm_pos = voice.tlm.seq;
    bool finished = false, stopped = false;
    std::vector<float> z(latent), emb, lm_hidden;
    FrameOutputs fo;

    while (!finished && !stopped && frames() < max_frames) {
        if (aborted_by_host()) { stopped = true; break; }
        ensure_text(text_pos + kTextWindow);
        if (text.empty() && text_eof) break;   // nothing to say

        // ---- text window ----
        if (text_pos < static_cast<int>(text.size())) {
            BenchScope bs(&bench, "text window");
            const int win = std::min(kTextWindow, static_cast<int>(text.size()) - text_pos);
            if (lm_pos + win > kv_lm.max_seq || tlm_pos + win > kv_tlm.max_seq) {
                LV_LOG_ERROR("generate: text exceeds max_text_tokens (%d)", p.max_text_tokens);
                rc = LV_ERR_CAPACITY;
                break;
            }
            if (!model.embed(text.data() + text_pos, win, &emb)) { rc = LV_ERR_GENERIC; break; }
            if (!run_stack(model, model.lm, nullptr, lm_pos, win, emb.data(), kv_lm, &lm_hidden, nullptr)) {
                rc = LV_ERR_COMPUTE;
                break;
            }
            for (int t = 0; t < win; ++t)
                for (int i = 0; i < hidden; ++i) lm_hidden[static_cast<size_t>(t) * hidden + i] += model.type_text[i];
            if (!run_stack(model, model.tlm, model.tlm_norm, tlm_pos, win, lm_hidden.data(), kv_tlm, nullptr, &hid_pos)) {
                rc = LV_ERR_COMPUTE;
                break;
            }
            lm_pos += win;
            tlm_pos += win;
            text_pos += win;
            LV_LOG_DEBUG("generate: text %d/%zu tokens", text_pos, text.size());
        }

        // ---- speech window ----
        const int budget = std::min(kSpeechWindow, max_frames - frames());
        for (int sp = 0; sp < budget; ++sp) {
            if (aborted_by_host()) { stopped = true; break; }
            const int frame = frames();
            for (auto& v : z) v = normal(rng);
            if (guided)
                for (size_t i = 0; i < neg_cond.size(); ++i)
                    neg_cond[i] = hid_neg[i] * (1.0f - anchor_w) + neg_anchor[i] * anchor_w;

            FrameInputs fi;
            fi.z0 = z.data();
            fi.cond_pos = hid_pos.data();
            fi.cond_neg = guided ? neg_cond.data() : nullptr;
            fi.cfg_scale = p.cfg_scale;
            fi.t_sin = t_sin.data();
            bool ok;
            {
                BenchScope bs(&bench, "frame");
                ok = run_frame(model, sched, kv_tlm, guided ? &kv_neg : nullptr, fi, &fo, &bench);
            }
            if (!ok) {
                rc = LV_ERR_COMPUTE;
                stopped = true;
                break;
            }
            z = fo.latent;
            hid_pos = fo.hidden_pos;
            ++tlm_pos;
            if (guided) hid_neg = fo.hidden_neg;
            const float eos = 1.0f / (1.0f + std::exp(-fo.eos_logit));
            latents.insert(latents.end(), z.begin(), z.end());

            float ss = 0.0f;
            for (float v : z) ss += v * v;
            const float nrm = std::sqrt(ss);
            if (frame >= 1 && frame <= 3)   // the lead-in; frame 0 sometimes carries a click
                silent_ref = silent_ref == 0.0f ? nrm : std::min(silent_ref, nrm);
            float thresh = kSilentNormFloor;
            if (silent_ref > 0.0f && silent_ref < kSilentNormFloor) thresh = std::max(thresh, kSilentNormRatio * silent_ref);
            if (nrm < thresh) {
                ++silent_run;
            } else {
                silent_run = 0;
                ++pending_sound;
                if (frame >= 1) ++pending_speech;
            }
            LV_LOG_DEBUG("generate: frame %d, |z| %.2f, eos %.3f", frame, static_cast<double>(nrm), static_cast<double>(eos));
            if (eos > 0.5f) {
                finished = true;
                break;
            }

            if (frames() - frames_emitted >= emit_chunk) {
                if (!emit_pending()) { stopped = true; break; }
                advance_chunk();   // reads pending_speech, so before the reset
                pending_sound = pending_speech = 0;
            } else if (p.flush_on_silence && silent_run >= kTailSilentFrames && pending_sound > 0) {
                if (!emit_pending()) { stopped = true; break; }
                pending_sound = pending_speech = 0;
            }
        }
    }

    if (!stopped && rc == LV_OK) {
        bool ok = emit_pending();
        if (ok && use_pf) ok = deliver(nullptr, 0, true);   // lookahead and overlap-add tail
        if (ok && !lead_done && !lead_keep.empty()) ok = io.emit_audio(lead_keep.data(), static_cast<int>(lead_keep.size()));
        stopped = !ok;
    }
    if (bench_enabled()) {
        char title[160];
        std::snprintf(title, sizeof(title), "%d frames, %d emits, %d steps, %s%s", frames(), emits, sched.steps(),
                      backend_name(), stopped ? " (stopped)" : "");
        bench.report(title);
    }
    if (const char* dump = std::getenv("LIVEVOICE_DUMP_LATENTS")) {
        if (FILE* f = std::fopen(dump, "wb")) {
            std::fwrite(latents.data(), sizeof(float), latents.size(), f);
            std::fclose(f);
        }
    }
    LV_LOG_DEBUG("generate: %d frames, %d emits%s", frames(), emits, finished ? ", EOS" : "");
    return rc;
}

int synthesize(const Model& model, const Voice& voice, const std::string& text, const lv_params& p,
               const std::function<bool(const float* pcm, int n)>& on_audio) {
    const std::vector<int32_t> ids = model.tokenizer.encode(terminate_text(text), encode_options(p));
    if (ids.size() <= 1) return LV_ERR_INVALID_ARG;   // nothing but the terminator
    lv_params pp = p;
    pp.max_text_tokens = std::max(pp.max_text_tokens, static_cast<int>(ids.size()) + 8);
    size_t pos = 0;
    GenerateIO io;
    io.pull_text = [&](int max_tokens, std::vector<int32_t>* out, bool* eof) {
        const size_t n = std::min(static_cast<size_t>(std::max(0, max_tokens)), ids.size() - pos);
        out->insert(out->end(), ids.begin() + static_cast<long>(pos), ids.begin() + static_cast<long>(pos + n));
        pos += n;
        *eof = pos >= ids.size();
    };
    io.emit_audio = on_audio;
    return generate(model, voice, pp, io);
}

}  // namespace lv

extern "C" void lv_params_init(lv_params* p) {
    if (!p) return;
    *p = lv_params{};
    p->n_steps              = 5;
    p->cfg_scale            = 1.5f;
    p->seed                 = 0;
    p->max_frames           = 1500;
    p->max_text_tokens      = 1024;
    p->first_chunk_frames   = 3;
    p->max_chunk_frames     = 32;
    p->lead_chunk_frames    = 0;
    p->flush_on_silence     = 1;
    p->neg_anchor           = 0.2f;
    p->postfilter           = 1;
    p->trim_lead            = 1;
    p->merge_cap            = 80000;
    p->canonical_first_word = 1;
    p->last_word_merge_cap  = 20000;
}
