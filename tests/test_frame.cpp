// The per-frame graph against the diffusion sampler alone. Needs
// LIVEVOICE_MODEL and LIVEVOICE_VOICE; skips otherwise.
//
// From the voice prompt's state, one frame must produce the same latent as
// a graph with nothing but dpm_sample on the same inputs: the frame graph
// adds the connector, both TTS-LM columns and the EOS head after it, and
// none of that may disturb the latent it returns.
#include "backend.hpp"
#include "frame_graph.hpp"
#include "model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    const char* model_path = std::getenv("LIVEVOICE_MODEL");
    const char* voice_path = std::getenv("LIVEVOICE_VOICE");
    if (!model_path || !*model_path || !voice_path || !*voice_path) {
        std::fprintf(stderr, "skip: set LIVEVOICE_MODEL and LIVEVOICE_VOICE\n");
        return 77;
    }
    lv::Model model;
    lv::Voice voice;
    if (!model.load(model_path) || !voice.load(voice_path, model) || !voice.has_neg) {
        std::fprintf(stderr, "FAIL: load (the voice needs a negative branch)\n");
        return 1;
    }
    const auto& c = model.cfg;
    const float cfg_scale = 1.5f;
    lv::DpmSchedule sched;
    sched.init(5);
    const int freq = model.head_cfg.freq_size;
    std::vector<float> t_sin(static_cast<size_t>(freq) * sched.steps());
    for (int i = 0; i < sched.steps(); ++i)
        lv::timestep_sinusoidal(static_cast<float>(sched.timesteps[i]), freq, &t_sin[static_cast<size_t>(i) * freq]);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> z(c.latent);
    for (auto& v : z) v = nd(rng);

    // ---- the frame ----
    lv::ResidentKV kv_pos, kv_neg;
    kv_pos.init(c.n_layers_tlm, c.head_dim, c.n_kv_heads, voice.tlm.seq + 4);
    kv_neg.init(c.n_layers_tlm, c.head_dim, c.n_kv_heads, voice.neg_tlm.seq + 4);
    for (int pass = 0; pass < 2; ++pass) {
        lv::ResidentKV& kv = pass ? kv_neg : kv_pos;
        const lv::Voice::Stack& s = pass ? voice.neg_tlm : voice.tlm;
        for (size_t l = 0; l < s.k.size(); ++l) {
            ggml_backend_tensor_set(kv.k[l], s.k[l].data(), 0, s.k[l].size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(kv.v[l], s.v[l].data(), 0, s.v[l].size() * sizeof(ggml_fp16_t));
        }
        kv.past_len = s.seq;
    }
    lv::FrameInputs fi;
    fi.z0 = z.data();
    fi.cond_pos = voice.tlm_hidden.data();
    fi.cond_neg = voice.neg_tlm_hidden.data();
    fi.cfg_scale = cfg_scale;
    fi.t_sin = t_sin.data();
    lv::FrameOutputs fo;
    if (!lv::run_frame(model, sched, kv_pos, &kv_neg, fi, &fo, nullptr)) { std::fprintf(stderr, "FAIL: frame\n"); return 2; }

    // ---- the sampler alone ----
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * 4;
    ip.no_alloc = true;
    ggml_context* ictx = ggml_init(ip);
    ggml_tensor* tz = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, c.latent, 1);
    ggml_tensor* tc = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, c.hidden, 2);
    ggml_tensor* tt = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, freq, sched.steps());
    ggml_backend_buffer_t ibuf = lv::allocate_ctx_tensors(ictx);
    ggml_backend_tensor_set(tz, z.data(), 0, ggml_nbytes(tz));
    ggml_backend_tensor_set(tc, voice.tlm_hidden.data(), 0, sizeof(float) * c.hidden);
    ggml_backend_tensor_set(tc, voice.neg_tlm_hidden.data(), sizeof(float) * c.hidden, sizeof(float) * c.hidden);
    ggml_backend_tensor_set(tt, t_sin.data(), 0, ggml_nbytes(tt));
    ip.mem_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    ggml_context* g = ggml_init(ip);
    ggml_tensor* y = lv::dpm_sample(g, tz, tc, tt, cfg_scale, sched, model.head, model.head_cfg);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);
    ggml_build_forward_expand(gf, y);
    std::vector<float> want(c.latent);
    if (!lv::compute_graph(gf)) return 3;
    ggml_backend_tensor_get(y, want.data(), 0, sizeof(float) * c.latent);

    double d = 0, r = 0;
    for (int i = 0; i < c.latent; ++i) {
        d += (fo.latent[i] - want[i]) * (fo.latent[i] - want[i]);
        r += want[i] * want[i];
    }
    const double rel = std::sqrt(d / r);
    const float eos = 1.0f / (1.0f + std::exp(-fo.eos_logit));
    std::printf("latent rel err %.2e, |z| %.2f, eos %.3f, positions %d / %d\n", rel, std::sqrt(r), eos, kv_pos.past_len,
                kv_neg.past_len);
    ggml_free(g);
    ggml_backend_buffer_free(ibuf);
    ggml_free(ictx);
    if (rel > 1e-4) { std::fprintf(stderr, "FAIL: the frame's latent differs from the sampler's\n"); return 4; }
    if (eos > 0.5f) { std::fprintf(stderr, "FAIL: EOS on the first frame\n"); return 5; }
    if (kv_pos.past_len != voice.tlm.seq + 1 || kv_neg.past_len != voice.neg_tlm.seq + 1) return 6;
    std::printf("OK\n");
    return 0;
}
