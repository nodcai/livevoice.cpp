#include "model.hpp"

#include "backend.hpp"
#include "common.hpp"

#include <cstdio>
#include <cstring>

namespace lv {

namespace {

// The gguf metadata keys follow the converter (scripts/convert_model.py).
constexpr const char* kVariant = "realtime-0.5b";

}  // namespace

bool Model::load(const std::string& path) {
    // Uploads and the decoder's re-layout touch the backend; keep them out
    // of a generation running on another thread.
    std::lock_guard<std::mutex> lock(compute_mutex());
    if (!loader.load(path)) return false;
    auto& m = loader;
    m.promote_small_f16_to_f32();

    const std::string variant = m.get_str("vibevoice.variant");
    if (variant != kVariant) {
        LV_LOG_ERROR("model: %s is variant '%s', expected '%s'", path.c_str(), variant.c_str(), kVariant);
        return false;
    }
    if (m.get_str("tokenizer.model").empty() || !tokenizer.load(m)) {
        LV_LOG_ERROR("model: %s carries no tokenizer; add one with scripts/merge_gguf.py --tokenizer", path.c_str());
        return false;
    }

    cfg.hidden       = m.get_i32("vibevoice.hidden");
    cfg.n_layers_lm  = m.get_i32("vibevoice.n_layers_lm");
    cfg.n_layers_tlm = m.get_i32("vibevoice.n_layers_tlm");
    cfg.n_heads      = m.get_i32("vibevoice.n_heads");
    cfg.n_kv_heads   = m.get_i32("vibevoice.n_kv_heads");
    cfg.head_dim     = m.get_i32("vibevoice.head_dim");
    cfg.vocab_size   = m.get_i32("vibevoice.vocab_size");
    cfg.rope_theta   = m.get_f32("vibevoice.rope_theta", 1.0e6f);
    cfg.rms_norm_eps = m.get_f32("vibevoice.rms_norm_eps", 1.0e-6f);
    cfg.latent       = m.get_i32("vibevoice.diffusion.latent", 64);
    cfg.sample_rate  = m.get_i32("vibevoice.sample_rate", 24000);
    if (cfg.hidden <= 0 || cfg.n_layers_lm <= 0 || cfg.n_layers_tlm <= 0 || cfg.head_dim <= 0) {
        LV_LOG_ERROR("model: incomplete vibevoice.* metadata");
        return false;
    }

    tok_embd    = m.tensor("lm.tok_embd.weight");
    tlm_norm    = m.tensor("tlm.output_norm.weight");
    auto* types = m.tensor("tts.input_types.weight");
    ac_fc1_w  = m.tensor("ac.fc1.weight");
    ac_fc1_b  = m.tensor("ac.fc1.bias");
    ac_norm   = m.tensor("ac.norm.weight");
    ac_fc2_w  = m.tensor("ac.fc2.weight");
    ac_fc2_b  = m.tensor("ac.fc2.bias");
    eos_fc1_w = m.tensor("eos.fc1.weight");
    eos_fc1_b = m.tensor("eos.fc1.bias");
    eos_fc2_w = m.tensor("eos.fc2.weight");
    eos_fc2_b = m.tensor("eos.fc2.bias");
    if (!tok_embd || !tlm_norm || !types || !ac_fc1_w || !ac_norm || !ac_fc2_w || !eos_fc1_w || !eos_fc2_w) {
        LV_LOG_ERROR("model: missing embedding, connector, EOS or input-type tensors");
        return false;
    }
    if (!load_qwen2_stack(m, "lm.", cfg.n_layers_lm, &lm) || !load_qwen2_stack(m, "tlm.", cfg.n_layers_tlm, &tlm))
        return false;
    if (const struct ggml_tensor* ids = m.tensor("lm.tok_embd.ids")) {
        embed_merge_cap = m.get_i32("vibevoice.embed_merge_cap");
        std::vector<int32_t> kept(static_cast<size_t>(ggml_nelements(ids)));
        if (ids->type != GGML_TYPE_I32 || embed_merge_cap <= 0 || static_cast<int64_t>(kept.size()) != tok_embd->ne[1]) {
            LV_LOG_ERROR("model: pruned embedding without a matching id list or merge cap");
            return false;
        }
        ggml_backend_tensor_get(ids, kept.data(), 0, kept.size() * sizeof(int32_t));
        embed_row.assign(cfg.vocab_size, -1);
        for (size_t r = 0; r < kept.size(); ++r)
            if (kept[r] >= 0 && kept[r] < cfg.vocab_size) embed_row[kept[r]] = static_cast<int32_t>(r);
    }
    {
        const std::vector<float> t = tensor_to_f32(types);
        if (t.size() != static_cast<size_t>(2) * cfg.hidden) {
            LV_LOG_ERROR("model: tts.input_types is not [hidden, 2]");
            return false;
        }
        type_speech.assign(t.begin(), t.begin() + cfg.hidden);
        type_text.assign(t.begin() + cfg.hidden, t.end());
    }

    head_cfg.hidden    = cfg.hidden;
    head_cfg.latent    = cfg.latent;
    head_cfg.layers    = m.get_i32("vibevoice.diffusion.head_layers", 4);
    head_cfg.freq_size = 256;
    head_cfg.eps       = cfg.rms_norm_eps;
    if (!load_diffusion_head(m, "dh.", head_cfg.layers, &head)) return false;
    if (!decoder.load(m, "at.dec", m.get_f32("vibevoice.acoustic.eps", 1e-5f))) return false;
    if (decoder.latent_dim() != cfg.latent) {
        LV_LOG_ERROR("model: decoder takes %d-d latents, the head makes %d", decoder.latent_dim(), cfg.latent);
        return false;
    }

    auto scalar = [](const struct ggml_tensor* t, float def) {
        if (!t) return def;
        const std::vector<float> v = tensor_to_f32(t);
        return v.empty() ? def : v[0];
    };
    cfg.speech_scaling = scalar(m.tensor("speech.scaling"), 1.0f);
    cfg.speech_bias    = scalar(m.tensor("speech.bias"), 0.0f);

    if (dfn_model_present(m) && !dfn_model_load(m, &dfn))
        LV_LOG_WARN("model: post-filter tensors present but unusable; continuing without it");

    LV_LOG_INFO("model: hidden %d, layers %d+%d, vocab %d, post-filter %s", cfg.hidden, cfg.n_layers_lm,
                cfg.n_layers_tlm, cfg.vocab_size, dfn.ready ? "yes" : "no");
    return true;
}

Qwen2Hparams Model::hparams() const {
    Qwen2Hparams hp;
    hp.hidden       = cfg.hidden;
    hp.n_heads      = cfg.n_heads;
    hp.n_kv_heads   = cfg.n_kv_heads;
    hp.head_dim     = cfg.head_dim;
    hp.rope_theta   = cfg.rope_theta;
    hp.rms_norm_eps = cfg.rms_norm_eps;
    return hp;
}

bool Model::embed(const int32_t* ids, int n, std::vector<float>* out) const {
    const int    hidden = cfg.hidden;
    const size_t row    = ggml_row_size(tok_embd->type, hidden);
    const auto*  tt     = ggml_get_type_traits(tok_embd->type);
    if (tok_embd->type != GGML_TYPE_F32 && (!tt || !tt->to_float)) {
        LV_LOG_ERROR("model: token embedding type %s is not supported", ggml_type_name(tok_embd->type));
        return false;
    }
    out->assign(static_cast<size_t>(hidden) * n, 0.0f);
    std::vector<uint8_t> raw(row);
    for (int t = 0; t < n; ++t) {
        if (ids[t] < 0 || ids[t] >= cfg.vocab_size) {
            LV_LOG_ERROR("model: token id %d out of range", ids[t]);
            return false;
        }
        const int32_t r = embed_row.empty() ? ids[t] : embed_row[ids[t]];
        if (r < 0) {
            LV_LOG_ERROR("model: token id %d is not in this file's pruned embedding (merge cap %d)", ids[t], embed_merge_cap);
            return false;
        }
        float* dst = out->data() + static_cast<size_t>(hidden) * t;
        if (tok_embd->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(tok_embd, dst, row * static_cast<size_t>(r), row);
        } else {
            ggml_backend_tensor_get(tok_embd, raw.data(), row * static_cast<size_t>(r), row);
            tt->to_float(raw.data(), dst, hidden);
        }
    }
    return true;
}

bool Voice::load(const std::string& path, const Model& model) {
    GgufLoader m;
    if (!m.load_host(path)) return false;
    const auto& c = model.cfg;
    const int hidden = m.get_i32("voice.hidden"), head_dim = m.get_i32("voice.head_dim");
    const int n_kv = m.get_i32("voice.n_kv_heads");
    const int n_lm = m.get_i32("voice.lm.n_layers"), n_tlm = m.get_i32("voice.tts_lm.n_layers");
    if (hidden != c.hidden || head_dim != c.head_dim || n_kv != c.n_kv_heads || n_lm != c.n_layers_lm ||
        n_tlm != c.n_layers_tlm) {
        LV_LOG_ERROR("voice: %s does not fit the model (hidden %d/%d, head_dim %d/%d, kv heads %d/%d, layers %d+%d/%d+%d)",
                     path.c_str(), hidden, c.hidden, head_dim, c.head_dim, n_kv, c.n_kv_heads, n_lm, n_tlm,
                     c.n_layers_lm, c.n_layers_tlm);
        return false;
    }

    auto stack = [&](const char* name, int n_layers, Stack* dst) {
        dst->seq = m.get_i32(std::string("voice.") + name + ".seq_len");
        const size_t per = static_cast<size_t>(head_dim) * n_kv * dst->seq;
        dst->k.assign(n_layers, {});
        dst->v.assign(n_layers, {});
        for (int i = 0; i < n_layers; ++i) {
            for (int kv = 0; kv < 2; ++kv) {
                char key[64];
                std::snprintf(key, sizeof(key), "voice.%s.%c.%d", name, kv ? 'v' : 'k', i);
                const struct ggml_tensor* t = m.tensor(key);
                if (!t || static_cast<size_t>(ggml_nelements(t)) != per) {
                    LV_LOG_ERROR("voice: %s missing or misshapen in %s", key, path.c_str());
                    return false;
                }
                const std::vector<float> f = tensor_to_f32(t);
                auto& out = kv ? dst->v[i] : dst->k[i];
                out.resize(per);
                for (size_t j = 0; j < per; ++j) out[j] = ggml_fp32_to_fp16(f[j]);
            }
        }
        return dst->seq > 0;
    };
    // The prompt's last position: column seq-1 of [hidden, seq].
    auto last_hidden = [&](const char* key, int seq, std::vector<float>* dst) {
        const struct ggml_tensor* t = m.tensor(key);
        if (!t || ggml_nelements(t) != static_cast<int64_t>(hidden) * seq) {
            LV_LOG_ERROR("voice: %s missing or misshapen in %s", key, path.c_str());
            return false;
        }
        const std::vector<float> f = tensor_to_f32(t);
        dst->assign(f.end() - hidden, f.end());
        return true;
    };
    if (!stack("lm", n_lm, &lm) || !stack("tts_lm", n_tlm, &tlm) ||
        !last_hidden("voice.tts_lm.last_hidden", tlm.seq, &tlm_hidden))
        return false;
    has_neg = m.get_bool("voice.has_neg", false) && stack("neg_tts_lm", n_tlm, &neg_tlm) &&
              last_hidden("voice.neg_tts_lm.last_hidden", neg_tlm.seq, &neg_tlm_hidden);
    LV_LOG_INFO("voice: %s, prompt %d + %d positions, guidance %s", path.c_str(), lm.seq, tlm.seq,
                has_neg ? "yes" : "no");
    return true;
}

}  // namespace lv
