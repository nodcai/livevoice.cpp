#ifndef LIVEVOICE_MODEL_HPP
#define LIVEVOICE_MODEL_HPP

// VibeVoice-Realtime-0.5B weights and voice prompts.
//
//   text ids -> token embedding -> LM (4 lower Qwen2 layers, no final norm)
//            -> + text type embedding -> TTS-LM (20 upper layers, final norm)
//   per speech frame:
//     TTS-LM last hidden -> diffusion head (DPM-Solver++, CFG) -> 64-d latent
//     latent -> acoustic connector + speech type embedding -> TTS-LM step
//     TTS-LM hidden -> EOS classifier
//   latents -> acoustic decoder -> 24 kHz -> [DFN3 post-filter at 48 kHz]
//
// One gguf carries all of it, the tokenizer and, optionally, the post-filter.
// A voice is a separate gguf with the K/V caches of a pre-run voice prompt.

#include "acoustic_decoder.hpp"
#include "dfn.hpp"
#include "diffusion.hpp"
#include "gguf_loader.hpp"
#include "qwen2.hpp"
#include "tokenizer.hpp"

#include <string>
#include <vector>

namespace lv {

struct ModelConfig {
    int   hidden       = 0;
    int   n_layers_lm  = 0;
    int   n_layers_tlm = 0;
    int   n_heads      = 0;
    int   n_kv_heads   = 0;
    int   head_dim     = 0;
    int   vocab_size   = 0;
    float rope_theta   = 1.0e6f;
    float rms_norm_eps = 1.0e-6f;
    int   latent       = 64;
    int   sample_rate  = 24000;
    // Latents are generated in a normalised space: decode(z / scaling - bias).
    float speech_scaling = 1.0f;
    float speech_bias    = 0.0f;
};

struct Model {
    GgufLoader  loader;
    ModelConfig cfg;
    Tokenizer   tokenizer;

    struct ggml_tensor*            tok_embd = nullptr;
    // A pruned embedding table holds only the rows the tokenizer can produce
    // under embed_merge_cap: embed_row maps a token id to its row (-1 = not
    // stored). Empty and 0 for a full table.
    std::vector<int32_t>           embed_row;
    int                            embed_merge_cap = 0;
    std::vector<Qwen2LayerWeights> lm, tlm;
    struct ggml_tensor*            tlm_norm = nullptr;
    // tts_input_types rows (0 = speech, 1 = text), host copies.
    std::vector<float> type_speech, type_text;

    struct ggml_tensor* ac_fc1_w = nullptr, *ac_fc1_b = nullptr, *ac_norm = nullptr;
    struct ggml_tensor* ac_fc2_w = nullptr, *ac_fc2_b = nullptr;
    struct ggml_tensor* eos_fc1_w = nullptr, *eos_fc1_b = nullptr;
    struct ggml_tensor* eos_fc2_w = nullptr, *eos_fc2_b = nullptr;

    DiffusionHeadConfig  head_cfg;
    DiffusionHeadWeights head;
    AcousticDecoder      decoder;
    DfnModel             dfn;   // dfn.ready when the file carries the post-filter

    bool load(const std::string& path);
    Qwen2Hparams hparams() const;
    // Embedding rows for `ids` ([hidden * n]); the table may be quantized
    // and pruned.
    bool embed(const int32_t* ids, int n, std::vector<float>* out) const;
};

// K/V caches of a voice prompt, as F16 [head_dim, n_kv, seq] per layer.
struct Voice {
    struct Stack {
        int seq = 0;
        std::vector<std::vector<ggml_fp16_t>> k, v;
    };
    Stack lm, tlm;
    std::vector<float> tlm_hidden;       // TTS-LM last hidden after the prompt
    // Negative branch for classifier-free guidance. Only its TTS-LM part is
    // used: the negative sequence never sees the text.
    bool  has_neg = false;
    Stack neg_tlm;
    std::vector<float> neg_tlm_hidden;

    bool load(const std::string& path, const Model& model);
};

}  // namespace lv

#endif  // LIVEVOICE_MODEL_HPP
