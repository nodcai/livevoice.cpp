#ifndef LIVEVOICE_QWEN2_HPP
#define LIVEVOICE_QWEN2_HPP

// Qwen2 transformer layers with a resident K/V cache.
//
//   h = x + Attn(RMSNorm(x))      GQA, NeoX RoPE on Q/K, biased q/k/v
//   y = h + FFN(RMSNorm(h))       SwiGLU
//
// The K/V cache lives in a backend buffer for the whole utterance. A graph
// writes the new rows into it with ggml_cpy and reads the full history
// through views, so no step round-trips the cache through the host.

#include "gguf_loader.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <string>
#include <vector>

namespace lv {

constexpr int kRopeMode = GGML_ROPE_TYPE_NEOX;

struct Qwen2Hparams {
    int   hidden       = 0;
    int   n_heads      = 0;
    int   n_kv_heads   = 0;
    int   head_dim     = 0;
    float rope_theta   = 1.0e6f;
    float rms_norm_eps = 1.0e-6f;
    bool  flash_attn   = false;   // ggml_flash_attn_ext; the mask must then be F16
};

struct Qwen2LayerWeights {
    struct ggml_tensor* attn_norm   = nullptr;
    struct ggml_tensor* attn_q      = nullptr;
    struct ggml_tensor* attn_q_bias = nullptr;
    struct ggml_tensor* attn_k      = nullptr;
    struct ggml_tensor* attn_k_bias = nullptr;
    struct ggml_tensor* attn_v      = nullptr;
    struct ggml_tensor* attn_v_bias = nullptr;
    struct ggml_tensor* attn_o      = nullptr;
    struct ggml_tensor* ffn_norm    = nullptr;
    struct ggml_tensor* ffn_gate    = nullptr;
    struct ggml_tensor* ffn_up      = nullptr;
    struct ggml_tensor* ffn_down    = nullptr;
};

// `<prefix>blk.<i>.{attn_norm,attn_q,...}.weight` for i in [0, n_layers).
bool load_qwen2_stack(const GgufLoader& m, const std::string& prefix, int n_layers,
                      std::vector<Qwen2LayerWeights>* out);

// Per-layer K/V of shape [head_dim, n_kv, max_seq, 1].
struct ResidentKV {
    struct ggml_context*             ctx    = nullptr;
    ggml_backend_buffer_t            buffer = nullptr;
    std::vector<struct ggml_tensor*> k, v;
    int       max_seq  = 0;
    int       past_len = 0;
    ggml_type type     = GGML_TYPE_F16;

    ResidentKV() = default;
    ResidentKV(const ResidentKV&)            = delete;
    ResidentKV& operator=(const ResidentKV&) = delete;
    ~ResidentKV() { free(); }

    bool init(int n_layers, int head_dim, int n_kv, int max_seq, ggml_type type = GGML_TYPE_F16);
    void free();
};

inline struct ggml_tensor* rms_norm(struct ggml_context* g, struct ggml_tensor* x,
                                    struct ggml_tensor* w, float eps) {
    return ggml_mul(g, ggml_rms_norm(g, x, eps), w);
}

// x: [head_dim, n_heads, n_tokens, B]; pos: I32 [n_tokens].
struct ggml_tensor* qwen2_rope(struct ggml_context* g, struct ggml_tensor* x,
                               struct ggml_tensor* pos, const Qwen2Hparams& hp);

// Attention core.
//   q_p:      [head_dim, n_tokens, n_heads, B]   (permuted view)
//   k_p, v_p: [head_dim, kv_len, n_kv, B]        (permuted cache views)
//   mask:     additive [kv_len, n_tokens], or null for a single query
// Returns [n_heads*head_dim, n_tokens*B], contiguous.
struct ggml_tensor* qwen2_attention(struct ggml_context* g, struct ggml_tensor* q_p,
                                    struct ggml_tensor* k_p, struct ggml_tensor* v_p,
                                    struct ggml_tensor* mask, const Qwen2Hparams& hp);

struct Qwen2LayerOut {
    struct ggml_tensor* y       = nullptr;   // [hidden, n_tokens, B]
    struct ggml_tensor* k_write = nullptr;   // ggml_cpy into the cache
    struct ggml_tensor* v_write = nullptr;
};

// One layer over `x` [hidden, n_tokens, B] at absolute positions `pos`. The
// new K/V go to slots [kv.past_len, kv.past_len + n_tokens) of layer
// `layer`; mask is [kv.past_len + n_tokens, n_tokens]. The caller expands
// k_write / v_write into the graph before the layer's output, so the writes
// land before the attention reads the cache, and advances kv.past_len after
// the graph ran.
Qwen2LayerOut qwen2_layer(struct ggml_context* g, struct ggml_tensor* x, struct ggml_tensor* pos,
                          struct ggml_tensor* mask, ResidentKV& kv, int layer,
                          const Qwen2LayerWeights& w, const Qwen2Hparams& hp);

}  // namespace lv

#endif  // LIVEVOICE_QWEN2_HPP
