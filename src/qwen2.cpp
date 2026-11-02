#include "qwen2.hpp"

#include "backend.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdio>

namespace lv {

namespace {

struct ggml_tensor* linear(struct ggml_context* g, struct ggml_tensor* w, struct ggml_tensor* x,
                           struct ggml_tensor* b) {
    struct ggml_tensor* y = ggml_mul_mat(g, w, x);
    return b ? ggml_add(g, y, b) : y;
}

}  // namespace

bool load_qwen2_stack(const GgufLoader& m, const std::string& prefix, int n_layers,
                      std::vector<Qwen2LayerWeights>* out) {
    out->assign(n_layers, {});
    for (int i = 0; i < n_layers; ++i) {
        char p[64];
        std::snprintf(p, sizeof(p), "%sblk.%d.", prefix.c_str(), i);
        auto get = [&](const char* n) { return m.tensor(std::string(p) + n); };
        Qwen2LayerWeights& l = (*out)[i];
        l.attn_norm   = get("attn_norm.weight");
        l.attn_q      = get("attn_q.weight");
        l.attn_q_bias = get("attn_q.bias");
        l.attn_k      = get("attn_k.weight");
        l.attn_k_bias = get("attn_k.bias");
        l.attn_v      = get("attn_v.weight");
        l.attn_v_bias = get("attn_v.bias");
        l.attn_o      = get("attn_o.weight");
        l.ffn_norm    = get("ffn_norm.weight");
        l.ffn_gate    = get("ffn_gate.weight");
        l.ffn_up      = get("ffn_up.weight");
        l.ffn_down    = get("ffn_down.weight");
        if (!l.attn_norm || !l.attn_q || !l.attn_k || !l.attn_v || !l.attn_o ||
            !l.ffn_norm || !l.ffn_gate || !l.ffn_up || !l.ffn_down) {
            LV_LOG_ERROR("qwen2: layer %s is incomplete", p);
            return false;
        }
    }
    return true;
}

bool ResidentKV::init(int n_layers, int head_dim, int n_kv, int max_seq_in, ggml_type type_in) {
    free();
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * (2 * n_layers + 1);
    p.no_alloc = true;
    ctx = ggml_init(p);
    if (!ctx) return false;
    k.assign(n_layers, nullptr);
    v.assign(n_layers, nullptr);
    for (int i = 0; i < n_layers; ++i) {
        k[i] = ggml_new_tensor_4d(ctx, type_in, head_dim, n_kv, max_seq_in, 1);
        v[i] = ggml_new_tensor_4d(ctx, type_in, head_dim, n_kv, max_seq_in, 1);
    }
    buffer = allocate_ctx_tensors(ctx);
    if (!buffer) {
        free();
        return false;
    }
    max_seq  = max_seq_in;
    past_len = 0;
    type     = type_in;
    return true;
}

void ResidentKV::free() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    buffer = nullptr;
    ctx    = nullptr;
    k.clear();
    v.clear();
    max_seq = past_len = 0;
}

struct ggml_tensor* qwen2_rope(struct ggml_context* g, struct ggml_tensor* x,
                               struct ggml_tensor* pos, const Qwen2Hparams& hp) {
    return ggml_rope_ext(g, x, pos, nullptr, hp.head_dim, kRopeMode, 0, hp.rope_theta,
                         1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

struct ggml_tensor* qwen2_attention(struct ggml_context* g, struct ggml_tensor* q_p,
                                    struct ggml_tensor* k_p, struct ggml_tensor* v_p,
                                    struct ggml_tensor* mask, const Qwen2Hparams& hp) {
    const int64_t n_tokens = q_p->ne[1];
    const int64_t n_batch  = q_p->ne[3];
    const int64_t width    = static_cast<int64_t>(hp.n_heads) * hp.head_dim;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(hp.head_dim));
    struct ggml_tensor* o;
    if (hp.flash_attn) {
        // F16 cache views go in as they are; an F32 cache is cast.
        struct ggml_tensor* k = k_p->type == GGML_TYPE_F32 ? ggml_cast(g, k_p, GGML_TYPE_F16) : k_p;
        struct ggml_tensor* v = v_p->type == GGML_TYPE_F32 ? ggml_cast(g, v_p, GGML_TYPE_F16) : v_p;
        o = ggml_flash_attn_ext(g, q_p, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    } else {
        struct ggml_tensor* scores = ggml_mul_mat(g, k_p, q_p);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        struct ggml_tensor* attn = ggml_soft_max_ext(g, scores, mask, scale, 0.0f);
        o = ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, v_p)), attn);
        o = ggml_permute(g, o, 0, 2, 1, 3);
    }
    return ggml_cont_2d(g, o, width, n_tokens * n_batch);
}

Qwen2LayerOut qwen2_layer(struct ggml_context* g, struct ggml_tensor* x, struct ggml_tensor* pos,
                          struct ggml_tensor* mask, ResidentKV& kv, int layer,
                          const Qwen2LayerWeights& w, const Qwen2Hparams& hp) {
    const int64_t n_tokens = x->ne[1];
    const int64_t n_batch  = x->ne[2];
    const int     hd = hp.head_dim, n_h = hp.n_heads, n_kv = hp.n_kv_heads;
    const int     full = kv.past_len + static_cast<int>(n_tokens);
    GGML_ASSERT(full <= kv.max_seq);

    struct ggml_tensor* xn = rms_norm(g, x, w.attn_norm, hp.rms_norm_eps);
    struct ggml_tensor* q  = ggml_reshape_4d(g, linear(g, w.attn_q, xn, w.attn_q_bias), hd, n_h,  n_tokens, n_batch);
    struct ggml_tensor* k  = ggml_reshape_4d(g, linear(g, w.attn_k, xn, w.attn_k_bias), hd, n_kv, n_tokens, n_batch);
    struct ggml_tensor* v  = ggml_reshape_4d(g, linear(g, w.attn_v, xn, w.attn_v_bias), hd, n_kv, n_tokens, n_batch);
    q = qwen2_rope(g, q, pos, hp);
    k = qwen2_rope(g, k, pos, hp);

    struct ggml_tensor* kc = kv.k[layer];
    struct ggml_tensor* vc = kv.v[layer];
    auto slots = [&](struct ggml_tensor* c, int64_t first, int64_t n) {
        return ggml_view_4d(g, c, hd, n_kv, n, n_batch, c->nb[1], c->nb[2], c->nb[3],
                            static_cast<size_t>(first) * c->nb[2]);
    };
    Qwen2LayerOut out;
    out.k_write = ggml_cpy(g, k, slots(kc, kv.past_len, n_tokens));
    out.v_write = ggml_cpy(g, v, slots(vc, kv.past_len, n_tokens));

    struct ggml_tensor* o = qwen2_attention(g, ggml_permute(g, q, 0, 2, 1, 3),
                                            ggml_permute(g, slots(kc, 0, full), 0, 2, 1, 3),
                                            ggml_permute(g, slots(vc, 0, full), 0, 2, 1, 3), mask, hp);
    if (n_batch > 1) o = ggml_reshape_3d(g, o, static_cast<int64_t>(n_h) * hd, n_tokens, n_batch);

    struct ggml_tensor* h  = ggml_add(g, x, ggml_mul_mat(g, w.attn_o, o));
    struct ggml_tensor* hn = rms_norm(g, h, w.ffn_norm, hp.rms_norm_eps);
    struct ggml_tensor* f  = ggml_mul_mat(g, w.ffn_down,
                                          ggml_swiglu_split(g, ggml_mul_mat(g, w.ffn_gate, hn),
                                                            ggml_mul_mat(g, w.ffn_up, hn)));
    out.y = ggml_add(g, h, f);
    return out;
}

}  // namespace lv
