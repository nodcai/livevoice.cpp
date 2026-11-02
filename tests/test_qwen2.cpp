// Qwen2 layer with the resident K/V cache against a plain C++ reference.
//
// A small random layer runs over a 7-token sequence three ways: one graph
// for all tokens, the same split into chunks of 3 + 4 (the second chunk must
// see the first chunk's K/V through the cache), and token by token. Each is
// compared with a direct implementation of the layer formula. Runs on the
// active backend, with flash attention and an F16 cache where available and
// with the eager path and an F32 cache.
#include "backend.hpp"
#include "qwen2.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kHidden = 128, kHeads = 4, kKv = 2, kHd = 64, kFfn = 192, kSeq = 7;
constexpr float kEps = 1e-6f, kTheta = 10000.0f;

struct HostLayer {
    std::vector<float> attn_norm, q, qb, k, kb, v, vb, o, ffn_norm, gate, up, down;
};

// y = W x with W stored row-major [out][in] (ggml ne0 = in).
std::vector<float> matvec(const std::vector<float>& W, const float* x, int in, int out) {
    std::vector<float> y(out, 0.0f);
    for (int r = 0; r < out; ++r)
        for (int c = 0; c < in; ++c) y[r] += W[static_cast<size_t>(r) * in + c] * x[c];
    return y;
}

std::vector<float> rmsnorm(const float* x, const std::vector<float>& w) {
    double ss = 0;
    for (int i = 0; i < kHidden; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float s = 1.0f / std::sqrt(static_cast<float>(ss / kHidden) + kEps);
    std::vector<float> y(kHidden);
    for (int i = 0; i < kHidden; ++i) y[i] = x[i] * s * w[i];
    return y;
}

void rope(float* v, int pos) {
    for (int i = 0; i < kHd / 2; ++i) {
        const float th = pos * std::pow(kTheta, -2.0f * i / kHd);
        const float a = v[i], b = v[i + kHd / 2];
        v[i]           = a * std::cos(th) - b * std::sin(th);
        v[i + kHd / 2] = a * std::sin(th) + b * std::cos(th);
    }
}

std::vector<float> reference(const HostLayer& L, const std::vector<float>& x) {
    std::vector<std::vector<float>> Q(kSeq), K(kSeq), V(kSeq);
    for (int t = 0; t < kSeq; ++t) {
        const auto xn = rmsnorm(&x[static_cast<size_t>(t) * kHidden], L.attn_norm);
        Q[t] = matvec(L.q, xn.data(), kHidden, kHeads * kHd);
        K[t] = matvec(L.k, xn.data(), kHidden, kKv * kHd);
        V[t] = matvec(L.v, xn.data(), kHidden, kKv * kHd);
        for (int i = 0; i < kHeads * kHd; ++i) Q[t][i] += L.qb[i];
        for (int i = 0; i < kKv * kHd; ++i) { K[t][i] += L.kb[i]; V[t][i] += L.vb[i]; }
        for (int h = 0; h < kHeads; ++h) rope(&Q[t][h * kHd], t);
        for (int h = 0; h < kKv; ++h) rope(&K[t][h * kHd], t);
    }
    std::vector<float> y(x.size());
    for (int t = 0; t < kSeq; ++t) {
        std::vector<float> att(kHeads * kHd, 0.0f);
        for (int h = 0; h < kHeads; ++h) {
            const int kh = h / (kHeads / kKv);
            std::vector<float> s(t + 1);
            float mx = -1e30f;
            for (int j = 0; j <= t; ++j) {
                float d = 0;
                for (int i = 0; i < kHd; ++i) d += Q[t][h * kHd + i] * K[j][kh * kHd + i];
                s[j] = d / std::sqrt(static_cast<float>(kHd));
                mx = std::max(mx, s[j]);
            }
            float sum = 0;
            for (auto& e : s) { e = std::exp(e - mx); sum += e; }
            for (int j = 0; j <= t; ++j)
                for (int i = 0; i < kHd; ++i) att[h * kHd + i] += s[j] / sum * V[j][kh * kHd + i];
        }
        const float* xt = &x[static_cast<size_t>(t) * kHidden];
        const auto o = matvec(L.o, att.data(), kHeads * kHd, kHidden);
        std::vector<float> h(kHidden);
        for (int i = 0; i < kHidden; ++i) h[i] = xt[i] + o[i];
        const auto hn = rmsnorm(h.data(), L.ffn_norm);
        const auto g = matvec(L.gate, hn.data(), kHidden, kFfn);
        const auto u = matvec(L.up, hn.data(), kHidden, kFfn);
        std::vector<float> a(kFfn);
        for (int i = 0; i < kFfn; ++i) a[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        const auto f = matvec(L.down, a.data(), kFfn, kHidden);
        for (int i = 0; i < kHidden; ++i) y[static_cast<size_t>(t) * kHidden + i] = h[i] + f[i];
    }
    return y;
}

// Runs the sequence through the resident layer in the given chunk sizes.
bool run(const lv::Qwen2LayerWeights& w, const lv::Qwen2Hparams& hp, ggml_type kv_type,
         const std::vector<float>& x, const std::vector<int>& chunks, std::vector<float>* y) {
    lv::ResidentKV kv;
    if (!kv.init(1, kHd, kKv, kSeq, kv_type)) return false;
    y->clear();
    int t0 = 0;
    for (int n : chunks) {
        struct ggml_init_params p {};
        p.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
        p.no_alloc = true;
        ggml_context* ctx = ggml_init(p);
        ggml_tensor* in   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHidden, n, 1);
        ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, hp.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, t0 + n, n);
        ggml_backend_buffer_t buf = lv::allocate_ctx_tensors(ctx);
        ggml_context* gctx = ggml_init(p);
        auto out = lv::qwen2_layer(gctx, in, pos, mask, kv, 0, w, hp);
        ggml_cgraph* gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, out.k_write);
        ggml_build_forward_expand(gf, out.v_write);
        ggml_build_forward_expand(gf, out.y);

        ggml_backend_tensor_set(in, x.data() + static_cast<size_t>(t0) * kHidden, 0, sizeof(float) * kHidden * n);
        std::vector<int32_t> pv(n);
        for (int i = 0; i < n; ++i) pv[i] = t0 + i;
        ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int32_t) * n);
        std::vector<float> mv(static_cast<size_t>(t0 + n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < t0 + n; ++j) mv[static_cast<size_t>(i) * (t0 + n) + j] = j > t0 + i ? -INFINITY : 0.0f;
        if (hp.flash_attn) {
            std::vector<ggml_fp16_t> mh(mv.size());
            for (size_t i = 0; i < mv.size(); ++i) mh[i] = ggml_fp32_to_fp16(mv[i]);
            ggml_backend_tensor_set(mask, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(float));
        }
        const bool ok = lv::compute_graph(gf);
        if (ok) {
            std::vector<float> part(static_cast<size_t>(kHidden) * n);
            ggml_backend_tensor_get(out.y, part.data(), 0, part.size() * sizeof(float));
            y->insert(y->end(), part.begin(), part.end());
            kv.past_len += n;
        }
        ggml_free(gctx);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        if (!ok) return false;
        t0 += n;
    }
    return true;
}

double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        r += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(d / std::max(r, 1e-30));
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rnd = [&](size_t n, float scale, float offset = 0.0f) {
        std::vector<float> v(n);
        for (auto& e : v) e = nd(rng) * scale + offset;
        return v;
    };
    HostLayer L;
    L.attn_norm = rnd(kHidden, 0.1f, 1.0f);
    L.q  = rnd(static_cast<size_t>(kHidden) * kHeads * kHd, 0.15f);
    L.qb = rnd(kHeads * kHd, 0.1f);
    L.k  = rnd(static_cast<size_t>(kHidden) * kKv * kHd, 0.15f);
    L.kb = rnd(kKv * kHd, 0.1f);
    L.v  = rnd(static_cast<size_t>(kHidden) * kKv * kHd, 0.15f);
    L.vb = rnd(kKv * kHd, 0.1f);
    L.o  = rnd(static_cast<size_t>(kHeads) * kHd * kHidden, 0.1f);
    L.ffn_norm = rnd(kHidden, 0.1f, 1.0f);
    L.gate = rnd(static_cast<size_t>(kHidden) * kFfn, 0.1f);
    L.up   = rnd(static_cast<size_t>(kHidden) * kFfn, 0.1f);
    L.down = rnd(static_cast<size_t>(kFfn) * kHidden, 0.1f);
    const std::vector<float> x = rnd(static_cast<size_t>(kHidden) * kSeq, 1.0f);
    const std::vector<float> want = reference(L, x);

    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16;
    p.no_alloc = true;
    ggml_context* wctx = ggml_init(p);
    auto put = [&](const std::vector<float>& v, int ne0, int ne1) {
        ggml_tensor* t = ne1 > 1 ? ggml_new_tensor_2d(wctx, GGML_TYPE_F32, ne0, ne1)
                                 : ggml_new_tensor_1d(wctx, GGML_TYPE_F32, ne0);
        return std::make_pair(t, &v);
    };
    std::vector<std::pair<ggml_tensor*, const std::vector<float>*>> all;
    lv::Qwen2LayerWeights w;
    auto add = [&](ggml_tensor** dst, const std::vector<float>& v, int ne0, int ne1) {
        auto pr = put(v, ne0, ne1);
        *dst = pr.first;
        all.push_back(pr);
    };
    add(&w.attn_norm, L.attn_norm, kHidden, 1);
    add(&w.attn_q, L.q, kHidden, kHeads * kHd);  add(&w.attn_q_bias, L.qb, kHeads * kHd, 1);
    add(&w.attn_k, L.k, kHidden, kKv * kHd);     add(&w.attn_k_bias, L.kb, kKv * kHd, 1);
    add(&w.attn_v, L.v, kHidden, kKv * kHd);     add(&w.attn_v_bias, L.vb, kKv * kHd, 1);
    add(&w.attn_o, L.o, kHeads * kHd, kHidden);
    add(&w.ffn_norm, L.ffn_norm, kHidden, 1);
    add(&w.ffn_gate, L.gate, kHidden, kFfn);
    add(&w.ffn_up, L.up, kHidden, kFfn);
    add(&w.ffn_down, L.down, kFfn, kHidden);
    ggml_backend_buffer_t wbuf = lv::allocate_ctx_tensors(wctx);
    for (auto& [t, v] : all) ggml_backend_tensor_set(t, v->data(), 0, v->size() * sizeof(float));

    lv::Qwen2Hparams hp;
    hp.hidden = kHidden; hp.n_heads = kHeads; hp.n_kv_heads = kKv; hp.head_dim = kHd;
    hp.rope_theta = kTheta; hp.rms_norm_eps = kEps;

    struct Mode { const char* name; bool fa; ggml_type kv; double tol; };
    std::vector<Mode> modes = {{"eager/f32", false, GGML_TYPE_F32, 1e-4}};
    if (lv::backend_supports_flash_attn()) modes.push_back({"flash/f16", true, GGML_TYPE_F16, 5e-3});

    int failed = 0;
    for (const Mode& m : modes) {
        hp.flash_attn = m.fa;
        for (const auto& chunks : {std::vector<int>{7}, std::vector<int>{3, 4}, std::vector<int>(7, 1)}) {
            std::vector<float> got;
            if (!run(w, hp, m.kv, x, chunks, &got)) {
                std::fprintf(stderr, "FAIL: %s compute\n", m.name);
                return 1;
            }
            const double e = rel_err(got, want);
            const bool ok = e < m.tol;
            std::printf("%-10s %zu chunk(s): rel err %.2e %s\n", m.name, chunks.size(), e, ok ? "ok" : "FAIL");
            failed += !ok;
        }
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return failed ? 1 : 0;
}
