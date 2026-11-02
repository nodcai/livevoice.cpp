// Diffusion head and the in-graph DPM-Solver++ sampler against plain C++.
//
// A small random head is evaluated once (v-prediction) and then sampled for
// 5 and 20 steps with classifier-free guidance, all compared with a direct
// implementation of the head and of the solver's update rule. Also checks
// the schedule's timesteps.
#include "backend.hpp"
#include "diffusion.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr int kHidden = 64, kLatent = 16, kLayers = 2, kFfn = 192, kCols = 3, kFreq = 256;
constexpr float kEps = 1e-5f;

using Vec = std::vector<float>;

struct HostHead {
    Vec noisy_proj, cond_proj, lin1, lin2, final_proj, final_adaln;
    struct Layer { Vec norm, gate, up, down, adaln; };
    std::vector<Layer> layers;
};

Vec matvec(const Vec& W, const float* x, int in, int out) {
    Vec y(out, 0.0f);
    for (int r = 0; r < out; ++r)
        for (int c = 0; c < in; ++c) y[r] += W[static_cast<size_t>(r) * in + c] * x[c];
    return y;
}
float silu(float v) { return v / (1.0f + std::exp(-v)); }
Vec rms(const Vec& x, const Vec* w) {
    double ss = 0;
    for (float v : x) ss += static_cast<double>(v) * v;
    const float s = 1.0f / std::sqrt(static_cast<float>(ss / x.size()) + kEps);
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * s * (w ? (*w)[i] : 1.0f);
    return y;
}

// v for one column.
Vec head_ref(const HostHead& H, const float* z, const float* cond, float t) {
    Vec ts(kFreq);
    lv::timestep_sinusoidal(t, kFreq, ts.data());
    Vec x  = matvec(H.noisy_proj, z, kLatent, kHidden);
    Vec te = matvec(H.lin1, ts.data(), kFreq, kHidden);
    for (auto& v : te) v = silu(v);
    te = matvec(H.lin2, te.data(), kHidden, kHidden);
    Vec c = matvec(H.cond_proj, cond, kHidden, kHidden);
    for (int i = 0; i < kHidden; ++i) c[i] = silu(c[i] + te[i]);
    for (const auto& l : H.layers) {
        const Vec m  = matvec(l.adaln, c.data(), kHidden, 3 * kHidden);
        Vec u = rms(x, &l.norm);
        for (int i = 0; i < kHidden; ++i) u[i] = u[i] * (1.0f + m[kHidden + i]) + m[i];
        Vec gt = matvec(l.gate, u.data(), kHidden, kFfn), up = matvec(l.up, u.data(), kHidden, kFfn);
        for (int i = 0; i < kFfn; ++i) gt[i] = silu(gt[i]) * up[i];
        const Vec f = matvec(l.down, gt.data(), kFfn, kHidden);
        for (int i = 0; i < kHidden; ++i) x[i] += f[i] * m[2 * kHidden + i];
    }
    const Vec m = matvec(H.final_adaln, c.data(), kHidden, 2 * kHidden);
    Vec u = rms(x, nullptr);
    for (int i = 0; i < kHidden; ++i) u[i] = u[i] * (1.0f + m[kHidden + i]) + m[i];
    return matvec(H.final_proj, u.data(), kHidden, kLatent);
}

// The solver on the host, one column at a time.
Vec sample_ref(const HostHead& H, const lv::DpmSchedule& S, const float* z0, const float* cp, const float* cn,
               float cfg) {
    Vec z(z0, z0 + kLatent), prev(kLatent, 0.0f);
    const int M = S.steps();
    for (int i = 0; i < M; ++i) {
        const int t = S.timesteps[i], s = S.timesteps[i + 1];
        Vec v = head_ref(H, z.data(), cp, static_cast<float>(t));
        const Vec vn = head_ref(H, z.data(), cn, static_cast<float>(t));
        for (int k = 0; k < kLatent; ++k) v[k] = vn[k] + cfg * (v[k] - vn[k]);
        Vec x0(kLatent);
        for (int k = 0; k < kLatent; ++k) x0[k] = S.alpha[t] * z[k] - S.sigma[t] * v[k];
        if (s == -1) {
            z = x0;
        } else {
            const float h = S.lambda[s] - S.lambda[t];
            const float A = S.sigma[s] / S.sigma[t], B = S.alpha[s] * (std::exp(-h) - 1.0f);
            const bool second = i > 0 && i < M - 1;
            const float r0 = second ? (S.lambda[t] - S.lambda[S.timesteps[i - 1]]) / h : 1.0f;
            for (int k = 0; k < kLatent; ++k)
                z[k] = A * z[k] - B * x0[k] - (second ? 0.5f * B * (x0[k] - prev[k]) / r0 : 0.0f);
        }
        prev = x0;
    }
    return z;
}

double rel_err(const Vec& a, const Vec& b) {
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        r += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(d / std::max(r, 1e-30));
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rnd = [&](size_t n, float sc, float off = 0.0f) {
        Vec v(n);
        for (auto& e : v) e = nd(rng) * sc + off;
        return v;
    };
    HostHead H;
    H.noisy_proj  = rnd(kLatent * kHidden, 0.2f);
    H.cond_proj   = rnd(kHidden * kHidden, 0.1f);
    H.lin1        = rnd(kFreq * kHidden, 0.05f);
    H.lin2        = rnd(kHidden * kHidden, 0.1f);
    H.final_proj  = rnd(kHidden * kLatent, 0.1f);
    H.final_adaln = rnd(kHidden * 2 * kHidden, 0.05f);
    for (int i = 0; i < kLayers; ++i)
        H.layers.push_back({rnd(kHidden, 0.1f, 1.0f), rnd(kHidden * kFfn, 0.1f), rnd(kHidden * kFfn, 0.1f),
                            rnd(kFfn * kHidden, 0.1f), rnd(kHidden * 3 * kHidden, 0.05f)});

    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * 64;
    ip.no_alloc = true;
    ggml_context* wctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor*, const Vec*>> uploads;
    auto t2 = [&](const Vec& v, int ne0, int ne1) {
        ggml_tensor* t = ne1 > 1 ? ggml_new_tensor_2d(wctx, GGML_TYPE_F32, ne0, ne1)
                                 : ggml_new_tensor_1d(wctx, GGML_TYPE_F32, ne0);
        uploads.push_back({t, &v});
        return t;
    };
    lv::DiffusionHeadWeights w;
    w.noisy_proj   = t2(H.noisy_proj, kLatent, kHidden);
    w.cond_proj    = t2(H.cond_proj, kHidden, kHidden);
    w.t_embed_lin1 = t2(H.lin1, kFreq, kHidden);
    w.t_embed_lin2 = t2(H.lin2, kHidden, kHidden);
    w.final_proj   = t2(H.final_proj, kHidden, kLatent);
    w.final_adaln  = t2(H.final_adaln, kHidden, 2 * kHidden);
    for (const auto& l : H.layers)
        w.layers.push_back({t2(l.norm, kHidden, 1), t2(l.gate, kHidden, kFfn), t2(l.up, kHidden, kFfn),
                            t2(l.down, kFfn, kHidden), t2(l.adaln, kHidden, 3 * kHidden)});
    ggml_backend_buffer_t wbuf = lv::allocate_ctx_tensors(wctx);
    for (auto& [t, v] : uploads) ggml_backend_tensor_set(t, v->data(), 0, v->size() * sizeof(float));

    lv::DiffusionHeadConfig cfg;
    cfg.hidden = kHidden; cfg.latent = kLatent; cfg.layers = kLayers; cfg.freq_size = kFreq; cfg.eps = kEps;

    const Vec z0   = rnd(kLatent * kCols, 1.0f);
    const Vec cond = rnd(kHidden * 2 * kCols, 1.0f);   // positive columns, then negative
    const float guidance = 1.5f;
    int failed = 0;

    // Runs `build` over inputs z0 / cond / t_sin and reads back [kLatent, kCols].
    auto run = [&](int steps, bool head_only, Vec* out) {
        lv::DpmSchedule S;
        S.init(steps);
        ggml_init_params p {};
        p.mem_size = ggml_tensor_overhead() * 8;
        p.no_alloc = true;
        ggml_context* ictx = ggml_init(p);
        ggml_tensor* tz = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kLatent, kCols);
        ggml_tensor* tc = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kHidden, head_only ? kCols : 2 * kCols);
        ggml_tensor* tt = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kFreq, S.steps());
        ggml_backend_buffer_t ibuf = lv::allocate_ctx_tensors(ictx);
        ggml_backend_tensor_set(tz, z0.data(), 0, ggml_nbytes(tz));
        ggml_backend_tensor_set(tc, cond.data(), 0, ggml_nbytes(tc));
        Vec ts(static_cast<size_t>(kFreq) * S.steps());
        for (int i = 0; i < S.steps(); ++i) lv::timestep_sinusoidal(static_cast<float>(S.timesteps[i]), kFreq, &ts[i * kFreq]);
        ggml_backend_tensor_set(tt, ts.data(), 0, ggml_nbytes(tt));

        p.mem_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
        ggml_context* g = ggml_init(p);
        ggml_tensor* y = head_only
            ? lv::diffusion_head_forward(g, tz, tc, ggml_view_1d(g, tt, kFreq, 0), w, cfg)
            : lv::dpm_sample(g, tz, tc, tt, guidance, S, w, cfg);
        ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);
        ggml_build_forward_expand(gf, y);
        const bool ok = lv::compute_graph(gf);
        out->resize(static_cast<size_t>(kLatent) * kCols);
        if (ok) ggml_backend_tensor_get(y, out->data(), 0, out->size() * sizeof(float));
        ggml_free(g);
        ggml_backend_buffer_free(ibuf);
        ggml_free(ictx);
        return ok ? S : lv::DpmSchedule{};
    };

    {
        Vec got;
        const lv::DpmSchedule S = run(5, true, &got);
        Vec want;
        for (int c = 0; c < kCols; ++c) {
            const Vec v = head_ref(H, &z0[c * kLatent], &cond[c * kHidden], static_cast<float>(S.timesteps[0]));
            want.insert(want.end(), v.begin(), v.end());
        }
        const double e = rel_err(got, want);
        std::printf("head:     rel err %.2e\n", e);
        failed += !(e < 1e-4);
    }
    for (int steps : {1, 5, 20}) {
        Vec got;
        const lv::DpmSchedule S = run(steps, false, &got);
        Vec want;
        for (int c = 0; c < kCols; ++c) {
            const Vec z = sample_ref(H, S, &z0[c * kLatent], &cond[c * kHidden], &cond[(kCols + c) * kHidden], guidance);
            want.insert(want.end(), z.begin(), z.end());
        }
        const double e = rel_err(got, want);
        std::printf("sample %2d steps: rel err %.2e\n", steps, e);
        failed += !(e < 1e-4);
    }
    {
        lv::DpmSchedule S;
        S.init(5);
        const std::vector<int> want = {999, 799, 599, 400, 200, -1};
        if (S.timesteps != want) {
            std::printf("schedule: timesteps differ\n");
            ++failed;
        }
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    std::printf("%s\n", failed ? "FAIL" : "OK");
    return failed ? 1 : 0;
}
