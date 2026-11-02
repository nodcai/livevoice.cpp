#include "diffusion.hpp"

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace lv {

namespace {

// Chunk `idx` of width `n` along ne[0].
struct ggml_tensor* chunk0(struct ggml_context* g, struct ggml_tensor* t, int64_t n, int64_t idx) {
    return ggml_cont(g, ggml_view_2d(g, t, n, t->ne[1], t->nb[1], idx * n * ggml_type_size(t->type)));
}

// x * (1 + scale) + shift
struct ggml_tensor* modulate(struct ggml_context* g, struct ggml_tensor* x,
                             struct ggml_tensor* shift, struct ggml_tensor* scale) {
    return ggml_add(g, ggml_add(g, x, ggml_mul(g, x, scale)), shift);
}

}  // namespace

bool load_diffusion_head(const GgufLoader& m, const std::string& prefix, int n_layers,
                         DiffusionHeadWeights* out) {
    auto get = [&](const std::string& n) { return m.tensor(prefix + n); };
    out->noisy_proj   = get("noisy_proj");
    out->cond_proj    = get("cond_proj");
    out->t_embed_lin1 = get("t_embed_lin1");
    out->t_embed_lin2 = get("t_embed_lin2");
    out->final_proj   = get("final.proj");
    out->final_adaln  = get("final.adaln");
    if (!out->noisy_proj || !out->cond_proj || !out->t_embed_lin1 || !out->t_embed_lin2 ||
        !out->final_proj || !out->final_adaln) {
        LV_LOG_ERROR("diffusion head: missing tensors under %s", prefix.c_str());
        return false;
    }
    out->layers.assign(n_layers, {});
    for (int i = 0; i < n_layers; ++i) {
        char p[32];
        std::snprintf(p, sizeof(p), "layer_%d.", i);
        auto& l = out->layers[i];
        l.norm     = get(std::string(p) + "norm");
        l.ffn_gate = get(std::string(p) + "ffn_gate");
        l.ffn_up   = get(std::string(p) + "ffn_up");
        l.ffn_down = get(std::string(p) + "ffn_down");
        l.adaln    = get(std::string(p) + "adaln");
        if (!l.norm || !l.ffn_gate || !l.ffn_up || !l.ffn_down || !l.adaln) {
            LV_LOG_ERROR("diffusion head: layer %d incomplete", i);
            return false;
        }
    }
    return true;
}

void timestep_sinusoidal(float t, int freq_size, float* out) {
    const int   half    = freq_size / 2;
    const float log_max = std::log(10000.0f);
    for (int k = 0; k < half; ++k) {
        const float arg = t * std::exp(-log_max * static_cast<float>(k) / half);
        out[k]        = std::cos(arg);
        out[k + half] = std::sin(arg);
    }
    if (freq_size % 2 == 1) out[freq_size - 1] = 0.0f;
}

struct ggml_tensor* diffusion_head_forward(struct ggml_context* g, struct ggml_tensor* noisy,
                                           struct ggml_tensor* cond, struct ggml_tensor* t_sin,
                                           const DiffusionHeadWeights& w, const DiffusionHeadConfig& cfg) {
    const int h = cfg.hidden;
    struct ggml_tensor* x = ggml_mul_mat(g, w.noisy_proj, noisy);
    struct ggml_tensor* t = ggml_mul_mat(g, w.t_embed_lin2, ggml_silu(g, ggml_mul_mat(g, w.t_embed_lin1, t_sin)));
    struct ggml_tensor* c = ggml_add(g, ggml_mul_mat(g, w.cond_proj, cond), t);   // t broadcasts over columns
    struct ggml_tensor* sc = ggml_silu(g, c);

    for (const auto& l : w.layers) {
        struct ggml_tensor* m = ggml_mul_mat(g, l.adaln, sc);
        struct ggml_tensor* xn = ggml_mul(g, ggml_rms_norm(g, x, cfg.eps), l.norm);
        struct ggml_tensor* u  = modulate(g, xn, chunk0(g, m, h, 0), chunk0(g, m, h, 1));
        struct ggml_tensor* f  = ggml_mul_mat(g, l.ffn_down,
                                              ggml_mul(g, ggml_silu(g, ggml_mul_mat(g, l.ffn_gate, u)),
                                                       ggml_mul_mat(g, l.ffn_up, u)));
        x = ggml_add(g, x, ggml_mul(g, f, chunk0(g, m, h, 2)));
    }
    struct ggml_tensor* m = ggml_mul_mat(g, w.final_adaln, sc);
    struct ggml_tensor* u = modulate(g, ggml_rms_norm(g, x, cfg.eps), chunk0(g, m, h, 0), chunk0(g, m, h, 1));
    return ggml_mul_mat(g, w.final_proj, u);
}

void DpmSchedule::init(int steps, int n_train) {
    num_train_timesteps = n_train;
    auto alpha_bar = [](float t) {
        const float v = std::cos((t + 0.008f) / 1.008f * 3.14159265358979323846f * 0.5f);
        return v * v;
    };
    alpha.resize(n_train);
    sigma.resize(n_train);
    lambda.resize(n_train);
    double ac = 1.0;
    for (int i = 0; i < n_train; ++i) {
        const float beta = std::min(1.0f - alpha_bar(static_cast<float>(i + 1) / n_train) /
                                           alpha_bar(static_cast<float>(i) / n_train), 0.999f);
        ac *= 1.0 - beta;
        alpha[i]  = static_cast<float>(std::sqrt(ac));
        sigma[i]  = static_cast<float>(std::sqrt(1.0 - ac));
        lambda[i] = std::log(alpha[i]) - std::log(sigma[i]);
    }
    // linspace(n_train - 1, 0, steps + 1).round(); the last step lands on
    // the terminal state alpha = 1, sigma = 0.
    steps = std::max(1, steps);
    timesteps.assign(steps + 1, 0);
    for (int i = 0; i <= steps; ++i)
        timesteps[i] = static_cast<int>(std::round((1.0f - static_cast<float>(i) / steps) * (n_train - 1)));
    timesteps.back() = -1;
}

struct ggml_tensor* dpm_sample(struct ggml_context* g, struct ggml_tensor* z0, struct ggml_tensor* cond,
                               struct ggml_tensor* t_sin, float cfg_scale, const DpmSchedule& sched,
                               const DiffusionHeadWeights& w, const DiffusionHeadConfig& cfg) {
    const int64_t latent = z0->ne[0];
    const int64_t n      = z0->ne[1];
    const bool    guided = cond->ne[1] == 2 * n;
    const int     M      = sched.steps();

    struct ggml_tensor* z       = z0;
    struct ggml_tensor* prev_x0 = nullptr;
    for (int i = 0; i < M; ++i) {
        const int   t   = sched.timesteps[i];
        const int   s   = sched.timesteps[i + 1];
        const float a_t = sched.alpha[t], sg_t = sched.sigma[t], l_t = sched.lambda[t];

        // The guided pair runs as 2n columns of one head evaluation.
        struct ggml_tensor* noisy = guided ? ggml_concat(g, z, z, 1) : z;
        struct ggml_tensor* tsv   = ggml_view_1d(g, t_sin, t_sin->ne[0], static_cast<size_t>(i) * t_sin->nb[1]);
        struct ggml_tensor* v     = diffusion_head_forward(g, noisy, cond, tsv, w, cfg);
        if (guided) {
            struct ggml_tensor* v_pos = ggml_view_2d(g, v, latent, n, v->nb[1], 0);
            struct ggml_tensor* v_neg = ggml_view_2d(g, v, latent, n, v->nb[1], static_cast<size_t>(n) * v->nb[1]);
            v = ggml_add(g, ggml_scale(g, v_neg, 1.0f - cfg_scale), ggml_scale(g, v_pos, cfg_scale));
        }
        struct ggml_tensor* x0 = ggml_sub(g, ggml_scale(g, z, a_t), ggml_scale(g, v, sg_t));

        const bool first_order = i == 0 || i == M - 1;
        if (s == -1) {
            z = x0;
        } else {
            const float a_s = sched.alpha[s], sg_s = sched.sigma[s];
            const float hh  = sched.lambda[s] - l_t;
            const float A   = sg_s / sg_t;
            const float B   = a_s * (std::exp(-hh) - 1.0f);
            z = ggml_sub(g, ggml_scale(g, z, A), ggml_scale(g, x0, B));
            if (!first_order) {
                const float r0 = (l_t - sched.lambda[sched.timesteps[i - 1]]) / hh;
                struct ggml_tensor* D1 = ggml_scale(g, ggml_sub(g, x0, prev_x0), 1.0f / r0);
                z = ggml_sub(g, z, ggml_scale(g, D1, 0.5f * B));
            }
        }
        prev_x0 = x0;
    }
    return z;
}

}  // namespace lv
