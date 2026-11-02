#ifndef LIVEVOICE_DIFFUSION_HPP
#define LIVEVOICE_DIFFUSION_HPP

// Diffusion head and DPM-Solver++ sampler that turn a TTS-LM hidden state
// into one 64-d speech latent.
//
// Head (v-prediction):
//   x = noisy_proj(z)
//   c = cond_proj(cond) + t_embed(t)
//   per layer: (shift, scale, gate) = adaln(silu(c))
//              x += gate * SwiGLU(modulate(RMSNorm(x), shift, scale))
//   final:     (shift, scale) = adaln_final(silu(c))
//              v = proj(modulate(RMSNorm_no_weight(x), shift, scale))
//
// Sampler: DPM-Solver++ (2M, lower order on the first and last step), cosine
// beta schedule, v-prediction, as diffusers' DPMSolverMultistepScheduler with
// algorithm_type="dpmsolver++". The whole solve, classifier-free guidance
// included, is built into one graph; the schedule's coefficients are graph
// constants.

#include "gguf_loader.hpp"

#include "ggml.h"

#include <string>
#include <vector>

namespace lv {

struct DiffusionHeadConfig {
    int   hidden      = 896;
    int   latent      = 64;
    int   layers      = 4;
    int   freq_size   = 256;
    float eps         = 1e-5f;
};

struct DiffusionHeadWeights {
    struct HeadLayer {
        struct ggml_tensor* norm     = nullptr;  // [hidden]
        struct ggml_tensor* ffn_gate = nullptr;  // [hidden, ffn]
        struct ggml_tensor* ffn_up   = nullptr;
        struct ggml_tensor* ffn_down = nullptr;  // [ffn, hidden]
        struct ggml_tensor* adaln    = nullptr;  // [hidden, 3*hidden]
    };
    struct ggml_tensor* noisy_proj   = nullptr;  // [latent, hidden]
    struct ggml_tensor* cond_proj    = nullptr;  // [hidden, hidden]
    struct ggml_tensor* t_embed_lin1 = nullptr;  // [freq_size, hidden]
    struct ggml_tensor* t_embed_lin2 = nullptr;  // [hidden, hidden]
    std::vector<HeadLayer> layers;
    struct ggml_tensor* final_proj   = nullptr;  // [hidden, latent]
    struct ggml_tensor* final_adaln  = nullptr;  // [hidden, 2*hidden]
};

bool load_diffusion_head(const GgufLoader& m, const std::string& prefix, int n_layers,
                         DiffusionHeadWeights* out);

// cos/sin embedding of the scalar timestep `t`, `freq_size` floats.
void timestep_sinusoidal(float t, int freq_size, float* out);

//   noisy: [latent, n], cond: [hidden, n], t_sin: [freq_size] -> v: [latent, n]
struct ggml_tensor* diffusion_head_forward(struct ggml_context* g, struct ggml_tensor* noisy,
                                           struct ggml_tensor* cond, struct ggml_tensor* t_sin,
                                           const DiffusionHeadWeights& w, const DiffusionHeadConfig& cfg);

struct DpmSchedule {
    int                num_train_timesteps = 1000;
    std::vector<float> alpha, sigma, lambda;   // per training timestep
    std::vector<int>   timesteps;              // steps + 1 entries, the last is -1 (sigma = 0)

    void init(int steps, int num_train_timesteps = 1000);
    int  steps() const { return static_cast<int>(timesteps.size()) - 1; }
};

// The full solve as graph ops.
//   z0:    [latent, n] initial noise
//   cond:  [hidden, n] positive condition, followed by n negative columns when
//          cfg_scale > 1 ([hidden, 2n]); v = v_neg + cfg_scale * (v_pos - v_neg)
//   t_sin: [freq_size, steps] timestep embeddings of schedule.timesteps
// Returns the denoised latent [latent, n].
struct ggml_tensor* dpm_sample(struct ggml_context* g, struct ggml_tensor* z0, struct ggml_tensor* cond,
                               struct ggml_tensor* t_sin, float cfg_scale, const DpmSchedule& sched,
                               const DiffusionHeadWeights& w, const DiffusionHeadConfig& cfg);

}  // namespace lv

#endif  // LIVEVOICE_DIFFUSION_HPP
