#ifndef LIVEVOICE_FRAME_GRAPH_HPP
#define LIVEVOICE_FRAME_GRAPH_HPP

// One ggml graph per speech frame:
//
//   * every DPM-Solver++ step, with the guidance pair batched as two columns
//     of the diffusion head and the blend and update as graph ops
//   * the acoustic connector plus the speech type embedding
//   * the TTS-LM step for the positive and the negative sequence as two
//     columns, so each layer's weights are read once for both; each column
//     attends to its own resident K/V cache
//   * the EOS classifier on the positive hidden state
//
// A frame used to be a dozen graphs with a device round trip after each;
// on a GPU the per-frame cost is now one launch and the weight bandwidth.

#include "bench.hpp"
#include "diffusion.hpp"
#include "model.hpp"
#include "qwen2.hpp"

#include <vector>

namespace lv {

struct FrameInputs {
    const float* z0        = nullptr;   // [latent] Gaussian noise
    const float* cond_pos  = nullptr;   // [hidden] TTS-LM last hidden
    const float* cond_neg  = nullptr;   // [hidden] negative condition, or null without guidance
    float        cfg_scale = 1.0f;
    const float* t_sin     = nullptr;   // [freq_size * steps] timestep embeddings
};

struct FrameOutputs {
    std::vector<float> latent;       // [latent]
    std::vector<float> hidden_pos;   // [hidden] TTS-LM last hidden after the frame
    std::vector<float> hidden_neg;   // [hidden], empty without guidance
    float              eos_logit = 0.0f;
};

// Advances kv_pos.past_len, and kv_neg->past_len under guidance, by one.
bool run_frame(const Model& model, const DpmSchedule& sched, ResidentKV& kv_pos, ResidentKV* kv_neg,
               const FrameInputs& in, FrameOutputs* out, BenchTotals* bench);

}  // namespace lv

#endif  // LIVEVOICE_FRAME_GRAPH_HPP
