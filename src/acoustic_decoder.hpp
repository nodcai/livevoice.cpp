#ifndef LIVEVOICE_ACOUSTIC_DECODER_HPP
#define LIVEVOICE_ACOUSTIC_DECODER_HPP

// Stateful σ-VAE acoustic decoder: speech latents (7.5 Hz) -> 24 kHz audio.
//
//   stem conv K=7 -> (transposed-conv upsample -> ConvNeXt Block1D stage) x N
//   -> [final RMSNorm] -> head conv K=7 -> 1 channel
//
// built from ops that are cheap on every ggml backend and that carry their
// left context from one call to the next:
//
//   * dense causal conv K=7 (stem, head)  ->  sum_k mul_mat(W_k, shift_k(x))
//   * depthwise causal conv K=7 (mixer)   ->  sum_k mul(shift_k(x), w_k)
//   * transposed conv, K = 2*stride       ->  one mul_mat C_in -> 2*stride*C_out
//                                             and an overlap-add of the halves
//
// Every conv keeps its left context in a small persistent tensor that is
// concatenated in front of the chunk and rewritten inside the graph (the
// K/V-cache pattern), so decoding a stream chunk by chunk gives the same
// signal as decoding it at once, and nothing is ever decoded twice.
// ggml_conv_transpose_1d is not used: on Metal it is pathologically slow or
// never returns for this 3200x upsample, and Metal has no left pad.
//
// Layout is channels-first, [C, T] in ggml order, so a frame-major latent
// buffer ([n_frames][latent] on the host) uploads as it is.

#include "gguf_loader.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <string>
#include <vector>

namespace lv {

class AcousticDecoder {
public:
    struct Block {
        struct ggml_tensor* norm = nullptr, *mixer_bias = nullptr, *gamma = nullptr;
        struct ggml_tensor* ffn_norm = nullptr, *ffn_gamma = nullptr;
        struct ggml_tensor* lin1 = nullptr, *lin1_b = nullptr, *lin2 = nullptr, *lin2_b = nullptr;
        struct ggml_tensor* dw = nullptr;   // re-laid depthwise kernel [C, 7] F32
    };

    AcousticDecoder() = default;
    ~AcousticDecoder() { free(); }
    AcousticDecoder(const AcousticDecoder&)            = delete;
    AcousticDecoder& operator=(const AcousticDecoder&) = delete;

    // Reads `<prefix>.stem`, `.up_<s>` (s = 1..), `.stage_<s>_block_<b>`,
    // `.final_norm` (optional) and `.head`. The geometry (stages, channels,
    // strides, depths) comes from the tensors themselves.
    bool load(const GgufLoader& m, const std::string& prefix, float eps);
    void free();

    bool ready() const { return ready_; }
    int  latent_dim() const { return c_in_; }
    // Output samples per latent frame (product of the strides).
    int  hop() const;

    struct State {
        struct ggml_context*             ctx = nullptr;
        ggml_backend_buffer_t            buf = nullptr;
        struct ggml_tensor*              stem = nullptr;   // [C_in, 6]
        struct ggml_tensor*              head = nullptr;   // [C_last, 6]
        std::vector<struct ggml_tensor*> dw;               // per block [C, 6]
        std::vector<struct ggml_tensor*> tr;               // per stage [C_s, stride, 1]

        State() = default;
        ~State() { free(); }
        State(const State&)            = delete;
        State& operator=(const State&) = delete;
        void free();
    };

    // Zeroed conv state for one stream.
    bool init_state(State* st) const;
    void reset_state(State& st) const;

    // Decodes `n_frames` latents (frame-major, already unscaled) continuing
    // from `st`, appending hop() samples per frame to *audio. One graph per
    // `chunk_frames` keeps the activation footprint bounded.
    bool decode(State& st, const float* latents, int n_frames, std::vector<float>* audio,
                int chunk_frames = 32) const;

private:
    struct ggml_tensor* build(struct ggml_context* g, State& st, struct ggml_tensor* in,
                              std::vector<struct ggml_tensor*>* state_writes) const;

    struct ggml_context*  ctx_ = nullptr;   // re-laid weights
    ggml_backend_buffer_t buf_ = nullptr;
    bool  ready_ = false;
    float eps_   = 1e-5f;
    int   c_in_  = 0;
    std::vector<int> C_, depth_, stride_;   // per stage; stride_[0] = 0

    struct ggml_tensor* stem_w_ = nullptr, *stem_b_ = nullptr;   // [C_in, C_0, 7] F16
    struct ggml_tensor* head_w_ = nullptr, *head_b_ = nullptr;   // [C_last, 1, 7] F16
    std::vector<struct ggml_tensor*> tr_w_, tr_b_;               // per stage: [C_{s-1}, 2*stride*C_s] F16
    std::vector<Block>  blocks_;                                 // stage-major
    struct ggml_tensor* final_norm_ = nullptr;
};

}  // namespace lv

#endif  // LIVEVOICE_ACOUSTIC_DECODER_HPP
