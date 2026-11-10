#include "acoustic_decoder.hpp"

#include "backend.hpp"
#include "bench.hpp"
#include "common.hpp"

#include <cstdio>
#include <cstring>

namespace lv {

namespace {

constexpr int kTaps = 7;
constexpr int kCtx  = kTaps - 1;

void write_f16(struct ggml_tensor* dst, const std::vector<float>& src) {
    std::vector<ggml_fp16_t> tmp(src.size());
    for (size_t i = 0; i < src.size(); ++i) tmp[i] = ggml_fp32_to_fp16(src[i]);
    ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
}

// Dense kernel [K, Ci, Co] (ggml ne) -> tap-major planes [Ci, Co, K].
void relay_dense(const struct ggml_tensor* src_t, struct ggml_tensor* dst_t) {
    const int K = static_cast<int>(src_t->ne[0]), Ci = static_cast<int>(src_t->ne[1]), Co = static_cast<int>(src_t->ne[2]);
    const auto src = tensor_to_f32(src_t);
    std::vector<float> dst(src.size());
    for (int k = 0; k < K; ++k)
        for (int ci = 0; ci < Ci; ++ci)
            for (int co = 0; co < Co; ++co)
                dst[static_cast<size_t>(k) * Ci * Co + static_cast<size_t>(co) * Ci + ci] =
                    src[static_cast<size_t>(co) * Ci * K + static_cast<size_t>(ci) * K + k];
    write_f16(dst_t, dst);
}

// Transposed kernel [K, Co, Ci] -> [Ci, K*Co], row r = k*Co + co.
void relay_transposed(const struct ggml_tensor* src_t, struct ggml_tensor* dst_t) {
    const int K = static_cast<int>(src_t->ne[0]), Co = static_cast<int>(src_t->ne[1]), Ci = static_cast<int>(src_t->ne[2]);
    const auto src = tensor_to_f32(src_t);
    std::vector<float> dst(src.size());
    for (int ci = 0; ci < Ci; ++ci)
        for (int co = 0; co < Co; ++co)
            for (int k = 0; k < K; ++k)
                dst[(static_cast<size_t>(k) * Co + co) * Ci + ci] =
                    src[static_cast<size_t>(ci) * K * Co + static_cast<size_t>(co) * K + k];
    write_f16(dst_t, dst);
}

// Depthwise kernel [K, 1, C] -> [C, K], tap plane k at offset k*C.
void relay_depthwise(const struct ggml_tensor* src_t, struct ggml_tensor* dst_t) {
    const int K = static_cast<int>(src_t->ne[0]), C = static_cast<int>(src_t->ne[2]);
    const auto src = tensor_to_f32(src_t);
    std::vector<float> dst(src.size());
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < K; ++k) dst[static_cast<size_t>(k) * C + c] = src[static_cast<size_t>(c) * K + k];
    ggml_backend_tensor_set(dst_t, dst.data(), 0, dst.size() * sizeof(float));
}

struct ggml_tensor* norm_mul(struct ggml_context* g, struct ggml_tensor* x, struct ggml_tensor* w, float eps) {
    x = ggml_rms_norm(g, x, eps);
    return w ? ggml_mul(g, x, w) : x;
}

}  // namespace

void AcousticDecoder::free() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr;
    ctx_ = nullptr;
    ready_ = false;
}

void AcousticDecoder::State::free() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
    buf = nullptr;
    ctx = nullptr;
}

int AcousticDecoder::hop() const {
    int h = 1;
    for (size_t s = 1; s < stride_.size(); ++s) h *= stride_[s];
    return h;
}

bool AcousticDecoder::load(const GgufLoader& m, const std::string& prefix, float eps) {
    free();
    eps_ = eps;
    auto get = [&](const std::string& n) { return m.tensor(prefix + "." + n); };

    struct ggml_tensor* stem_k = get("stem.weight");
    struct ggml_tensor* head_k = get("head.weight");
    if (!stem_k || !head_k || stem_k->ne[0] != kTaps || head_k->ne[0] != kTaps || head_k->ne[2] != 1) {
        LV_LOG_ERROR("decoder: %s.stem / .head missing or not K=%d", prefix.c_str(), kTaps);
        return false;
    }
    c_in_ = static_cast<int>(stem_k->ne[1]);
    C_     = {static_cast<int>(stem_k->ne[2])};
    stride_ = {0};
    std::vector<struct ggml_tensor*> up_k = {nullptr};
    tr_b_ = {nullptr};
    for (int s = 1;; ++s) {
        struct ggml_tensor* k = get("up_" + std::to_string(s) + ".weight");
        if (!k) break;
        struct ggml_tensor* bias = get("up_" + std::to_string(s) + ".bias");
        // Either the kernel [K, C_out, C_in], or the matrix it is re-laid to,
        // [C_in, K*C_out]: the only layout a quantized kernel can be stored
        // in. That one is multiplied with as it is; C_out comes from the bias.
        const bool    relaid = ggml_n_dims(k) == 2;
        const int64_t c_out  = relaid ? (bias ? ggml_nelements(bias) : 0) : k->ne[1];
        const int64_t taps   = relaid ? (c_out > 0 ? k->ne[1] / c_out : 0) : k->ne[0];
        const int64_t c_from = relaid ? k->ne[0] : k->ne[2];
        if (taps <= 0 || taps % 2 != 0 || c_from != C_.back() || (relaid && taps * c_out != k->ne[1])) {
            LV_LOG_ERROR("decoder: up_%d is not a K = 2*stride transposed conv from %d channels", s, C_.back());
            return false;
        }
        up_k.push_back(k);
        tr_b_.push_back(bias);
        stride_.push_back(static_cast<int>(taps / 2));
        C_.push_back(static_cast<int>(c_out));
    }
    const int n_stages = static_cast<int>(C_.size());
    if (head_k->ne[1] != C_.back()) {
        LV_LOG_ERROR("decoder: head expects %lld channels, last stage has %d",
                     static_cast<long long>(head_k->ne[1]), C_.back());
        return false;
    }

    std::vector<struct ggml_tensor*> mixer_k;
    depth_.assign(n_stages, 0);
    for (int s = 0; s < n_stages; ++s) {
        for (int b = 0;; ++b) {
            const std::string p = "stage_" + std::to_string(s) + "_block_" + std::to_string(b) + ".weight.";
            struct ggml_tensor* mk = get(p + "mixer_weight");
            if (!mk) break;
            Block blk;
            blk.norm       = get(p + "norm");
            blk.mixer_bias = get(p + "mixer_bias");
            blk.gamma      = get(p + "gamma");
            blk.ffn_norm   = get(p + "ffn_norm");
            blk.ffn_gamma  = get(p + "ffn_gamma");
            blk.lin1       = get(p + "ffn_linear1");
            blk.lin1_b     = get(p + "ffn_linear1_bias");
            blk.lin2       = get(p + "ffn_linear2");
            blk.lin2_b     = get(p + "ffn_linear2_bias");
            if (!blk.norm || !blk.ffn_norm || !blk.lin1 || !blk.lin2 || mk->ne[0] != kTaps || mk->ne[2] != C_[s]) {
                LV_LOG_ERROR("decoder: stage %d block %d incomplete", s, b);
                return false;
            }
            blocks_.push_back(blk);
            mixer_k.push_back(mk);
            ++depth_[s];
        }
    }
    final_norm_ = get("final_norm");
    stem_b_ = get("stem.bias");
    head_b_ = get("head.bias");

    // ---- re-laid copies of the conv kernels ----
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * (2 + n_stages + blocks_.size());
    ip.no_alloc = true;
    ctx_ = ggml_init(ip);
    stem_w_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F16, c_in_, C_[0], kTaps);
    head_w_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F16, C_.back(), 1, kTaps);
    tr_w_.assign(n_stages, nullptr);
    for (int s = 1; s < n_stages; ++s)
        tr_w_[s] = ggml_n_dims(up_k[s]) == 2
            ? up_k[s]
            : ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, C_[s - 1], static_cast<int64_t>(2) * stride_[s] * C_[s]);
    for (size_t i = 0, s = 0, b = 0; i < blocks_.size(); ++i, ++b) {
        while (static_cast<int>(b) >= depth_[s]) { ++s; b = 0; }
        blocks_[i].dw = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, C_[s], kTaps);
    }
    buf_ = allocate_ctx_tensors(ctx_);
    if (!buf_) {
        LV_LOG_ERROR("decoder: weight buffer allocation failed");
        free();
        return false;
    }
    BenchClock clk;
    relay_dense(stem_k, stem_w_);
    relay_dense(head_k, head_w_);
    for (int s = 1; s < n_stages; ++s)
        if (tr_w_[s] != up_k[s]) relay_transposed(up_k[s], tr_w_[s]);
    for (size_t i = 0; i < blocks_.size(); ++i) relay_depthwise(mixer_k[i], blocks_[i].dw);
    ready_ = true;
    LV_LOG_INFO("decoder: %d stages, %zu blocks, hop %d (%.1f MB re-laid in %.0f ms)", n_stages, blocks_.size(),
                hop(), static_cast<double>(ggml_backend_buffer_get_size(buf_)) / 1e6, clk.ms());
    return true;
}

bool AcousticDecoder::init_state(State* st) const {
    if (!ready_ || !st) return false;
    st->free();
    const int n_stages = static_cast<int>(C_.size());
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * (2 + n_stages + blocks_.size());
    ip.no_alloc = true;
    st->ctx  = ggml_init(ip);
    st->stem = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, c_in_, kCtx);
    st->head = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, C_.back(), kCtx);
    st->dw.clear();
    for (int s = 0; s < n_stages; ++s)
        for (int b = 0; b < depth_[s]; ++b) st->dw.push_back(ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, C_[s], kCtx));
    st->tr.assign(n_stages, nullptr);
    for (int s = 1; s < n_stages; ++s) st->tr[s] = ggml_new_tensor_3d(st->ctx, GGML_TYPE_F32, C_[s], stride_[s], 1);
    st->buf = allocate_ctx_tensors(st->ctx);
    if (!st->buf) {
        LV_LOG_ERROR("decoder: state allocation failed");
        st->free();
        return false;
    }
    ggml_backend_buffer_clear(st->buf, 0);
    return true;
}

void AcousticDecoder::reset_state(State& st) const {
    if (st.buf) ggml_backend_buffer_clear(st.buf, 0);
}

struct ggml_tensor* AcousticDecoder::build(struct ggml_context* g, State& st, struct ggml_tensor* in,
                                           std::vector<struct ggml_tensor*>* writes) const {
    // The persistent left context goes in front of the chunk; the chunk's
    // tail becomes the next call's context.
    auto with_context = [&](struct ggml_tensor* x, struct ggml_tensor* state) {
        const int64_t T = x->ne[1];
        struct ggml_tensor* xin  = ggml_concat(g, state, x, 1);   // [C, 6 + T]
        struct ggml_tensor* tail = ggml_view_2d(g, xin, xin->ne[0], kCtx, xin->nb[1], static_cast<size_t>(T) * xin->nb[1]);
        writes->push_back(ggml_cpy(g, tail, state));
        return xin;
    };
    auto shift = [&](struct ggml_tensor* xin, int k, int64_t T) {
        return ggml_view_2d(g, xin, xin->ne[0], T, xin->nb[1], static_cast<size_t>(k) * xin->nb[1]);
    };
    // y = sum_k W_k * shift_k(xin)
    auto dense7 = [&](struct ggml_tensor* xin, struct ggml_tensor* wt, struct ggml_tensor* bias, int64_t T) {
        struct ggml_tensor* y = nullptr;
        for (int k = 0; k < kTaps; ++k) {
            struct ggml_tensor* wk = ggml_view_2d(g, wt, wt->ne[0], wt->ne[1], wt->nb[1], static_cast<size_t>(k) * wt->nb[2]);
            struct ggml_tensor* m  = ggml_mul_mat(g, wk, ggml_cont(g, shift(xin, k, T)));
            y = y ? ggml_add(g, y, m) : m;
        }
        return bias ? ggml_add(g, y, bias) : y;
    };

    struct ggml_tensor* h = dense7(with_context(in, st.stem), stem_w_, stem_b_, in->ne[1]);
    size_t bi = 0;
    for (size_t s = 0; s < C_.size(); ++s) {
        if (s > 0) {
            // Transposed conv, K = 2*stride: each frame's second half overlaps
            // the next frame's first half. The last frame's second half is
            // carried to the next call; at the end of the stream it is
            // dropped, which is the causal trim of K - stride samples.
            const int64_t Tc = h->ne[1];
            const int     sr = stride_[s], Co = C_[s];
            struct ggml_tensor* y  = ggml_reshape_3d(g, ggml_mul_mat(g, tr_w_[s], h), Co, 2 * sr, Tc);
            struct ggml_tensor* A  = ggml_view_3d(g, y, Co, sr, Tc, y->nb[1], y->nb[2], 0);
            struct ggml_tensor* Bc = ggml_cont(g, ggml_view_3d(g, y, Co, sr, Tc, y->nb[1], y->nb[2],
                                                               static_cast<size_t>(sr) * y->nb[1]));
            struct ggml_tensor* Bprev = Tc > 1
                ? ggml_concat(g, st.tr[s], ggml_view_3d(g, Bc, Co, sr, Tc - 1, Bc->nb[1], Bc->nb[2], 0), 2)
                : st.tr[s];
            writes->push_back(ggml_cpy(g, ggml_view_3d(g, Bc, Co, sr, 1, Bc->nb[1], Bc->nb[2],
                                                       static_cast<size_t>(Tc - 1) * Bc->nb[2]), st.tr[s]));
            h = ggml_reshape_2d(g, ggml_add(g, ggml_cont(g, A), Bprev), Co, sr * Tc);
            if (tr_b_[s]) h = ggml_add(g, h, tr_b_[s]);
        }
        for (int b = 0; b < depth_[s]; ++b, ++bi) {
            const Block& blk = blocks_[bi];
            const int64_t Tc = h->ne[1];
            // mixer: norm -> depthwise causal conv -> gamma -> residual
            struct ggml_tensor* xin  = with_context(norm_mul(g, h, blk.norm, eps_), st.dw[bi]);
            struct ggml_tensor* conv = nullptr;
            for (int k = 0; k < kTaps; ++k) {
                struct ggml_tensor* wk = ggml_view_1d(g, blk.dw, C_[s], static_cast<size_t>(k) * blk.dw->nb[1]);
                struct ggml_tensor* m  = ggml_mul(g, shift(xin, k, Tc), wk);
                conv = conv ? ggml_add(g, conv, m) : m;
            }
            if (blk.mixer_bias) conv = ggml_add(g, conv, blk.mixer_bias);
            if (blk.gamma)      conv = ggml_mul(g, conv, blk.gamma);
            h = ggml_add(g, h, conv);
            // ffn: norm -> linear -> GELU -> linear -> gamma -> residual
            struct ggml_tensor* x = ggml_mul_mat(g, blk.lin1, norm_mul(g, h, blk.ffn_norm, eps_));
            if (blk.lin1_b) x = ggml_add(g, x, blk.lin1_b);
            x = ggml_mul_mat(g, blk.lin2, ggml_gelu(g, x));
            if (blk.lin2_b)    x = ggml_add(g, x, blk.lin2_b);
            if (blk.ffn_gamma) x = ggml_mul(g, x, blk.ffn_gamma);
            h = ggml_add(g, h, x);
        }
    }
    if (final_norm_) h = norm_mul(g, h, final_norm_, eps_);
    return dense7(with_context(h, st.head), head_w_, head_b_, h->ne[1]);   // [1, T_audio]
}

bool AcousticDecoder::decode(State& st, const float* latents, int n_frames, std::vector<float>* audio,
                             int chunk_frames) const {
    if (!ready_ || !st.buf || !latents || !audio || n_frames <= 0) return false;
    if (chunk_frames <= 0) chunk_frames = 32;
    for (int f0 = 0; f0 < n_frames; f0 += chunk_frames) {
        const int T = std::min(chunk_frames, n_frames - f0);
        // The input gets its own buffer so the allocator can plan (and reuse)
        // every intermediate of the graph.
        struct ggml_init_params ipi {};
        ipi.mem_size = ggml_tensor_overhead() * 2;
        ipi.no_alloc = true;
        struct ggml_context* ctx_in = ggml_init(ipi);
        struct ggml_tensor*  in     = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, c_in_, T);
        ggml_backend_buffer_t in_buf = allocate_ctx_tensors(ctx_in);
        if (!in_buf) {
            ggml_free(ctx_in);
            return false;
        }
        ggml_backend_tensor_set(in, latents + static_cast<size_t>(f0) * c_in_, 0, sizeof(float) * c_in_ * T);

        struct ggml_init_params ipg {};
        ipg.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
        ipg.no_alloc = true;
        struct ggml_context* g = ggml_init(ipg);
        std::vector<struct ggml_tensor*> writes;
        struct ggml_tensor* y  = build(g, st, in, &writes);
        struct ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);
        ggml_build_forward_expand(gf, y);
        // State updates after the output, so every read precedes its write.
        for (auto* wr : writes) ggml_build_forward_expand(gf, wr);

        const bool ok = compute_graph(gf);
        if (ok) {
            const size_t n = static_cast<size_t>(ggml_nelements(y)), off = audio->size();
            audio->resize(off + n);
            ggml_backend_tensor_get(y, audio->data() + off, 0, sizeof(float) * n);
        }
        ggml_free(g);
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx_in);
        if (!ok) {
            LV_LOG_ERROR("decoder: compute failed");
            return false;
        }
    }
    return true;
}

}  // namespace lv
