// Stateful acoustic decoder against a plain C++ reference.
//
// Writes a small random decoder in the converter's tensor naming to a gguf,
// loads it like a model, and checks:
//   1. one decode call == a direct causal conv / transposed conv / Block1D
//      implementation of the decoder
//   2. decoding in pieces (5 + 4 frames, 3-frame graphs) == one call
//   3. a reset state reproduces the first output
#include "acoustic_decoder.hpp"
#include "gguf_loader.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using Vec = std::vector<float>;
constexpr int   kIn = 8, kFrames = 9;
constexpr float kEps = 1e-5f;
const std::vector<int> kC = {16, 8, 4}, kStride = {0, 2, 3}, kDepth = {2, 1, 1};

struct Tensor { std::vector<int64_t> ne; Vec data; };
std::map<std::string, Tensor> g_w;

const Vec& W(const std::string& n) { return g_w.at("at.dec." + n).data; }

// x is [C][T]; returns [Co][T].
std::vector<Vec> conv7(const std::vector<Vec>& x, const Vec& w, const Vec& b, int Co) {
    const int Ci = static_cast<int>(x.size()), T = static_cast<int>(x[0].size());
    std::vector<Vec> y(Co, Vec(T, 0.0f));
    for (int co = 0; co < Co; ++co)
        for (int t = 0; t < T; ++t) {
            float acc = b[co];
            for (int ci = 0; ci < Ci; ++ci)
                for (int k = 0; k < 7; ++k) {
                    const int src = t + k - 6;
                    if (src >= 0) acc += w[(static_cast<size_t>(co) * Ci + ci) * 7 + k] * x[ci][src];
                }
            y[co][t] = acc;
        }
    return y;
}

std::vector<Vec> conv_transposed(const std::vector<Vec>& x, const Vec& w, const Vec& b, int Co, int s) {
    const int Ci = static_cast<int>(x.size()), T = static_cast<int>(x[0].size()), K = 2 * s;
    std::vector<Vec> y(Co, Vec(static_cast<size_t>(T) * s + s, 0.0f));
    for (int ci = 0; ci < Ci; ++ci)
        for (int co = 0; co < Co; ++co)
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < K; ++k) y[co][t * s + k] += x[ci][t] * w[(static_cast<size_t>(ci) * Co + co) * K + k];
    for (int co = 0; co < Co; ++co) {
        y[co].resize(static_cast<size_t>(T) * s);   // causal trim of K - s on the right
        for (auto& v : y[co]) v += b[co];
    }
    return y;
}

std::vector<Vec> rmsnorm_c(const std::vector<Vec>& x, const Vec& w) {
    std::vector<Vec> y = x;
    const int C = static_cast<int>(x.size()), T = static_cast<int>(x[0].size());
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int c = 0; c < C; ++c) ss += static_cast<double>(x[c][t]) * x[c][t];
        const float sc = 1.0f / std::sqrt(static_cast<float>(ss / C) + kEps);
        for (int c = 0; c < C; ++c) y[c][t] = x[c][t] * sc * w[c];
    }
    return y;
}

float gelu(float v) { return 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v))); }

Vec reference(const Vec& lat) {
    std::vector<Vec> h(kIn, Vec(kFrames));
    for (int t = 0; t < kFrames; ++t)
        for (int c = 0; c < kIn; ++c) h[c][t] = lat[static_cast<size_t>(t) * kIn + c];
    h = conv7(h, W("stem.weight"), W("stem.bias"), kC[0]);
    for (size_t s = 0; s < kC.size(); ++s) {
        if (s > 0) {
            const std::string u = "up_" + std::to_string(s);
            h = conv_transposed(h, W(u + ".weight"), W(u + ".bias"), kC[s], kStride[s]);
        }
        const int C = kC[s], T = static_cast<int>(h[0].size());
        for (int b = 0; b < kDepth[s]; ++b) {
            const std::string p = "stage_" + std::to_string(s) + "_block_" + std::to_string(b) + ".weight.";
            const auto xn = rmsnorm_c(h, W(p + "norm"));
            const Vec& dw = W(p + "mixer_weight");
            for (int c = 0; c < C; ++c)
                for (int t = 0; t < T; ++t) {
                    float acc = W(p + "mixer_bias")[c];
                    for (int k = 0; k < 7; ++k)
                        if (t + k - 6 >= 0) acc += dw[static_cast<size_t>(c) * 7 + k] * xn[c][t + k - 6];
                    h[c][t] += W(p + "gamma")[c] * acc;
                }
            const auto fn = rmsnorm_c(h, W(p + "ffn_norm"));
            const int F = 4 * C;
            const Vec &l1 = W(p + "ffn_linear1"), &l2 = W(p + "ffn_linear2");
            for (int t = 0; t < T; ++t) {
                Vec a(F);
                for (int f = 0; f < F; ++f) {
                    float acc = W(p + "ffn_linear1_bias")[f];
                    for (int c = 0; c < C; ++c) acc += l1[static_cast<size_t>(f) * C + c] * fn[c][t];
                    a[f] = gelu(acc);
                }
                for (int c = 0; c < C; ++c) {
                    float acc = W(p + "ffn_linear2_bias")[c];
                    for (int f = 0; f < F; ++f) acc += l2[static_cast<size_t>(c) * F + f] * a[f];
                    h[c][t] += W(p + "ffn_gamma")[c] * acc;
                }
            }
        }
    }
    return conv7(h, W("head.weight"), W("head.bias"), 1)[0];
}

double rel_err(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) return 1e30;
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        r += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(d / std::max(r, 1e-30));
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // fp16-representable values, so the decoder's F16 kernels see the same numbers.
    auto add = [&](const std::string& n, std::vector<int64_t> ne, float scale, float offset = 0.0f) {
        size_t count = 1;
        for (auto e : ne) count *= static_cast<size_t>(e);
        Vec v(count);
        for (auto& x : v) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(rng) * scale + offset));
        g_w["at.dec." + n] = {ne, v};
    };
    add("stem.weight", {7, kIn, kC[0]}, 0.3f);
    add("stem.bias", {kC[0]}, 0.1f);
    for (size_t s = 1; s < kC.size(); ++s) {
        add("up_" + std::to_string(s) + ".weight", {2 * kStride[s], kC[s], kC[s - 1]}, 0.3f);
        add("up_" + std::to_string(s) + ".bias", {kC[s]}, 0.1f);
    }
    for (size_t s = 0; s < kC.size(); ++s)
        for (int b = 0; b < kDepth[s]; ++b) {
            const std::string p = "stage_" + std::to_string(s) + "_block_" + std::to_string(b) + ".weight.";
            const int C = kC[s];
            add(p + "norm", {C}, 0.1f, 1.0f);
            add(p + "mixer_weight", {7, 1, C}, 0.3f);
            add(p + "mixer_bias", {C}, 0.1f);
            add(p + "gamma", {C}, 0.1f, 0.5f);
            add(p + "ffn_norm", {C}, 0.1f, 1.0f);
            add(p + "ffn_linear1", {C, 4 * C}, 0.3f);
            add(p + "ffn_linear1_bias", {4 * C}, 0.1f);
            add(p + "ffn_linear2", {4 * C, C}, 0.3f);
            add(p + "ffn_linear2_bias", {C}, 0.1f);
            add(p + "ffn_gamma", {C}, 0.1f, 0.5f);
        }
    add("head.weight", {7, kC.back(), 1}, 0.3f);
    add("head.bias", {1}, 0.1f);

    // ---- write and load as a model file would be ----
    const std::string path = "test_decoder.gguf";
    {
        gguf_context* gg = gguf_init_empty();
        struct ggml_init_params ip {};
        ip.mem_size = ggml_tensor_overhead() * (g_w.size() + 1);
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        for (auto& [name, t] : g_w) {
            ggml_tensor* gt = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(t.ne.size()), t.ne.data());
            ggml_set_name(gt, name.c_str());
            gt->data = t.data.data();
            gguf_add_tensor(gg, gt);
        }
        const bool ok = gguf_write_to_file(gg, path.c_str(), false);
        gguf_free(gg);
        ggml_free(ctx);
        if (!ok) { std::fprintf(stderr, "FAIL: write %s\n", path.c_str()); return 1; }
    }
    lv::GgufLoader m;
    lv::AcousticDecoder dec;
    if (!m.load(path) || !dec.load(m, "at.dec", kEps)) { std::fprintf(stderr, "FAIL: load\n"); return 2; }
    std::remove(path.c_str());
    if (dec.hop() != 6 || dec.latent_dim() != kIn) { std::fprintf(stderr, "FAIL: geometry\n"); return 3; }

    Vec lat(static_cast<size_t>(kFrames) * kIn);
    for (auto& v : lat) v = nd(rng);
    const Vec want = reference(lat);

    lv::AcousticDecoder::State st;
    if (!dec.init_state(&st)) return 4;
    Vec whole, pieces, again;
    if (!dec.decode(st, lat.data(), kFrames, &whole)) return 5;
    dec.reset_state(st);
    if (!dec.decode(st, lat.data(), 5, &pieces, 3) || !dec.decode(st, lat.data() + 5 * kIn, 4, &pieces, 3)) return 6;
    dec.reset_state(st);
    if (!dec.decode(st, lat.data(), kFrames, &again)) return 7;

    const double e_ref = rel_err(whole, want), e_chunk = rel_err(pieces, whole), e_reset = rel_err(again, whole);
    std::printf("vs reference %.2e, chunked vs whole %.2e, reset %.2e (%zu samples)\n", e_ref, e_chunk, e_reset,
                whole.size());
    // F16 kernels: the CPU rounds the activations to F16 for those matmuls,
    // and GPUs pick other matmul kernels for other chunk widths.
    const bool ok = e_ref < 5e-3 && e_chunk < 1e-3 && e_reset == 0.0;
    std::printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 8;
}
