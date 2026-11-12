// End-to-end synthesis with a real model. Needs LIVEVOICE_MODEL and
// LIVEVOICE_VOICE; skips otherwise.
//
// Checks that a sentence comes out as plausible speech (length, level, no
// clipping), in several chunks, at the advertised rate, that the same seed
// gives the same samples, and that the first chunk, after the lead-in trim,
// is a first chunk long (a sliver there stalls playback).
#include "generate.hpp"
#include "model.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const char* model_path = std::getenv("LIVEVOICE_MODEL");
    const char* voice_path = std::getenv("LIVEVOICE_VOICE");
    if (!model_path || !*model_path || !voice_path || !*voice_path) {
        std::fprintf(stderr, "skip: set LIVEVOICE_MODEL and LIVEVOICE_VOICE\n");
        return 77;
    }
    lv::Model model;
    lv::Voice voice;
    if (!model.load(model_path) || !voice.load(voice_path, model)) { std::fprintf(stderr, "FAIL: load\n"); return 1; }

    lv_params p;
    lv_params_init(&p);
    p.seed = 1234;
    const std::string text = "The quick brown fox jumps over the lazy dog.";
    int first = 0;
    auto run = [&](std::vector<float>* pcm, int* chunks) {
        *chunks = 0;
        return lv::synthesize(model, voice, text, p, [&](const float* s, int n) {
            if (*chunks == 0) first = n;
            pcm->insert(pcm->end(), s, s + n);
            ++*chunks;
            return true;
        });
    };
    std::vector<float> a, b;
    int ca = 0, cb = 0;
    if (run(&a, &ca) != LV_OK || run(&b, &cb) != LV_OK) { std::fprintf(stderr, "FAIL: synthesize\n"); return 2; }

    const int rate = lv::output_sample_rate(model, p);
    double ss = 0, peak = 0;
    for (float v : a) { ss += static_cast<double>(v) * v; peak = std::max(peak, static_cast<double>(std::fabs(v))); }
    const double secs = static_cast<double>(a.size()) / rate, rms = std::sqrt(ss / std::max<size_t>(a.size(), 1));
    std::printf("%.2f s at %d Hz in %d chunks, rms %.4f, peak %.3f\n", secs, rate, ca, rms, peak);
    if (secs < 1.5 || secs > 8.0) { std::fprintf(stderr, "FAIL: implausible length\n"); return 3; }
    if (rms < 0.01 || peak >= 1.0) { std::fprintf(stderr, "FAIL: silent or clipped\n"); return 4; }
    if (ca < 2) { std::fprintf(stderr, "FAIL: expected several chunks\n"); return 5; }
    if (rate != (model.dfn.ready ? 48000 : 24000)) { std::fprintf(stderr, "FAIL: rate\n"); return 6; }
    if (a != b) { std::fprintf(stderr, "FAIL: same seed, different audio\n"); return 7; }

    // Where the sound starts in a chunk varies with the seed; before the
    // first audio was held, most seeds started with less than a first chunk.
    const int first_min = p.first_chunk_frames * rate * 2 / 15;   // 7.5 frames per second
    for (uint32_t seed = 1; seed <= 6; ++seed) {
        p.seed = seed;
        std::vector<float> c;
        int cc = 0;
        if (run(&c, &cc) != LV_OK) { std::fprintf(stderr, "FAIL: synthesize\n"); return 2; }
        std::printf("seed %u: first chunk %.3f s\n", seed, static_cast<double>(first) / rate);
        if (first < first_min && cc > 1) { std::fprintf(stderr, "FAIL: first chunk shorter than first_chunk_frames\n"); return 8; }
    }
    std::printf("OK\n");
    return 0;
}
