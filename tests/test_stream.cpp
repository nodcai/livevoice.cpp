// Streaming session against one-shot synthesis. Needs LIVEVOICE_MODEL and
// LIVEVOICE_VOICE; skips otherwise.
//
// Text pushed in pieces that split words must produce the same token
// windows as the whole text, so with the same seed the stream's chunks
// concatenate to exactly the synthesized waveform. Also checks done() and
// that abort() returns promptly.
#include "generate.hpp"
#include "model.hpp"
#include "stream.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
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
    p.seed = 4242;

    const std::string text = "Hello there! Welcome to the demo, it should sound quite natural.";
    std::vector<float> whole;
    if (lv::synthesize(model, voice, text, p, [&](const float* s, int n) {
            whole.insert(whole.end(), s, s + n);
            return true;
        }) != LV_OK || whole.empty()) {
        std::fprintf(stderr, "FAIL: synthesize\n");
        return 2;
    }

    std::vector<float> streamed;
    int chunks = 0;
    lv::Stream st;
    if (!st.begin(model, voice, p, [&](const float* s, int n) {
            streamed.insert(streamed.end(), s, s + n);
            ++chunks;
            return true;
        })) { std::fprintf(stderr, "FAIL: begin\n"); return 3; }
    for (const char* piece : {"Hello th", "ere! Wel", "come to the de", "mo, it should", " sound quite natural."}) {
        st.push(piece);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    if (st.done()) { std::fprintf(stderr, "FAIL: done before end()\n"); return 4; }
    st.end();
    st.join();
    if (!st.done() || st.result() != LV_OK) { std::fprintf(stderr, "FAIL: stream rc %d\n", st.result()); return 5; }
    std::printf("synthesized %zu samples, streamed %zu in %d chunks\n", whole.size(), streamed.size(), chunks);
    if (streamed != whole) { std::fprintf(stderr, "FAIL: stream differs from synthesis\n"); return 6; }
    if (chunks < 2) { std::fprintf(stderr, "FAIL: expected several chunks\n"); return 7; }

    lv::Stream st2;
    int cut = 0;
    if (!st2.begin(model, voice, p, [&](const float*, int) { ++cut; return true; })) return 8;
    st2.push("This is a long sentence that will be cut short before it is finished speaking, hopefully.");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto t0 = std::chrono::steady_clock::now();
    st2.abort();
    st2.join();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("abort joined in %.0f ms after %d chunks\n", ms, cut);
    if (ms > 3000) { std::fprintf(stderr, "FAIL: abort too slow\n"); return 9; }
    std::printf("OK\n");
    return 0;
}
