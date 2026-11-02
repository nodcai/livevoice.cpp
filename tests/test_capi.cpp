// The C API end to end. Needs LIVEVOICE_MODEL and LIVEVOICE_VOICE; skips
// otherwise.
#include "livevoice.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

struct Sink {
    std::vector<float> pcm;
    int chunks = 0;
};

int collect(const float* pcm, int n, void* user) {
    auto* s = static_cast<Sink*>(user);
    s->pcm.insert(s->pcm.end(), pcm, pcm + n);
    ++s->chunks;
    return 0;
}

int stop_after_one(const float*, int, void* user) {
    ++*static_cast<int*>(user);
    return 1;
}

}  // namespace

int main() {
    const char* model_path = std::getenv("LIVEVOICE_MODEL");
    const char* voice_path = std::getenv("LIVEVOICE_VOICE");
    if (!model_path || !*model_path || !voice_path || !*voice_path) {
        std::fprintf(stderr, "skip: set LIVEVOICE_MODEL and LIVEVOICE_VOICE\n");
        return 77;
    }
    lv_set_log_level(LV_LOG_WARN);
    if (lv_model_load("does-not-exist.gguf")) { std::fprintf(stderr, "FAIL: loaded a missing file\n"); return 1; }
    lv_model* model = lv_model_load(model_path);
    lv_voice* voice = model ? lv_voice_load(model, voice_path) : nullptr;
    if (!model || !voice) { std::fprintf(stderr, "FAIL: load\n"); return 2; }
    std::printf("backend %s, post-filter %d, rate %d\n", lv_backend_name(), lv_model_has_postfilter(model),
                lv_sample_rate(model, nullptr));

    lv_params p;
    lv_params_init(&p);
    p.seed = 99;
    const char* text = "Hello there! Welcome to the session test.";

    Sink whole;
    if (lv_synthesize(model, voice, text, &p, collect, &whole) != LV_OK || whole.pcm.empty()) {
        std::fprintf(stderr, "FAIL: synthesize\n");
        return 3;
    }
    int calls = 0;
    if (lv_synthesize(model, voice, text, &p, stop_after_one, &calls) != LV_OK || calls != 1) {
        std::fprintf(stderr, "FAIL: callback stop (%d calls)\n", calls);
        return 4;
    }
    if (lv_synthesize_wav(model, voice, text, &p, "test_capi.wav") != LV_OK) { std::fprintf(stderr, "FAIL: wav\n"); return 5; }
    std::remove("test_capi.wav");

    for (int pf = 1; pf >= 0; --pf) {
        p.postfilter = pf;
        Sink sink;
        lv_stream* s = lv_stream_begin(model, voice, &p, collect, &sink);
        if (!s) { std::fprintf(stderr, "FAIL: stream begin\n"); return 6; }
        for (const char* piece : {"Hello there! ", "Welcome to ", "the session test."}) {
            if (lv_stream_push(s, piece) != LV_OK) { std::fprintf(stderr, "FAIL: push\n"); return 7; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (lv_stream_done(s)) { std::fprintf(stderr, "FAIL: done before end\n"); return 8; }
        lv_stream_end(s);
        if (lv_stream_push(s, "more") != LV_ERR_INVALID_ARG) { std::fprintf(stderr, "FAIL: push after end\n"); return 9; }
        for (int i = 0; i < 600 && !lv_stream_done(s); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int rate = lv_stream_sample_rate(s), done = lv_stream_done(s), rc = lv_stream_result(s);
        lv_stream_free(s);
        std::printf("postfilter %d: %d Hz, %d chunks, %.2f s\n", pf, rate, sink.chunks,
                    static_cast<double>(sink.pcm.size()) / rate);
        if (!done || rc != LV_OK || sink.chunks < 2) { std::fprintf(stderr, "FAIL: stream\n"); return 10; }
        if (pf && sink.pcm != whole.pcm) { std::fprintf(stderr, "FAIL: stream differs from synthesis\n"); return 11; }
        if (rate != (pf && lv_model_has_postfilter(model) ? 48000 : 24000)) { std::fprintf(stderr, "FAIL: rate\n"); return 12; }
    }
    lv_voice_free(voice);
    lv_model_free(model);
    std::printf("OK\n");
    return 0;
}
