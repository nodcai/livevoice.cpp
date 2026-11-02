// C API over lv::Model, lv::Voice, lv::synthesize and lv::Stream.

#include "livevoice.h"

#include "backend.hpp"
#include "common.hpp"
#include "generate.hpp"
#include "model.hpp"
#include "stream.hpp"
#include "wav.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

struct lv_model {
    lv::Model m;
};

struct lv_voice {
    lv::Voice v;
};

struct lv_stream {
    lv::Stream  st;
    lv_audio_cb cb   = nullptr;
    void*       user = nullptr;
};

namespace {

// Models a host never freed are released at exit, before the backend: ggml
// -metal asserts if a buffer is still resident when its device goes away.
// The handler is registered after the backend's own (which the first load
// triggers), so it runs first.
std::mutex          g_live_mu;
std::set<lv_model*> g_live;

void track(lv_model* m) {
    static std::once_flag once;
    std::call_once(once, [] {
        std::atexit([] {
            std::lock_guard<std::mutex> lk(g_live_mu);
            for (lv_model* m : g_live) delete m;
            g_live.clear();
        });
    });
    std::lock_guard<std::mutex> lk(g_live_mu);
    g_live.insert(m);
}

lv_params params_or_default(const lv_params* p) {
    lv_params d;
    lv_params_init(&d);
    return p ? *p : d;
}

}  // namespace

extern "C" {

lv_model* lv_model_load(const char* path) {
    if (!path || !*path) return nullptr;
    auto m = std::make_unique<lv_model>();
    if (!m->m.load(path)) return nullptr;
    track(m.get());
    return m.release();
}

void lv_model_free(lv_model* model) {
    if (!model) return;
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        g_live.erase(model);
    }
    delete model;
}

int lv_model_has_postfilter(const lv_model* model) { return model && model->m.dfn.ready ? 1 : 0; }

int lv_sample_rate(const lv_model* model, const lv_params* p) {
    return model ? lv::output_sample_rate(model->m, params_or_default(p)) : 0;
}

const char* lv_backend_name(void) { return lv::backend_name(); }

lv_voice* lv_voice_load(const lv_model* model, const char* path) {
    if (!model || !path || !*path) return nullptr;
    auto v = std::make_unique<lv_voice>();
    return v->v.load(path, model->m) ? v.release() : nullptr;
}

void lv_voice_free(lv_voice* voice) { delete voice; }

int lv_synthesize(const lv_model* model, const lv_voice* voice, const char* text, const lv_params* p,
                  lv_audio_cb on_audio, void* user) {
    if (!model || !voice || !text || !on_audio) return LV_ERR_INVALID_ARG;
    return lv::synthesize(model->m, voice->v, text, params_or_default(p),
                          [&](const float* pcm, int n) { return on_audio(pcm, n, user) == 0; });
}

int lv_synthesize_wav(const lv_model* model, const lv_voice* voice, const char* text, const lv_params* p,
                      const char* wav_path) {
    if (!model || !voice || !text || !wav_path) return LV_ERR_INVALID_ARG;
    const lv_params pp = params_or_default(p);
    std::vector<float> pcm;
    const int rc = lv::synthesize(model->m, voice->v, text, pp, [&](const float* s, int n) {
        pcm.insert(pcm.end(), s, s + n);
        return true;
    });
    if (rc != LV_OK) return rc;
    return lv::write_wav(wav_path, pcm.data(), pcm.size(), lv::output_sample_rate(model->m, pp)) ? LV_OK : LV_ERR_FILE;
}

lv_stream* lv_stream_begin(const lv_model* model, const lv_voice* voice, const lv_params* p, lv_audio_cb on_audio,
                           void* user) {
    if (!model || !voice || !on_audio) return nullptr;
    auto s  = std::make_unique<lv_stream>();
    s->cb   = on_audio;
    s->user = user;
    lv_stream* raw = s.get();
    if (!s->st.begin(model->m, voice->v, params_or_default(p),
                     [raw](const float* pcm, int n) { return raw->cb(pcm, n, raw->user) == 0; }))
        return nullptr;
    return s.release();
}

int lv_stream_push(lv_stream* s, const char* utf8) {
    if (!s || !utf8) return LV_ERR_INVALID_ARG;
    if (s->st.ended()) return LV_ERR_INVALID_ARG;
    s->st.push(utf8);
    return LV_OK;
}

void lv_stream_end(lv_stream* s) {
    if (s) s->st.end();
}

void lv_stream_abort(lv_stream* s) {
    if (s) s->st.abort();
}

int lv_stream_done(const lv_stream* s) { return !s || s->st.done() ? 1 : 0; }

int lv_stream_result(const lv_stream* s) { return s ? s->st.result() : LV_ERR_INVALID_ARG; }

int lv_stream_sample_rate(const lv_stream* s) { return s ? s->st.sample_rate() : 0; }

void lv_stream_free(lv_stream* s) {
    if (!s) return;
    s->st.join();
    delete s;
}

}  // extern "C"
