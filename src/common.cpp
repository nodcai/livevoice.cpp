#include "common.hpp"

#include "ggml.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace lv {

namespace {

struct LogState {
    std::mutex   mu;
    lv_log_cb    cb      = nullptr;
    void*        user    = nullptr;
    lv_log_level min_lvl = LV_LOG_INFO;
};

LogState& state() {
    static LogState s;
    return s;
}

const char* lvl_str(lv_log_level l) {
    switch (l) {
        case LV_LOG_ERROR: return "E";
        case LV_LOG_WARN:  return "W";
        case LV_LOG_INFO:  return "I";
        case LV_LOG_DEBUG: return "D";
    }
    return "?";
}

void emit(lv_log_level lvl, const char* line) {
    auto& s = state();
    lv_log_cb cb;
    void*     user;
    {
        // Copy under the lock, call outside it: the callback may log again.
        std::lock_guard<std::mutex> lk(s.mu);
        if (lvl > s.min_lvl) return;
        cb   = s.cb;
        user = s.user;
    }
    if (cb) cb(lvl, line, user);
    else    std::fprintf(stderr, "[lv %s] %s", lvl_str(lvl), line);
}

// ggml logs to stderr on its own, which is wrong for a library: a host that
// installed a callback would still get "compiling pipeline: ..." the first
// time Metal builds a kernel, in the middle of a streamed reply. It has to be
// this library's ggml_log_set: a host that links its own ggml cannot reach a
// statically linked copy with hidden symbols from outside.
void ggml_log_forward(ggml_log_level glvl, const char* text, void*) {
    lv_log_level lvl = LV_LOG_INFO;
    switch (glvl) {
        case GGML_LOG_LEVEL_ERROR: lvl = LV_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  lvl = LV_LOG_WARN;  break;
        case GGML_LOG_LEVEL_DEBUG: lvl = LV_LOG_DEBUG; break;
        default:                   break;              // INFO and CONT
    }
    emit(lvl, text);   // already newline-terminated
}

}  // namespace

void route_ggml_log() { ggml_log_set(ggml_log_forward, nullptr); }

void log(lv_log_level lvl, const char* fmt, ...) {
    // One complete line per call, newline included, so a host callback can
    // write the text straight out.
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf) - 2, fmt, ap);
    va_end(ap);
    size_t len = n < 0 ? 0 : std::min(static_cast<size_t>(n), sizeof(buf) - 2);
    buf[len++] = '\n';
    buf[len]   = '\0';
    emit(lvl, buf);
}

}  // namespace lv

extern "C" {

void lv_set_log_callback(lv_log_cb cb, void* user) {
    {
        auto& s = lv::state();
        std::lock_guard<std::mutex> lk(s.mu);
        s.cb   = cb;
        s.user = user;
    }
    // Outside the lock: ggml may log from inside ggml_log_set.
    lv::route_ggml_log();
}

void lv_set_log_level(lv_log_level level) {
    {
        auto& s = lv::state();
        std::lock_guard<std::mutex> lk(s.mu);
        s.min_lvl = level;
    }
    lv::route_ggml_log();
}

const char* lv_version(void) { return "livevoice 0.1.0"; }

}  // extern "C"
