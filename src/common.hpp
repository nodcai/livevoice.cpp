#ifndef LIVEVOICE_COMMON_HPP
#define LIVEVOICE_COMMON_HPP

#include "livevoice.h"

namespace lv {

void log(lv_log_level lvl, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#define LV_LOG_ERROR(...) ::lv::log(LV_LOG_ERROR, __VA_ARGS__)
#define LV_LOG_WARN(...)  ::lv::log(LV_LOG_WARN,  __VA_ARGS__)
#define LV_LOG_INFO(...)  ::lv::log(LV_LOG_INFO,  __VA_ARGS__)
#define LV_LOG_DEBUG(...) ::lv::log(LV_LOG_DEBUG, __VA_ARGS__)

// Routes ggml's own messages through the same logger and level.
void route_ggml_log();

}  // namespace lv

#endif  // LIVEVOICE_COMMON_HPP
