/* livevoice.h - C API for VibeVoice-Realtime-0.5B text-to-speech. */

#ifndef LIVEVOICE_H
#define LIVEVOICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(LIVEVOICE_SHARED)
#  ifdef LIVEVOICE_BUILD
#    define LV_API __declspec(dllexport)
#  else
#    define LV_API __declspec(dllimport)
#  endif
#else
#  define LV_API
#endif

/* ---------- status codes ---------- */
typedef enum {
    LV_OK              =  0,
    LV_ERR_GENERIC     = -1,
    LV_ERR_INVALID_ARG = -2,
    LV_ERR_FILE        = -3,
    LV_ERR_FORMAT      = -4,
    LV_ERR_TOKENIZER   = -5,
    LV_ERR_CAPACITY    = -6,
    LV_ERR_COMPUTE     = -7
} lv_status;

/* ---------- logging ---------- */
typedef enum {
    LV_LOG_ERROR = 0,
    LV_LOG_WARN  = 1,
    LV_LOG_INFO  = 2,
    LV_LOG_DEBUG = 3
} lv_log_level;

/* `msg` is one complete line, newline included. ggml's own messages are
   routed through the same callback. NULL restores logging to stderr. */
typedef void (*lv_log_cb)(lv_log_level level, const char* msg, void* user);

LV_API void        lv_set_log_callback(lv_log_cb cb, void* user);
/* Messages above `level` are dropped. Default LV_LOG_INFO. */
LV_API void        lv_set_log_level(lv_log_level level);
LV_API const char* lv_version(void);

#ifdef __cplusplus
}
#endif

#endif /* LIVEVOICE_H */
