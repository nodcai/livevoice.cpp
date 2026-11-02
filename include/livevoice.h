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

/* ---------- synthesis parameters ----------
   Always start from lv_params_init(); fields may be appended in later
   versions. */
typedef struct {
    int      n_steps;              /* DPM-Solver++ steps per speech frame. 5 */
    float    cfg_scale;            /* classifier-free guidance; <= 1 turns it off. 1.5 */
    uint32_t seed;                 /* 0 = random */
    int      max_frames;           /* speech frames (7.5 per second) before stopping. 1500 */
    int      max_text_tokens;      /* text capacity of a stream's caches. 1024 */

    /* Chunking: the first chunk is small for a low time to first audio,
       later chunks grow by half up to max_chunk_frames, so each one plays
       for longer than the next takes to make. */
    int      first_chunk_frames;   /* 3 (about 400 ms of audio) */
    int      max_chunk_frames;     /* 32 */
    int      lead_chunk_frames;    /* hold chunks at this size until one carried speech. 0 = off */
    int      flush_on_silence;     /* emit pending frames once the latents go silent. 1 */

    /* The negative (guidance) condition drifts as speech frames accumulate,
       audible as the voice losing energy over a long reply; blending the
       voice prompt's state back in by this much holds it steady. 0.2 */
    float    neg_anchor;

    int      postfilter;           /* DFN3 post-filter when the model has it (output 48 kHz). 1 */
    int      trim_lead;            /* drop the lead-in silence before the first sound. 1 */

    /* Text encoding for rare-word pronunciation (see AGENTS.md):
       refuse BPE merges into token ids >= merge_cap (0 = off). 80000 */
    int      merge_cap;
    /* encode a text-initial word as it would split mid-sentence. 1 */
    int      canonical_first_word;
    /* merge cap for the last word of the text (0 = off); the model rushes the
       final word and garbles a rare single-token one. 20000 */
    int      last_word_merge_cap;
} lv_params;

LV_API void lv_params_init(lv_params* p);

/* ---------- models and voices ----------
   A model is one gguf with the weights, the tokenizer and, optionally, the
   DFN3 post-filter. A voice is a gguf with a pre-computed voice prompt. The
   model must outlive its voices and streams. */
typedef struct lv_model  lv_model;
typedef struct lv_voice  lv_voice;
typedef struct lv_stream lv_stream;

LV_API lv_model*   lv_model_load(const char* path);   /* NULL on failure */
LV_API void        lv_model_free(lv_model* model);
LV_API int         lv_model_has_postfilter(const lv_model* model);
/* Rate of the audio delivered with these parameters (NULL: the defaults):
   48000 with the post-filter, else 24000. */
LV_API int         lv_sample_rate(const lv_model* model, const lv_params* p);
/* The ggml backend in use, e.g. "MTL0", "CUDA0", "CPU". */
LV_API const char* lv_backend_name(void);

LV_API lv_voice*   lv_voice_load(const lv_model* model, const char* path);   /* NULL on failure */
LV_API void        lv_voice_free(lv_voice* voice);

/* ---------- synthesis ----------
   Audio is mono float in [-1, 1] at lv_sample_rate(). `pcm` is valid only
   for the call. Return non-zero to stop. Generation is serialised across
   the process: concurrent calls and streams wait for each other. */
typedef int (*lv_audio_cb)(const float* pcm, int n_samples, void* user);

/* The whole text at once, delivered in chunks as it is generated. Returns
   LV_OK (also when the callback stopped it) or an lv_status error. */
LV_API int lv_synthesize(const lv_model* model, const lv_voice* voice, const char* text,
                         const lv_params* p, lv_audio_cb on_audio, void* user);
/* Same, written to a 16-bit WAV. */
LV_API int lv_synthesize_wav(const lv_model* model, const lv_voice* voice, const char* text,
                             const lv_params* p, const char* wav_path);

/* ---------- streaming ----------
   Push text while it is being written (by an LLM, say) and receive audio
   from a worker thread as it is generated. The result is sample for sample
   what lv_synthesize gives for the complete text. */
LV_API lv_stream* lv_stream_begin(const lv_model* model, const lv_voice* voice, const lv_params* p,
                                  lv_audio_cb on_audio, void* user);   /* NULL on failure */
/* Appends UTF-8 text without blocking. LV_ERR_INVALID_ARG after end(). */
LV_API int  lv_stream_push(lv_stream* s, const char* utf8);
/* No more text; the worker finishes what it has. Does not block. */
LV_API void lv_stream_end(lv_stream* s);
/* Stops as soon as possible; no further audio is delivered. */
LV_API void lv_stream_abort(lv_stream* s);
/* Non-zero once all audio was delivered (or the stream was stopped). */
LV_API int  lv_stream_done(const lv_stream* s);
/* LV_OK or an lv_status error, once done. */
LV_API int  lv_stream_result(const lv_stream* s);
LV_API int  lv_stream_sample_rate(const lv_stream* s);
/* Ends the stream if needed, waits for the worker and frees it. */
LV_API void lv_stream_free(lv_stream* s);

#ifdef __cplusplus
}
#endif

#endif /* LIVEVOICE_H */
