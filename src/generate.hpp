#ifndef LIVEVOICE_GENERATE_HPP
#define LIVEVOICE_GENERATE_HPP

// The realtime generation loop.
//
// Text goes in five tokens at a time; after each text window up to six
// speech frames are generated, until the EOS classifier fires. Audio leaves
// in chunks while generation continues. The same loop serves a text that
// is known up front (synthesize) and one that is still being written by an
// LLM (Stream): only the text source differs.

#include "model.hpp"

#include "livevoice.h"

#include <functional>
#include <string>
#include <vector>

namespace lv {

struct GenerateIO {
    // Blocks until it can return at least one token or the text has ended
    // (*eof); may return fewer than `max_tokens`.
    std::function<void(int max_tokens, std::vector<int32_t>* out, bool* eof)> pull_text;
    // Receives every chunk of audio; false stops generation.
    std::function<bool(const float* pcm, int n)> emit_audio;
    // Optional; checked once per speech frame.
    std::function<bool()> should_abort;
};

// The tokenizer options the parameters ask for.
Tokenizer::EncodeOptions encode_options(const lv_params& p);

// A complete text as the model wants it: trailing whitespace replaced by
// the "\n" that ends the user turn.
std::string terminate_text(const std::string& text);

// 48 kHz with the post-filter on and present, else the model's 24 kHz.
int output_sample_rate(const Model& model, const lv_params& p);

// Returns LV_OK, or a negative lv_status. Generation on the backend is
// serialised process-wide: a second call waits for the first.
int generate(const Model& model, const Voice& voice, const lv_params& p, const GenerateIO& io);

// The whole text at once; `on_audio` returns false to stop.
int synthesize(const Model& model, const Voice& voice, const std::string& text, const lv_params& p,
               const std::function<bool(const float* pcm, int n)>& on_audio);

}  // namespace lv

#endif  // LIVEVOICE_GENERATE_HPP
