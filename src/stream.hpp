#ifndef LIVEVOICE_STREAM_HPP
#define LIVEVOICE_STREAM_HPP

// Streaming session: text is pushed while an LLM writes it, audio comes
// back per chunk from a worker thread.
//
// The worker tokenizes the stable prefix of what was pushed: everything up
// to the last whitespace, because Qwen2's byte-level BPE folds a space into
// the following word, and a trailing word is held back until the next one
// (or end()) shows whether it is the last. On end() the whole text plus the
// terminating "\n" is tokenized exactly like a one-shot synthesis, so a
// stream and a synthesis of the same text and seed give the same samples.

#include "generate.hpp"
#include "model.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace lv {

class Stream {
public:
    // Runs on the worker thread for every chunk; false stops the stream.
    // `pcm` is valid only for the call.
    using AudioCb = std::function<bool(const float* pcm, int n)>;

    Stream() = default;
    ~Stream() { join(); }
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    // `model` and `voice` must outlive the stream.
    bool begin(const Model& model, const Voice& voice, const lv_params& p, AudioCb on_audio);
    // Appends UTF-8 text; ignored after end().
    void push(const std::string& utf8);
    // No more text; the worker drains and finishes.
    void end();
    // Stops as soon as possible; no further audio is delivered.
    void abort();
    bool done() const;
    // The generation result (an lv_status), valid once done().
    int  result() const;
    int  sample_rate() const { return rate_; }
    // end(), then waits for the worker.
    void join();

private:
    void worker();

    const Model* model_ = nullptr;
    const Voice* voice_ = nullptr;
    lv_params    params_{};
    AudioCb      on_audio_;
    std::thread  thread_;
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::string       text_;
    size_t            handed_out_ = 0;   // tokens already given to the loop
    bool              ended_ = false, done_ = false;
    std::atomic<bool> aborted_{false};
    int               rc_   = LV_OK;
    int               rate_ = 0;
};

}  // namespace lv

#endif  // LIVEVOICE_STREAM_HPP
