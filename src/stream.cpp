#include "stream.hpp"

#include <algorithm>

namespace lv {

namespace {
bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
}  // namespace

bool Stream::begin(const Model& model, const Voice& voice, const lv_params& p, AudioCb on_audio) {
    if (!on_audio || thread_.joinable()) return false;
    model_    = &model;
    voice_    = &voice;
    params_   = p;
    on_audio_ = std::move(on_audio);
    rate_     = output_sample_rate(model, p);
    text_.clear();
    handed_out_ = 0;
    ended_ = done_ = false;
    aborted_.store(false);
    rc_ = LV_OK;
    thread_ = std::thread([this] { worker(); });
    return true;
}

void Stream::push(const std::string& utf8) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (ended_) return;
        text_ += utf8;
    }
    cv_.notify_all();
}

void Stream::end() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        ended_ = true;
    }
    cv_.notify_all();
}

void Stream::abort() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        aborted_.store(true);
        ended_ = true;
    }
    cv_.notify_all();
}

bool Stream::done() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_;
}

int Stream::result() const {
    std::lock_guard<std::mutex> lk(mu_);
    return rc_;
}

void Stream::join() {
    end();
    if (thread_.joinable()) thread_.join();
}

void Stream::worker() {
    const Tokenizer::EncodeOptions enc = encode_options(params_);
    Tokenizer::EncodeOptions enc_prefix = enc;   // a prefix has no last word yet
    enc_prefix.last_word_max_merge_token_id = 0;

    GenerateIO io;
    io.pull_text = [&](int max_tokens, std::vector<int32_t>* out, bool* eof) {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            // Re-tokenize everything up to the stable cut, so the ids match a
            // one-shot encoding of the same text, and hand out the new ones.
            std::vector<int32_t> toks;
            if (ended_) {
                if (std::any_of(text_.begin(), text_.end(), [](char c) { return !is_ws(c); }))
                    toks = model_->tokenizer.encode(terminate_text(text_), enc);
            } else {
                // Cut at the last whitespace, not after it: the space merges
                // into the next word's token once that word arrives. Trailing
                // whitespace does not count.
                size_t stable = text_.size();
                while (stable > 0 && is_ws(text_[stable - 1])) --stable;
                const size_t ws = stable > 0 ? text_.find_last_of(" \t\n", stable - 1) : std::string::npos;
                if (ws != std::string::npos && ws > 0) toks = model_->tokenizer.encode(text_.substr(0, ws), enc_prefix);
            }
            if (handed_out_ < toks.size()) {
                const size_t n = std::min(toks.size() - handed_out_, static_cast<size_t>(std::max(1, max_tokens)));
                out->insert(out->end(), toks.begin() + static_cast<long>(handed_out_),
                            toks.begin() + static_cast<long>(handed_out_ + n));
                handed_out_ += n;
                *eof = ended_ && handed_out_ >= toks.size();
                return;
            }
            if (ended_ || aborted_.load()) {
                *eof = true;
                return;
            }
            cv_.wait(lk);
        }
    };
    io.emit_audio   = [this](const float* pcm, int n) { return !aborted_.load() && on_audio_(pcm, n); };
    io.should_abort = [this] { return aborted_.load(); };

    const int rc = generate(*model_, *voice_, params_, io);
    {
        std::lock_guard<std::mutex> lk(mu_);
        rc_   = rc;
        done_ = true;
    }
    cv_.notify_all();
}

}  // namespace lv
