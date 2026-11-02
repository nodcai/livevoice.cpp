// livevoice-cli: text to a WAV file through the public C API.
//
//   livevoice-cli --model M.gguf --voice V.gguf --text "Hello." --out hello.wav
//   llm ... | livevoice-cli --model M.gguf --voice V.gguf --stream --out reply.wav
//
// With --stream the text is pushed into a streaming session as it arrives
// (word by word for --text, as read for stdin) and the chunk timeline is
// printed.

#include "livevoice.h"
#include "wav.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::printf(
        "usage: %s --model <gguf> --voice <gguf> [--text <s> | --text-file <path>] [options]\n"
        "\n"
        "Reads the text from stdin when neither --text nor --text-file is given.\n"
        "\n"
        "  --out <path>              output WAV (default out.wav)\n"
        "  --stream                  push the text through a streaming session and\n"
        "                            print when each chunk arrives\n"
        "  --steps N                 diffusion steps per frame (default 5)\n"
        "  --cfg X                   guidance scale, <= 1 turns it off (default 1.5)\n"
        "  --seed N                  noise seed (default random)\n"
        "  --max-frames N            stop after N speech frames, 7.5 per second (default 1500)\n"
        "  --first-chunk N           first chunk in frames (default 3)\n"
        "  --lead-chunk N            hold chunks at N frames until one has speech (default off)\n"
        "  --no-postfilter           skip the DFN3 post-filter (output 24 kHz)\n"
        "  --no-trim                 keep the lead-in silence\n"
        "  --merge-cap N             refuse BPE merges into ids >= N, 0 = off (default 80000)\n"
        "  --no-canonical-first-word keep the plain encoding of a text-initial word\n"
        "  --last-word-merge-cap N   merge cap for the last word, 0 = off (default 20000)\n"
        "  --verbose                 debug logging\n"
        "  --quiet                   warnings and errors only\n"
        "  --version                 print the version\n",
        argv0);
}

bool read_all(FILE* f, std::string* out) {
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    return !std::ferror(f);
}

double now_ms() {
    using clk = std::chrono::steady_clock;
    static const auto t0 = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Sink {
    std::vector<float> pcm;
    int    rate   = 0;
    int    chunks = 0;
    double t0     = 0;
    bool   timeline = false;
};

int on_audio(const float* pcm, int n, void* user) {
    auto* s = static_cast<Sink*>(user);
    s->pcm.insert(s->pcm.end(), pcm, pcm + n);
    if (s->timeline)
        std::fprintf(stderr, "chunk %2d at %7.1f ms: %6d samples, %.2f s of audio so far\n", s->chunks,
                     now_ms() - s->t0, n, static_cast<double>(s->pcm.size()) / s->rate);
    ++s->chunks;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, voice_path, text, text_file, out_path = "out.wav";
    bool have_text = false, stream = false;
    lv_params p;
    lv_params_init(&p);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--model")       model_path = value();
        else if (a == "--voice")       voice_path = value();
        else if (a == "--text")        { text = value(); have_text = true; }
        else if (a == "--text-file")   text_file = value();
        else if (a == "--out")         out_path = value();
        else if (a == "--stream")      stream = true;
        else if (a == "--steps")       p.n_steps = std::atoi(value());
        else if (a == "--cfg")         p.cfg_scale = static_cast<float>(std::atof(value()));
        else if (a == "--seed")        p.seed = static_cast<uint32_t>(std::strtoul(value(), nullptr, 10));
        else if (a == "--max-frames")  p.max_frames = std::atoi(value());
        else if (a == "--first-chunk") p.first_chunk_frames = std::atoi(value());
        else if (a == "--lead-chunk")  p.lead_chunk_frames = std::atoi(value());
        else if (a == "--no-postfilter") p.postfilter = 0;
        else if (a == "--no-trim")     p.trim_lead = 0;
        else if (a == "--merge-cap")   p.merge_cap = std::atoi(value());
        else if (a == "--no-canonical-first-word") p.canonical_first_word = 0;
        else if (a == "--last-word-merge-cap") p.last_word_merge_cap = std::atoi(value());
        else if (a == "--verbose")     lv_set_log_level(LV_LOG_DEBUG);
        else if (a == "--quiet")       lv_set_log_level(LV_LOG_WARN);
        else if (a == "--version")     { std::printf("%s\n", lv_version()); return 0; }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown option %s (see --help)\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || voice_path.empty()) {
        usage(argv[0]);
        return 1;
    }
    if (!have_text && !text_file.empty()) {
        FILE* f = std::fopen(text_file.c_str(), "rb");
        if (!f || !read_all(f, &text)) { std::fprintf(stderr, "cannot read %s\n", text_file.c_str()); return 1; }
        std::fclose(f);
        have_text = true;
    }

    lv_model* model = lv_model_load(model_path.c_str());
    if (!model) return 2;
    lv_voice* voice = lv_voice_load(model, voice_path.c_str());
    if (!voice) return 2;

    Sink sink;
    sink.rate     = lv_sample_rate(model, &p);
    sink.timeline = stream;
    sink.t0       = now_ms();
    int rc;
    if (stream) {
        lv_stream* s = lv_stream_begin(model, voice, &p, on_audio, &sink);
        if (!s) return 3;
        if (have_text) {
            // Word by word, the way an LLM delivers it.
            for (size_t pos = 0; pos < text.size();) {
                size_t ws = text.find(' ', pos);
                ws = ws == std::string::npos ? text.size() : ws + 1;
                lv_stream_push(s, text.substr(pos, ws - pos).c_str());
                pos = ws;
            }
        } else {
            char buf[256];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf) - 1, stdin)) > 0) {
                buf[n] = '\0';
                lv_stream_push(s, buf);
            }
        }
        lv_stream_end(s);
        while (!lv_stream_done(s)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        rc = lv_stream_result(s);
        lv_stream_free(s);
        std::fprintf(stderr, "stream done in %.1f ms, %d chunks\n", now_ms() - sink.t0, sink.chunks);
    } else {
        if (!have_text && !read_all(stdin, &text)) { std::fprintf(stderr, "cannot read stdin\n"); return 1; }
        rc = lv_synthesize(model, voice, text.c_str(), &p, on_audio, &sink);
    }
    if (rc != LV_OK) {
        std::fprintf(stderr, "synthesis failed (%d)\n", rc);
        return 4;
    }
    const double secs = static_cast<double>(sink.pcm.size()) / sink.rate;
    std::fprintf(stderr, "%.2f s of audio at %d Hz in %.1f ms\n", secs, sink.rate, now_ms() - sink.t0);
    if (!lv::write_wav(out_path, sink.pcm.data(), sink.pcm.size(), sink.rate)) return 5;
    std::fprintf(stderr, "wrote %s\n", out_path.c_str());
    lv_voice_free(voice);
    lv_model_free(model);
    return 0;
}
