// Loads a real model and voice. Needs LIVEVOICE_MODEL and LIVEVOICE_VOICE;
// skips otherwise.
#include "model.hpp"

#include <cstdio>
#include <cstdlib>

int main() {
    const char* model_path = std::getenv("LIVEVOICE_MODEL");
    const char* voice_path = std::getenv("LIVEVOICE_VOICE");
    if (!model_path || !*model_path || !voice_path || !*voice_path) {
        std::fprintf(stderr, "skip: set LIVEVOICE_MODEL and LIVEVOICE_VOICE\n");
        return 77;
    }
    lv::Model model;
    if (!model.load(model_path)) { std::fprintf(stderr, "FAIL: model load\n"); return 1; }
    const auto& c = model.cfg;
    std::printf("hidden %d, layers %d+%d, heads %d/%d x %d, vocab %d, latent %d, hop %d, scaling %.4f bias %.4f\n",
                c.hidden, c.n_layers_lm, c.n_layers_tlm, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab_size, c.latent,
                model.decoder.hop(), static_cast<double>(c.speech_scaling), static_cast<double>(c.speech_bias));
    if (c.hidden != 896 || c.n_layers_lm != 4 || c.n_layers_tlm != 20 || model.decoder.hop() != 3200) {
        std::fprintf(stderr, "FAIL: not the realtime-0.5B geometry\n");
        return 2;
    }
    if (model.tokenizer.vocab_size() == 0) { std::fprintf(stderr, "FAIL: tokenizer\n"); return 3; }
    std::vector<float> e;
    const auto ids = model.tokenizer.encode("Hello there!");
    if (!model.embed(ids.data(), static_cast<int>(ids.size()), &e) || e.size() != ids.size() * 896) {
        std::fprintf(stderr, "FAIL: embed\n");
        return 4;
    }
    lv::Voice voice;
    if (!voice.load(voice_path, model)) { std::fprintf(stderr, "FAIL: voice load\n"); return 5; }
    std::printf("voice: lm %d, tts-lm %d, negative %s (%d); post-filter %s\n", voice.lm.seq, voice.tlm.seq,
                voice.has_neg ? "yes" : "no", voice.neg_tlm.seq, model.dfn.ready ? "yes" : "no");
    std::printf("OK\n");
    return 0;
}
