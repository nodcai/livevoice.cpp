// livevoice-quantize: quantize a model gguf per tensor family.
//
//   livevoice-quantize --src f16.gguf --out attn-q8.gguf --recipe attn-q8
//   livevoice-quantize --inspect model.gguf
//
// K-quants pack 256 weights per super-block, so a tensor takes one only when
// its row width is a multiple of 256. The LM's hidden size is 896, so every
// attention projection, ffn_gate, ffn_up and the token embedding cannot;
// only ffn_down (4864) can. Such rows take --fallback-type (default q8_0).
// The table printed at the end shows what ended up in which type.
//
// Everything that is not named below (norms, biases, the connector, the EOS
// head, conv kernels, the DFN3 weights) keeps its type: the decoder re-lays
// its conv kernels at load time and the post-filter runs on F32.

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace {

ggml_type parse_type(const std::string& s) {
    static const std::map<std::string, ggml_type> known = {
        {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1}, {"q5_0", GGML_TYPE_Q5_0}, {"q5_1", GGML_TYPE_Q5_1},
        {"q8_0", GGML_TYPE_Q8_0}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K},
        {"f16", GGML_TYPE_F16},   {"bf16", GGML_TYPE_BF16}, {"f32", GGML_TYPE_F32},
    };
    auto it = known.find(s);
    return it == known.end() ? GGML_TYPE_COUNT : it->second;
}

enum Family { kAttn, kGateUp, kDown, kEmbed, kDecoderFfn, kHead, kOther, kFamilies };
const char* const kFamilyName[kFamilies] = {"lm attn q/k/v/o", "lm ffn_gate/up", "lm ffn_down", "lm.tok_embd",
                                            "decoder ffn",     "diffusion head", "other"};

Family family_of(const std::string& s) {
    static const std::regex attn(R"(^t?lm\.blk\.\d+\.attn_[qkvo]\.weight$)");
    static const std::regex gate_up(R"(^t?lm\.blk\.\d+\.ffn_(gate|up)\.weight$)");
    static const std::regex down(R"(^t?lm\.blk\.\d+\.ffn_down\.weight$)");
    static const std::regex dec_ffn(R"(^at\.dec\.stage_\d+_block_\d+\.weight\.ffn_linear[12]$)");
    static const std::regex head(R"(^dh\.(noisy_proj|cond_proj|t_embed_lin[12]|final\.(proj|adaln)|layer_\d+\.(ffn_gate|ffn_up|ffn_down|adaln))$)");
    if (std::regex_match(s, attn))    return kAttn;
    if (std::regex_match(s, gate_up)) return kGateUp;
    if (std::regex_match(s, down))    return kDown;
    if (s == "lm.tok_embd.weight")    return kEmbed;
    if (std::regex_match(s, dec_ffn)) return kDecoderFfn;
    if (std::regex_match(s, head))    return kHead;
    return kOther;
}

// Target type per family; GGML_TYPE_COUNT keeps the source type.
struct Plan {
    ggml_type family[kFamilies] = {GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                                   GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT};
    ggml_type fallback = GGML_TYPE_Q8_0;
};

// The published files.
bool apply_recipe(const std::string& name, Plan* p) {
    auto set = [&](ggml_type attn, ggml_type gate_up, ggml_type down, ggml_type embed, ggml_type dec, ggml_type head,
                   ggml_type fb) {
        p->family[kAttn] = attn; p->family[kGateUp] = gate_up; p->family[kDown] = down; p->family[kEmbed] = embed;
        p->family[kDecoderFfn] = dec; p->family[kHead] = head; p->fallback = fb;
    };
    const ggml_type K = GGML_TYPE_COUNT;
    if (name == "attn-q8") set(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
    else if (name == "mixed") set(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
    else if (name == "q4_k") set(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
    else if (name == "q8_0") set(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
    else return false;
    return true;
}

std::vector<float> to_f32(const ggml_tensor* t) {
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (const auto* tt = ggml_get_type_traits(t->type); tt && tt->to_float) {
        tt->to_float(t->data, out.data(), static_cast<int64_t>(n));
    } else {
        out.clear();
    }
    return out;
}

struct Cell { size_t count = 0, bytes = 0; std::string note; };
using Summary = std::map<int, std::map<std::string, Cell>>;

void print_summary(const Summary& s) {
    std::printf("%-18s %-6s %7s %8s  note\n", "tensor family", "type", "tensors", "MB");
    for (const auto& [fam, types] : s)
        for (const auto& [ty, c] : types)
            std::printf("%-18s %-6s %7zu %8.1f  %s\n", kFamilyName[fam], ty.c_str(), c.count,
                        static_cast<double>(c.bytes) / 1e6, c.note.c_str());
}

int inspect(const std::string& path) {
    struct gguf_init_params p {};
    p.no_alloc = true;
    ggml_context* ctx = nullptr;
    p.ctx = &ctx;
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); return 2; }
    Summary s;
    size_t total = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const ggml_tensor* t = ggml_get_tensor(ctx, gguf_get_tensor_name(g, i));
        Cell& c = s[family_of(t->name)][ggml_type_name(t->type)];
        ++c.count;
        c.bytes += ggml_nbytes(t);
        total += ggml_nbytes(t);
    }
    print_summary(s);
    const int64_t pf = gguf_find_key(g, "dfn.sample_rate"), tok = gguf_find_key(g, "tokenizer.model");
    std::printf("%s: %.1f MB of tensors, tokenizer %s, post-filter %s\n", path.c_str(), total / 1e6,
                tok >= 0 ? "yes" : "no", pf >= 0 ? "yes" : "no");
    gguf_free(g);
    ggml_free(ctx);
    return 0;
}

void usage(const char* argv0) {
    std::printf(
        "usage: %s --src <gguf> --out <gguf> [--recipe <name>] [--<family>-type <type>]...\n"
        "       %s --inspect <gguf>\n"
        "\n"
        "recipes (the published files):\n"
        "  attn-q8  q8_0 attention, embedding and diffusion head; q4_0 gate/up;\n"
        "           q4_k ffn_down; q6_k decoder FFN. The recommended one\n"
        "  mixed    attn-q8 with q8_0 gate/up\n"
        "  q4_k     q4_0 attention, gate/up and embedding (896-wide rows); q4_k\n"
        "           ffn_down; q6_k decoder FFN; q8_0 diffusion head\n"
        "  q8_0     q8_0 everywhere but the embedding, which stays F16\n"
        "\n"
        "per family, after the recipe (types: q4_0 q4_1 q5_0 q5_1 q8_0 q4_k q5_k q6_k f16 bf16 f32):\n"
        "  --type <t>             every LM matrix (attention, gate/up, down)\n"
        "  --attn-type <t>        LM attention q/k/v/o\n"
        "  --ffn-type <t>         LM ffn_gate / ffn_up / ffn_down\n"
        "  --gate-up-type <t>     LM ffn_gate / ffn_up\n"
        "  --down-type <t>        LM ffn_down\n"
        "  --embed-type <t>       token embedding; block-32 types only, q8_0 or better\n"
        "                         keeps rare-token rows intact\n"
        "  --decoder-ffn-type <t> acoustic decoder FFN matrices\n"
        "  --head-type <t>        diffusion head matrices\n"
        "  --fallback-type <t>    for rows a K-quant cannot take (default q8_0; q4_0\n"
        "                         costs ~6 points of rare-word pronunciation)\n"
        "  --keep-tlm-embed       keep the TTS-LM's token embedding; it is never read,\n"
        "                         so it is dropped by default\n",
        argv0, argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string src, out;
    Plan plan;
    bool keep_tlm_embed = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        auto type = [&]() {
            const std::string v = value();
            const ggml_type t = parse_type(v);
            if (t == GGML_TYPE_COUNT) { std::fprintf(stderr, "unknown type %s\n", v.c_str()); std::exit(1); }
            return t;
        };
        if (a == "--src") src = value();
        else if (a == "--out") out = value();
        else if (a == "--inspect") return inspect(value());
        else if (a == "--recipe") {
            const std::string r = value();
            if (!apply_recipe(r, &plan)) { std::fprintf(stderr, "unknown recipe %s\n", r.c_str()); return 1; }
        }
        else if (a == "--type")             { const ggml_type t = type(); plan.family[kAttn] = plan.family[kGateUp] = plan.family[kDown] = t; }
        else if (a == "--attn-type")        plan.family[kAttn] = type();
        else if (a == "--ffn-type")         { const ggml_type t = type(); plan.family[kGateUp] = plan.family[kDown] = t; }
        else if (a == "--gate-up-type")     plan.family[kGateUp] = type();
        else if (a == "--down-type")        plan.family[kDown] = type();
        else if (a == "--embed-type")       plan.family[kEmbed] = type();
        else if (a == "--decoder-ffn-type") plan.family[kDecoderFfn] = type();
        else if (a == "--head-type")        plan.family[kHead] = type();
        else if (a == "--fallback-type")    plan.fallback = type();
        else if (a == "--keep-tlm-embed")   keep_tlm_embed = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown option %s (see --help)\n", a.c_str()); return 1; }
    }
    if (src.empty() || out.empty()) { usage(argv[0]); return 1; }
    // The runtime looks embedding rows up one at a time, which needs
    // self-contained block-32 rows.
    if (plan.family[kEmbed] != GGML_TYPE_COUNT && plan.family[kEmbed] != GGML_TYPE_F16 &&
        plan.family[kEmbed] != GGML_TYPE_F32 && ggml_blck_size(plan.family[kEmbed]) != 32) {
        std::fprintf(stderr, "--embed-type must be a block-32 type (q8_0, q5_0, q5_1, q4_0, q4_1)\n");
        return 1;
    }

    ggml_context* src_ctx = nullptr;
    struct gguf_init_params ip {};
    ip.no_alloc = false;
    ip.ctx = &src_ctx;
    gguf_context* sg = gguf_init_from_file(src.c_str(), ip);
    if (!sg) { std::fprintf(stderr, "cannot read %s\n", src.c_str()); return 2; }
    const int64_t n = gguf_get_n_tensors(sg);

    gguf_context* dg = gguf_init_empty();
    gguf_set_kv(dg, sg);
    struct ggml_init_params dp {};
    dp.mem_size = ggml_tensor_overhead() * (n + 1);
    dp.no_alloc = true;
    ggml_context* dst_ctx = ggml_init(dp);

    std::vector<std::vector<uint8_t>> data;
    data.reserve(static_cast<size_t>(n));
    Summary summary;
    size_t bytes_in = 0, bytes_out = 0, n_quant = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(sg, i);
        ggml_tensor* st = ggml_get_tensor(src_ctx, name);
        bytes_in += ggml_nbytes(st);
        if (!keep_tlm_embed && std::strcmp(name, "tlm.tok_embd.weight") == 0) {
            std::fprintf(stderr, "dropping %s (never read, %.0f MB)\n", name, ggml_nbytes(st) / 1e6);
            continue;
        }
        const Family fam = family_of(name);
        ggml_type target = plan.family[fam];
        std::string note;
        if (target != GGML_TYPE_COUNT && st->ne[0] % ggml_blck_size(target) != 0) {
            if (st->ne[0] % ggml_blck_size(plan.fallback) == 0) {
                note = std::string("fallback from ") + ggml_type_name(target);
                target = plan.fallback;
            } else {
                note = "kept: row width";
                target = GGML_TYPE_COUNT;
            }
        }
        const bool convert = target != GGML_TYPE_COUNT && target != st->type;
        const ggml_type out_type = convert ? target : st->type;
        ggml_tensor* dt = ggml_new_tensor(dst_ctx, out_type, ggml_n_dims(st), st->ne);
        ggml_set_name(dt, name);
        const int64_t row = st->ne[0], rows = ggml_nelements(st) / row;
        data.emplace_back(convert ? ggml_row_size(out_type, row) * rows : ggml_nbytes(st));
        dt->data = data.back().data();
        if (convert) {
            const std::vector<float> f = to_f32(st);
            if (f.empty()) { std::fprintf(stderr, "cannot convert %s from %s\n", name, ggml_type_name(st->type)); return 3; }
            if (out_type == GGML_TYPE_F32) {
                std::memcpy(dt->data, f.data(), f.size() * sizeof(float));
            } else if (out_type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(f.data(), static_cast<ggml_fp16_t*>(dt->data), static_cast<int64_t>(f.size()));
            } else if (out_type == GGML_TYPE_BF16) {
                ggml_fp32_to_bf16_row(f.data(), static_cast<ggml_bf16_t*>(dt->data), static_cast<int64_t>(f.size()));
            } else {
                ggml_quantize_init(out_type);
                if (ggml_quantize_chunk(out_type, f.data(), dt->data, 0, rows, row, nullptr) != data.back().size()) {
                    std::fprintf(stderr, "quantizing %s to %s failed\n", name, ggml_type_name(out_type));
                    return 4;
                }
            }
            ++n_quant;
        } else {
            std::memcpy(dt->data, st->data, ggml_nbytes(st));
        }
        bytes_out += data.back().size();
        Cell& c = summary[fam][ggml_type_name(out_type)];
        ++c.count;
        c.bytes += data.back().size();
        if (!note.empty()) c.note = note;
        gguf_add_tensor(dg, dt);
    }
    if (!gguf_write_to_file(dg, out.c_str(), false)) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 5; }
    print_summary(summary);
    std::printf("wrote %s: %zu tensors converted, %.0f MB -> %.0f MB\n", out.c_str(), n_quant, bytes_in / 1e6,
                bytes_out / 1e6);
    ggml_quantize_free();
    gguf_free(dg);
    gguf_free(sg);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    return 0;
}
