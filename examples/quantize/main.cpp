// livevoice-quantize: quantize a model gguf per tensor family.
//
//   livevoice-quantize --src f16.gguf --out compact.gguf --recipe compact
//   livevoice-quantize --inspect model.gguf
//
// K-quants pack 256 weights per super-block, so a tensor takes one only when
// its row width is a multiple of 256. The LM's hidden size is 896, so every
// attention projection, ffn_gate, ffn_up and the token embedding cannot;
// only ffn_down (4864) can. Such rows take --fallback-type (default q8_0).
// The table printed at the end shows what ended up in which type.
//
// Two families change shape as well as type:
//   * the token embedding can drop the rows the tokenizer cannot produce
//     under a merge cap (--embed-cap); the kept ids go to lm.tok_embd.ids
//   * the decoder's transposed-conv kernels are stored as the matrix the
//     decoder multiplies with ([C_in, K*C_out]) once they are quantized,
//     because a [K, C_out, C_in] kernel has no quantizable row
//
// Everything that is not named below (norms, biases, the connector, the EOS
// head, the small stem / head / depthwise kernels, the DFN3 weights) keeps
// its type.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

ggml_type parse_type(const std::string& s) {
    static const std::map<std::string, ggml_type> known = {
        {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1},     {"q5_0", GGML_TYPE_Q5_0}, {"q5_1", GGML_TYPE_Q5_1},
        {"q8_0", GGML_TYPE_Q8_0}, {"iq4_nl", GGML_TYPE_IQ4_NL}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K},
        {"q6_k", GGML_TYPE_Q6_K}, {"f16", GGML_TYPE_F16},       {"bf16", GGML_TYPE_BF16}, {"f32", GGML_TYPE_F32},
    };
    auto it = known.find(s);
    return it == known.end() ? GGML_TYPE_COUNT : it->second;
}

enum Family { kAttn, kGateUp, kDown, kEmbed, kDecoderFfn, kDecoderConv, kHead, kOther, kFamilies };
const char* const kFamilyName[kFamilies] = {"lm attn q/k/v/o", "lm ffn_gate/up", "lm ffn_down",    "lm.tok_embd",
                                            "decoder ffn",     "decoder convs",  "diffusion head", "other"};

Family family_of(const std::string& s) {
    static const std::regex attn(R"(^t?lm\.blk\.\d+\.attn_[qkvo]\.weight$)");
    static const std::regex gate_up(R"(^t?lm\.blk\.\d+\.ffn_(gate|up)\.weight$)");
    static const std::regex down(R"(^t?lm\.blk\.\d+\.ffn_down\.weight$)");
    static const std::regex dec_ffn(R"(^at\.dec\.stage_\d+_block_\d+\.weight\.ffn_linear[12]$)");
    static const std::regex dec_conv(R"(^at\.dec\.up_\d+\.weight$)");
    static const std::regex head(R"(^dh\.(noisy_proj|cond_proj|t_embed_lin[12]|final\.(proj|adaln)|layer_\d+\.(ffn_gate|ffn_up|ffn_down|adaln))$)");
    if (std::regex_match(s, attn))     return kAttn;
    if (std::regex_match(s, gate_up))  return kGateUp;
    if (std::regex_match(s, down))     return kDown;
    if (s == "lm.tok_embd.weight")     return kEmbed;
    if (std::regex_match(s, dec_ffn))  return kDecoderFfn;
    if (std::regex_match(s, dec_conv)) return kDecoderConv;
    if (std::regex_match(s, head))     return kHead;
    return kOther;
}

// Target type per family; GGML_TYPE_COUNT keeps the source type.
struct Plan {
    ggml_type family[kFamilies];
    ggml_type fallback  = GGML_TYPE_Q8_0;
    int       embed_cap = 0;   // drop embedding rows unreachable under this merge cap; 0 keeps all
    Plan() { for (auto& f : family) f = GGML_TYPE_COUNT; }
};

bool apply_recipe(const std::string& name, Plan* p) {
    const ggml_type keep = GGML_TYPE_COUNT, q8 = GGML_TYPE_Q8_0, q4 = GGML_TYPE_Q4_0;
    auto set = [&](ggml_type attn, ggml_type gate_up, ggml_type down, ggml_type embed, ggml_type dec, ggml_type conv,
                   ggml_type head, ggml_type fb, int cap) {
        p->family[kAttn] = attn; p->family[kGateUp] = gate_up; p->family[kDown] = down; p->family[kEmbed] = embed;
        p->family[kDecoderFfn] = dec; p->family[kDecoderConv] = conv; p->family[kHead] = head;
        p->fallback = fb; p->embed_cap = cap;
    };
    // The fallback only reaches the decoder's last, narrow FFN matrices. They
    // are a few hundred kB but run at the full sample rate, where q4_0 is
    // the fastest kernel by a third; their precision does not show.
    if (name == "compact")      set(q8, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, q4, GGML_TYPE_Q5_K, q8, q8, q4, 80000);
    else if (name == "attn-q8") set(q8, q4, GGML_TYPE_Q4_K, q8, GGML_TYPE_Q6_K, keep, q8, q4, 0);
    else if (name == "mixed")   set(q8, q8, GGML_TYPE_Q4_K, q8, GGML_TYPE_Q6_K, keep, q8, q8, 0);
    else if (name == "q4_k")    set(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, q4, GGML_TYPE_Q6_K, keep, q8, q4, 0);
    else if (name == "q8_0")    set(q8, q8, q8, keep, q8, keep, q8, q8, 0);
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

// `n` floats in rows of `row` as `type`. Empty on failure.
std::vector<uint8_t> encode(ggml_type type, const std::vector<float>& f, int64_t row) {
    const int64_t rows = static_cast<int64_t>(f.size()) / row;
    std::vector<uint8_t> out(ggml_row_size(type, row) * rows);
    if (type == GGML_TYPE_F32) {
        std::memcpy(out.data(), f.data(), out.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), reinterpret_cast<ggml_fp16_t*>(out.data()), static_cast<int64_t>(f.size()));
    } else if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), reinterpret_cast<ggml_bf16_t*>(out.data()), static_cast<int64_t>(f.size()));
    } else {
        ggml_quantize_init(type);
        if (ggml_quantize_chunk(type, f.data(), out.data(), 0, rows, row, nullptr) != out.size()) out.clear();
    }
    return out;
}

// Token ids the tokenizer can produce when merges into ids >= cap are
// refused: everything below the cap, the special tokens, and the spaceless
// twins the first-word rule swaps in ("Crypt" for " Crypt").
std::vector<int32_t> reachable_ids(const gguf_context* g, int cap, int64_t n_rows) {
    const int64_t kt = gguf_find_key(g, "tokenizer.tokens"), ks = gguf_find_key(g, "tokenizer.special_tokens_ids");
    if (kt < 0 || ks < 0) return {};
    const int64_t n = std::min<int64_t>(static_cast<int64_t>(gguf_get_arr_n(g, kt)), n_rows);
    std::unordered_map<std::string, int32_t> id_of;
    id_of.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) id_of.emplace(gguf_get_arr_str(g, kt, i), static_cast<int32_t>(i));
    int32_t first_special = static_cast<int32_t>(n);
    const auto* sp = static_cast<const int32_t*>(gguf_get_arr_data(g, ks));
    for (size_t i = 0; i < gguf_get_arr_n(g, ks); ++i) first_special = std::min(first_special, sp[i]);

    std::vector<bool> keep(static_cast<size_t>(n), false);
    const std::string space = "\xC4\xA0";   // the byte-level alphabet's space, U+0120
    for (int64_t i = 0; i < n; ++i) {
        if (i >= cap && i < first_special) continue;
        keep[i] = true;
        if (i >= cap) continue;
        const std::string t = gguf_get_arr_str(g, kt, i);
        if (t.size() > space.size() && t.compare(0, space.size(), space) == 0) {
            auto twin = id_of.find(t.substr(space.size()));
            if (twin != id_of.end()) keep[twin->second] = true;
        }
    }
    std::vector<int32_t> ids;
    for (int64_t i = 0; i < n; ++i) if (keep[i]) ids.push_back(static_cast<int32_t>(i));
    return ids;
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
        if (std::strcmp(t->name, "lm.tok_embd.weight") == 0) c.note = std::to_string(t->ne[1]) + " rows";
    }
    print_summary(s);
    const int64_t pf = gguf_find_key(g, "dfn.sample_rate"), tok = gguf_find_key(g, "tokenizer.model");
    const int64_t cap = gguf_find_key(g, "vibevoice.embed_merge_cap");
    std::printf("%s: %.1f MB of tensors, tokenizer %s, post-filter %s", path.c_str(), total / 1e6,
                tok >= 0 ? "yes" : "no", pf >= 0 ? "yes" : "no");
    if (cap >= 0) std::printf(", embedding pruned for merge cap %u", gguf_get_val_u32(g, cap));
    std::printf("\n");
    gguf_free(g);
    ggml_free(ctx);
    return 0;
}

void usage(const char* argv0) {
    std::printf(
        "usage: %s --src <gguf> --out <gguf> [--recipe <name>] [--<family>-type <type>]...\n"
        "       %s --inspect <gguf>\n"
        "\n"
        "recipes:\n"
        "  compact  q8_0 attention and diffusion head; iq4_nl gate/up; q4_k ffn_down;\n"
        "           q4_0 embedding without the rows a merge cap of 80000 cannot reach;\n"
        "           q5_k decoder FFN; q8_0 decoder convs. Smallest, and closer to the\n"
        "           q8_0 model than attn-q8\n"
        "  attn-q8  q8_0 attention, embedding and diffusion head; q4_0 gate/up;\n"
        "           q4_k ffn_down; q6_k decoder FFN\n"
        "  mixed    attn-q8 with q8_0 gate/up\n"
        "  q4_k     q4_0 attention, gate/up and embedding (896-wide rows); q4_k\n"
        "           ffn_down; q6_k decoder FFN; q8_0 diffusion head\n"
        "  q8_0     q8_0 everywhere but the embedding, which stays F16\n"
        "\n"
        "per family, after the recipe\n"
        "(types: q4_0 q4_1 q5_0 q5_1 q8_0 iq4_nl q4_k q5_k q6_k f16 bf16 f32):\n"
        "  --type <t>              every LM matrix (attention, gate/up, down)\n"
        "  --attn-type <t>         LM attention q/k/v/o; what rare words depend on most\n"
        "  --ffn-type <t>          LM ffn_gate / ffn_up / ffn_down\n"
        "  --gate-up-type <t>      LM ffn_gate / ffn_up\n"
        "  --down-type <t>         LM ffn_down\n"
        "  --embed-type <t>        token embedding; block-32 types only. It is the least\n"
        "                          sensitive matrix of the model\n"
        "  --embed-cap <n>         drop the embedding rows the tokenizer cannot produce\n"
        "                          with a merge cap of n (0 keeps all). The file then\n"
        "                          limits lv_params.merge_cap to n\n"
        "  --decoder-ffn-type <t>  acoustic decoder FFN matrices\n"
        "  --decoder-conv-type <t> acoustic decoder upsampling kernels\n"
        "  --head-type <t>         diffusion head matrices; keep q8_0 or better\n"
        "  --fallback-type <t>     for rows a K-quant cannot take (default q8_0)\n"
        "  --keep-tlm-embed        keep the TTS-LM's token embedding; it is never read,\n"
        "                          so it is dropped by default\n",
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
        else if (a == "--type")              { const ggml_type t = type(); plan.family[kAttn] = plan.family[kGateUp] = plan.family[kDown] = t; }
        else if (a == "--attn-type")         plan.family[kAttn] = type();
        else if (a == "--ffn-type")          { const ggml_type t = type(); plan.family[kGateUp] = plan.family[kDown] = t; }
        else if (a == "--gate-up-type")      plan.family[kGateUp] = type();
        else if (a == "--down-type")         plan.family[kDown] = type();
        else if (a == "--embed-type")        plan.family[kEmbed] = type();
        else if (a == "--embed-cap")         plan.embed_cap = std::atoi(value().c_str());
        else if (a == "--decoder-ffn-type")  plan.family[kDecoderFfn] = type();
        else if (a == "--decoder-conv-type") plan.family[kDecoderConv] = type();
        else if (a == "--head-type")         plan.family[kHead] = type();
        else if (a == "--fallback-type")     plan.fallback = type();
        else if (a == "--keep-tlm-embed")    keep_tlm_embed = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown option %s (see --help)\n", a.c_str()); return 1; }
    }
    if (src.empty() || out.empty()) { usage(argv[0]); return 1; }
    // The runtime looks embedding rows up one at a time, which needs
    // self-contained block-32 rows.
    if (plan.family[kEmbed] != GGML_TYPE_COUNT && plan.family[kEmbed] != GGML_TYPE_F16 &&
        plan.family[kEmbed] != GGML_TYPE_F32 && ggml_blck_size(plan.family[kEmbed]) != 32) {
        std::fprintf(stderr, "--embed-type must be a block-32 type (q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl)\n");
        return 1;
    }

    ggml_context* src_ctx = nullptr;
    struct gguf_init_params ip {};
    ip.no_alloc = false;
    ip.ctx = &src_ctx;
    gguf_context* sg = gguf_init_from_file(src.c_str(), ip);
    if (!sg) { std::fprintf(stderr, "cannot read %s\n", src.c_str()); return 2; }
    const int64_t n = gguf_get_n_tensors(sg);
    if (plan.embed_cap > 0 && ggml_get_tensor(src_ctx, "lm.tok_embd.ids")) {
        std::fprintf(stderr, "%s already has a pruned embedding; start from a full one for --embed-cap\n", src.c_str());
        return 1;
    }

    gguf_context* dg = gguf_init_empty();
    gguf_set_kv(dg, sg);
    struct ggml_init_params dp {};
    dp.mem_size = ggml_tensor_overhead() * (n + 2);
    dp.no_alloc = true;
    ggml_context* dst_ctx = ggml_init(dp);

    std::vector<std::vector<uint8_t>> data;
    data.reserve(static_cast<size_t>(n) + 1);
    Summary summary;
    size_t bytes_in = 0, bytes_out = 0, n_quant = 0;
    auto emit = [&](const char* name, ggml_type type, int n_dims, const int64_t* ne, std::vector<uint8_t> bytes,
                    Family fam, const std::string& note) {
        ggml_tensor* dt = ggml_new_tensor(dst_ctx, type, n_dims, ne);
        ggml_set_name(dt, name);
        data.push_back(std::move(bytes));
        dt->data = data.back().data();
        gguf_add_tensor(dg, dt);
        bytes_out += data.back().size();
        Cell& c = summary[fam][ggml_type_name(type)];
        ++c.count;
        c.bytes += data.back().size();
        if (!note.empty()) c.note = note;
    };
    // The type a family asks for, or what its rows can take instead.
    auto resolve = [&](ggml_type target, int64_t row, std::string* note) {
        if (target == GGML_TYPE_COUNT || row % ggml_blck_size(target) == 0) return target;
        if (row % ggml_blck_size(plan.fallback) == 0) {
            *note = std::string("fallback from ") + ggml_type_name(target);
            return plan.fallback;
        }
        *note = "kept: row width";
        return GGML_TYPE_COUNT;
    };

    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(sg, i);
        ggml_tensor* st = ggml_get_tensor(src_ctx, name);
        bytes_in += ggml_nbytes(st);
        if (!keep_tlm_embed && std::strcmp(name, "tlm.tok_embd.weight") == 0) {
            std::fprintf(stderr, "dropping %s (never read, %.0f MB)\n", name, ggml_nbytes(st) / 1e6);
            continue;
        }
        const Family fam = family_of(name);
        std::string note;
        const ggml_type target = resolve(plan.family[fam], st->ne[0], &note);
        const bool convert = target != GGML_TYPE_COUNT && target != st->type;
        const ggml_type out_type = convert ? target : st->type;
        std::vector<float> f;
        if (convert) {
            f = to_f32(st);
            if (f.empty()) { std::fprintf(stderr, "cannot convert %s from %s\n", name, ggml_type_name(st->type)); return 3; }
            ++n_quant;
        }

        if (fam == kEmbed && plan.embed_cap > 0) {
            const int64_t hidden = st->ne[0];
            const std::vector<int32_t> ids = reachable_ids(sg, plan.embed_cap, st->ne[1]);
            if (ids.empty()) { std::fprintf(stderr, "--embed-cap needs the tokenizer in %s\n", src.c_str()); return 3; }
            const int64_t ne[2] = {hidden, static_cast<int64_t>(ids.size())};
            std::vector<uint8_t> bytes;
            if (convert) {
                std::vector<float> kept(static_cast<size_t>(hidden) * ids.size());
                for (size_t r = 0; r < ids.size(); ++r)
                    std::memcpy(&kept[r * hidden], &f[static_cast<size_t>(ids[r]) * hidden], sizeof(float) * hidden);
                bytes = encode(out_type, kept, hidden);
            } else {
                const size_t row = ggml_row_size(st->type, hidden);
                bytes.resize(row * ids.size());
                for (size_t r = 0; r < ids.size(); ++r)
                    std::memcpy(&bytes[r * row], static_cast<const uint8_t*>(st->data) + static_cast<size_t>(ids[r]) * row, row);
            }
            if (bytes.empty()) { std::fprintf(stderr, "quantizing %s failed\n", name); return 4; }
            emit(name, out_type, 2, ne, std::move(bytes), fam,
                 std::to_string(ids.size()) + " of " + std::to_string(st->ne[1]) + " rows");
            const int64_t ne_ids[1] = {static_cast<int64_t>(ids.size())};
            const auto* raw = reinterpret_cast<const uint8_t*>(ids.data());
            emit("lm.tok_embd.ids", GGML_TYPE_I32, 1, ne_ids, std::vector<uint8_t>(raw, raw + ids.size() * sizeof(int32_t)),
                 kOther, "");
            gguf_set_val_u32(dg, "vibevoice.embed_merge_cap", static_cast<uint32_t>(plan.embed_cap));
            continue;
        }
        if (fam == kDecoderConv && plan.family[fam] != GGML_TYPE_COUNT && ggml_n_dims(st) == 3) {
            // [K, C_out, C_in] -> [C_in, K*C_out], row r = k*C_out + c_out: the
            // layout the decoder multiplies with, and one with quantizable rows.
            const int64_t K = st->ne[0], Co = st->ne[1], Ci = st->ne[2];
            std::string conv_note;
            const ggml_type ct = resolve(plan.family[fam], Ci, &conv_note);
            if (ct != GGML_TYPE_COUNT) {
                const std::vector<float> s = to_f32(st);
                std::vector<float> m(s.size());
                for (int64_t ci = 0; ci < Ci; ++ci)
                    for (int64_t co = 0; co < Co; ++co)
                        for (int64_t k = 0; k < K; ++k) m[(k * Co + co) * Ci + ci] = s[(ci * Co + co) * K + k];
                const int64_t ne[2] = {Ci, K * Co};
                std::vector<uint8_t> bytes = encode(ct, m, Ci);
                if (bytes.empty()) { std::fprintf(stderr, "quantizing %s failed\n", name); return 4; }
                emit(name, ct, 2, ne, std::move(bytes), fam, conv_note);
                ++n_quant;
                continue;
            }
        }

        std::vector<uint8_t> bytes;
        if (convert) {
            bytes = encode(out_type, f, st->ne[0]);
            if (bytes.empty()) { std::fprintf(stderr, "quantizing %s to %s failed\n", name, ggml_type_name(out_type)); return 4; }
        } else {
            const auto* raw = static_cast<const uint8_t*>(st->data);
            bytes.assign(raw, raw + ggml_nbytes(st));
        }
        emit(name, out_type, ggml_n_dims(st), st->ne, std::move(bytes), fam, note);
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
