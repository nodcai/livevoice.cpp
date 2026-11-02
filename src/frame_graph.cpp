#include "frame_graph.hpp"

#include "backend.hpp"
#include "common.hpp"

namespace lv {

namespace {

// Column `c` of a [d, n] tensor as a 1-D view.
struct ggml_tensor* column(struct ggml_context* g, struct ggml_tensor* t, int c) {
    return ggml_view_1d(g, t, t->ne[0], static_cast<size_t>(c) * t->nb[1]);
}

}  // namespace

bool run_frame(const Model& model, const DpmSchedule& sched, ResidentKV& kv_pos, ResidentKV* kv_neg,
               const FrameInputs& in, FrameOutputs* out, BenchTotals* bench) {
    if (!in.z0 || !in.cond_pos || !in.t_sin || !out) return false;
    const auto& cfg   = model.cfg;
    const int  M      = sched.steps();
    const int  latent = cfg.latent, hidden = cfg.hidden, freq = model.head_cfg.freq_size;
    const bool guided = in.cond_neg && kv_neg && in.cfg_scale > 1.0f;
    const int  nc     = guided ? 2 : 1;   // columns: positive (+ negative)

    Qwen2Hparams hp = model.hparams();
    hp.flash_attn   = backend_supports_flash_attn();
    const int hd = hp.head_dim, n_h = hp.n_heads, n_kv = hp.n_kv_heads;
    if (kv_pos.past_len + 1 > kv_pos.max_seq || (guided && kv_neg->past_len + 1 > kv_neg->max_seq)) {
        LV_LOG_ERROR("frame: K/V cache full");
        return false;
    }
    BenchClock clk_all;

    // ---- inputs, in their own buffer so the graph's intermediates can be planned ----
    struct ggml_init_params ipi {};
    ipi.mem_size = ggml_tensor_overhead() * 8;
    ipi.no_alloc = true;
    struct ggml_context* ctx_in = ggml_init(ipi);
    struct ggml_tensor* z0    = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, latent, 1);
    struct ggml_tensor* cond  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, hidden, nc);
    struct ggml_tensor* tsin  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, freq, M);
    struct ggml_tensor* stype = ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, hidden);
    struct ggml_tensor* pos   = ggml_new_tensor_1d(ctx_in, GGML_TYPE_I32, nc);
    ggml_backend_buffer_t in_buf = allocate_ctx_tensors(ctx_in);
    if (!in_buf) {
        ggml_free(ctx_in);
        return false;
    }
    ggml_backend_tensor_set(z0, in.z0, 0, sizeof(float) * latent);
    ggml_backend_tensor_set(cond, in.cond_pos, 0, sizeof(float) * hidden);
    if (guided) ggml_backend_tensor_set(cond, in.cond_neg, sizeof(float) * hidden, sizeof(float) * hidden);
    ggml_backend_tensor_set(tsin, in.t_sin, 0, sizeof(float) * freq * M);
    ggml_backend_tensor_set(stype, model.type_speech.data(), 0, sizeof(float) * hidden);
    const int32_t pv[2] = {kv_pos.past_len, guided ? kv_neg->past_len : 0};
    ggml_backend_tensor_set(pos, pv, 0, sizeof(int32_t) * nc);

    struct ggml_init_params ipg {};
    ipg.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    ipg.no_alloc = true;
    struct ggml_context* g = ggml_init(ipg);

    // 1. Diffusion: the latent for this frame.
    // Outputs are marked on the computed tensor itself, never on a view of
    // it: the allocator only protects the flagged tensor's own memory.
    struct ggml_tensor* lat = dpm_sample(g, z0, cond, tsin, in.cfg_scale, sched, model.head, model.head_cfg);
    ggml_set_output(lat);

    // 2. Acoustic connector + speech type embedding: the next TTS-LM input.
    struct ggml_tensor* e = ggml_mul_mat(g, model.ac_fc1_w, lat);
    if (model.ac_fc1_b) e = ggml_add(g, e, model.ac_fc1_b);
    e = ggml_mul(g, ggml_rms_norm(g, e, 1e-6f), model.ac_norm);
    e = ggml_mul_mat(g, model.ac_fc2_w, e);
    if (model.ac_fc2_b) e = ggml_add(g, e, model.ac_fc2_b);
    e = ggml_add(g, e, stype);

    // 3. TTS-LM step, positive and negative sequence as two columns.
    struct ggml_tensor* cur = guided ? ggml_concat(g, e, e, 1) : e;   // [hidden, nc]
    std::vector<struct ggml_tensor*> writes;
    const float eps = hp.rms_norm_eps;
    for (size_t il = 0; il < model.tlm.size(); ++il) {
        const Qwen2LayerWeights& lw = model.tlm[il];
        struct ggml_tensor* xn = rms_norm(g, cur, lw.attn_norm, eps);
        struct ggml_tensor* q = ggml_mul_mat(g, lw.attn_q, xn);
        if (lw.attn_q_bias) q = ggml_add(g, q, lw.attn_q_bias);
        struct ggml_tensor* k = ggml_mul_mat(g, lw.attn_k, xn);
        if (lw.attn_k_bias) k = ggml_add(g, k, lw.attn_k_bias);
        struct ggml_tensor* v = ggml_mul_mat(g, lw.attn_v, xn);
        if (lw.attn_v_bias) v = ggml_add(g, v, lw.attn_v_bias);
        q = ggml_reshape_4d(g, q, hd, n_h, nc, 1);
        k = ggml_reshape_4d(g, k, hd, n_kv, nc, 1);
        v = ggml_reshape_4d(g, v, hd, n_kv, nc, 1);
        q = qwen2_rope(g, q, pos, hp);
        k = qwen2_rope(g, k, pos, hp);

        struct ggml_tensor* o_cols[2] = {nullptr, nullptr};
        for (int c = 0; c < nc; ++c) {
            ResidentKV& kv = c == 0 ? kv_pos : *kv_neg;
            struct ggml_tensor* kr = kv.k[il];
            struct ggml_tensor* vr = kv.v[il];
            const int np = kv.past_len;
            auto col4 = [&](struct ggml_tensor* t, int n_heads) {
                return ggml_view_4d(g, t, hd, n_heads, 1, 1, t->nb[1], t->nb[2], t->nb[3], static_cast<size_t>(c) * t->nb[2]);
            };
            auto slots = [&](struct ggml_tensor* r, int first, int n) {
                return ggml_view_4d(g, r, hd, n_kv, n, 1, r->nb[1], r->nb[2], r->nb[3], static_cast<size_t>(first) * r->nb[2]);
            };
            writes.push_back(ggml_cpy(g, col4(k, n_kv), slots(kr, np, 1)));
            writes.push_back(ggml_cpy(g, col4(v, n_kv), slots(vr, np, 1)));
            o_cols[c] = qwen2_attention(g, ggml_permute(g, col4(q, n_h), 0, 2, 1, 3),
                                        ggml_permute(g, slots(kr, 0, np + 1), 0, 2, 1, 3),
                                        ggml_permute(g, slots(vr, 0, np + 1), 0, 2, 1, 3), nullptr, hp);
        }
        struct ggml_tensor* o  = guided ? ggml_concat(g, o_cols[0], o_cols[1], 1) : o_cols[0];
        struct ggml_tensor* h  = ggml_add(g, cur, ggml_mul_mat(g, lw.attn_o, o));
        struct ggml_tensor* hn = rms_norm(g, h, lw.ffn_norm, eps);
        struct ggml_tensor* f  = ggml_mul_mat(g, lw.ffn_down, ggml_swiglu_split(g, ggml_mul_mat(g, lw.ffn_gate, hn),
                                                                                ggml_mul_mat(g, lw.ffn_up, hn)));
        cur = ggml_add(g, h, f);
    }
    struct ggml_tensor* hid = rms_norm(g, cur, model.tlm_norm, eps);
    ggml_set_output(hid);

    // 4. EOS classifier on the positive column.
    struct ggml_tensor* eos = ggml_mul_mat(g, model.eos_fc1_w, column(g, hid, 0));
    if (model.eos_fc1_b) eos = ggml_add(g, eos, model.eos_fc1_b);
    eos = ggml_mul_mat(g, model.eos_fc2_w, ggml_relu(g, eos));
    if (model.eos_fc2_b) eos = ggml_add(g, eos, model.eos_fc2_b);
    ggml_set_output(eos);

    struct ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);
    // A layer's K/V writes must be in the graph before its attention reads
    // the cache, and that attention is only pulled in by a later expand, so
    // the writes go first.
    for (auto* wr : writes) ggml_build_forward_expand(gf, wr);
    ggml_build_forward_expand(gf, lat);
    ggml_build_forward_expand(gf, hid);
    ggml_build_forward_expand(gf, eos);

    const double build_ms = clk_all.ms();
    BenchClock clk_compute;
    const bool ok = compute_graph(gf);
    if (ok) {
        out->latent.resize(latent);
        ggml_backend_tensor_get(lat, out->latent.data(), 0, sizeof(float) * latent);
        out->hidden_pos.resize(hidden);
        ggml_backend_tensor_get(hid, out->hidden_pos.data(), 0, sizeof(float) * hidden);
        out->hidden_neg.clear();
        if (guided) {
            out->hidden_neg.resize(hidden);
            ggml_backend_tensor_get(hid, out->hidden_neg.data(), sizeof(float) * hidden, sizeof(float) * hidden);
        }
        ggml_backend_tensor_get(eos, &out->eos_logit, 0, sizeof(float));
        kv_pos.past_len += 1;
        if (guided) kv_neg->past_len += 1;
    } else {
        LV_LOG_ERROR("frame: compute failed");
    }
    if (bench && bench_enabled()) {
        bench->add("frame build", build_ms);
        bench->add("frame compute", clk_compute.ms());
    }
    ggml_free(g);
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx_in);
    return ok;
}

}  // namespace lv
