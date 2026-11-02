# Maintainer's guide

For whoever works on this next. The README covers what users see.

## What this is

VibeVoice-Realtime-0.5B text-to-speech in C++ on stock ggml (pinned
submodule, no fork). One model gguf carries the weights, the tokenizer and
the DeepFilterNet3 post-filter; a voice gguf carries a pre-computed voice
prompt. Streaming is the primary mode: everything, one-shot synthesis
included, runs through the same realtime loop.

References, in the order to trust them:
1. [`microsoft/VibeVoice`](https://github.com/microsoft/VibeVoice)
   (`vibevoice/modular/`): the code the model was trained with. The truth on
   shapes, ordering and the prompt format.
2. [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio): a
   non-PyTorch port, useful for a second opinion.
3. `transformers` `vibevoice_*`: a refactored re-port that differs in
   places (sampling formula, processor API). Cross-check before trusting.

## Layout

```
include/livevoice.h          C API (the only public header)
src/
  livevoice.cpp              C API over the classes below
  model.{hpp,cpp}            Model (weights, tokenizer, decoder, post-filter), Voice
  generate.{hpp,cpp}         the realtime loop: text windows, emit schedule,
                             silence flush, post-filter, lead trim; synthesize()
  frame_graph.{hpp,cpp}      one ggml graph per speech frame
  stream.{hpp,cpp}           Stream: worker thread, incremental text
  diffusion.{hpp,cpp}        diffusion head, DPM-Solver++ schedule and in-graph solve
  qwen2.{hpp,cpp}            Qwen2 layer, resident K/V cache, attention
  acoustic_decoder.{hpp,cpp} stateful sigma-VAE decoder (mul_mat + overlap-add)
  dfn.{hpp,cpp}, dfn_fast.*, dfn_fft.*   DeepFilterNet3 post-filter
  resample2x.{hpp,cpp}       exact 2x upsampler in front of the filter
  tokenizer.{hpp,cpp}        Qwen2 byte-level BPE with the realtime encode options
  gguf_loader.{hpp,cpp}      gguf reader (backend, host or metadata only)
  backend.{hpp,cpp}          backend choice, graph compute, the compute mutex
  wav.{hpp,cpp}              dr_wav wrapper
  bench.hpp, common.*        LIVEVOICE_BENCH timings, logging
examples/cli                 livevoice-cli
examples/quantize            livevoice-quantize (recipes, --inspect)
scripts/                     convert_model.py, convert_tokenizer.py, convert_voice.py,
                             convert_deepfilternet.py, merge_gguf.py
tests/                       one ctest per test_*.cpp; 77 = skipped
```

## Build and test

```bash
cmake -B build -DLIVEVOICE_BUILD_TESTS=ON && cmake --build build -j
LIVEVOICE_MODEL=$PWD/models/livevoice-0.5b-ATTN_Q8.gguf \
LIVEVOICE_VOICE=$PWD/models/voices/en-Clarissa_female.gguf \
  ctest --test-dir build --output-on-failure
```

| test | checks | model |
|---|---|:---:|
| `test_smoke` | the library links | |
| `test_tokenizer` | BPE parity with Hugging Face on a corpus, and the encode options | `tests/fixtures/tokenizer.gguf` or `LIVEVOICE_MODEL` |
| `test_qwen2` | a random layer against a plain C++ reference, whole and chunked, eager and flash attention | |
| `test_diffusion` | head and in-graph sampler against a plain C++ reference, with guidance | |
| `test_decoder` | the stateful decoder against plain convolutions; chunked = whole | |
| `test_dfn` | FFT, upsampler; with `LIVEVOICE_DFN_*`, the filter against a reference output | |
| `test_model` | geometry of a real model and voice | yes |
| `test_frame` | the frame graph's latent = the sampler's alone | yes |
| `test_generate` | plausible speech, several chunks, same seed = same samples | yes |
| `test_stream` | streamed pieces = one-shot synthesis; abort is prompt | yes |
| `test_capi` | the C API, callback stop, WAV output, both sample rates | yes |

The plain C++ references in the unit tests are direct transcriptions of the
formulas, independent of ggml. Run the suite on CPU too
(`LIVEVOICE_BACKEND=cpu`): kernels differ per backend.

Model-level output is not compared with golden files: a different backend
or quantization changes the samples, and so does anything that changes the
order of ggml ops (fusion on Metal). When a change is meant to be
numerically neutral, compare WAVs or `LIVEVOICE_DUMP_LATENTS` dumps of the
same text, voice and seed before and after; they should be bit-identical
on the same machine.

## gguf names

Keys are `vibevoice.*` (hyperparameters; `vibevoice.variant` must be
`realtime-0.5b`), `tokenizer.*` and `dfn.*`. Tensors:

| upstream | gguf |
|---|---|
| `model.language_model.*` (4 layers) | `lm.blk.<i>.{attn_norm,attn_q,attn_k,attn_v,attn_o,ffn_norm,ffn_gate,ffn_up,ffn_down}.{weight,bias}`, `lm.tok_embd.weight` |
| `model.tts_language_model.*` (20 layers) | `tlm.blk.<i>.*`, `tlm.output_norm.weight` |
| `model.tts_input_types` | `tts.input_types.weight` (row 0 speech, row 1 text) |
| `model.acoustic_connector.*` | `ac.{fc1,norm,fc2}.*` |
| `model.prediction_head.*` | `dh.{noisy_proj,cond_proj,t_embed_lin1,t_embed_lin2,layer_<i>.*,final.*}` |
| `model.acoustic_tokenizer.decoder.*` | `at.dec.{stem,up_<1..6>,stage_<s>_block_<b>.weight.*,head}` |
| `tts_eos_classifier.*` | `eos.{fc1,fc2}.*` |
| `model.speech_{scaling,bias}_factor` | `speech.{scaling,bias}` |

The decoder reads its geometry (stages, channels, strides, depths) from the
tensors. A voice gguf has `voice.{lm,tts_lm,neg_tts_lm}.{k,v}.<i>` as
`[head_dim, n_kv, seq]` and `voice.*.last_hidden` as `[hidden, seq]`.

## Gotchas

These bit before.

1. **Mark the computed tensor as the output, never a view of it.**
   `ggml_set_output` on a reshape protects only the view; the allocator then
   reuses the real buffer and the read-back is garbage (it was the EOS
   head's ReLU output, so every latent looked plausible but half zero, and
   the audio was still speech-like enough to pass a level check).
   `test_frame` guards this.

2. **K/V writes are expanded into the graph first.** A layer's `ggml_cpy`
   into the resident cache must come before the attention that reads the
   cache through a view, and that attention only enters the graph through a
   later expand. `run_stack` and `run_frame` expand all writes first.

3. **Small F16 tensors are promoted to F32 at load.** ggml's element-wise
   ops do not mix F32 and F16; the gguf context is sized exactly to the
   file, so the promoted copies live in a sibling context.

4. **No `conv_transpose_1d`.** On Metal it never returns for the decoder's
   3200x upsample and Metal has no left pad. The decoder is mul_mat, shifted
   views and an in-graph overlap-add with carried state; keep it that way.

5. **CPU matmul rounds activations to F16 for F16 weights, and GPUs pick
   other kernels for other widths.** Chunked vs whole decoding agrees to
   ~1e-4 on Metal, exactly on CPU. Do not chase it.

6. **Metal flash attention needs a supported head size** (64 here). The
   unit test uses the model's head size for that reason.

7. **The silence test needs an absolute floor.** Some voices speak from
   frame 1; a threshold relative to the lead-in then calls everything
   silent. Floor 3.5, lead-relative (1.6x) only when the lead is quieter.

8. **The post-filter makes the output 48 kHz.** Anything that writes or
   plays audio asks `lv_sample_rate` / `lv_stream_sample_rate`.

9. **The realtime model mispronounces rare words for tokenization reasons,
   so its text is not the canonical Qwen2 encoding.**
   (a) A rare word that is one very-high-id token (" cryptography" = 86837,
   " orthogonal" = 94580) is garbled; `merge_cap` (80000) refuses merges
   into ids at or above it, giving " crypt" + "ography". 80000 lifted the
   rare-word hit rate from 73.7 % to 77.1 % with no collateral damage;
   50000 broke common words (hyperbole, Kubernetes, quinoa).
   (b) A text-initial capitalised word splits as "C" + "ryptography" (no
   leading space), which the model reads as spelling; `canonical_first_word`
   encodes it as mid-sentence and swaps the first piece for its spaceless
   twin ("Crypt" + "ography"): sentence-initial rare words 1/10 -> 10/10. A
   literal leading space destabilises other sentence starts instead.
   (c) The model rushes the final word, and a rare single-token word is
   garbled there only ("...about cryptocurrency?" -> "cryptocur" in 9/10
   seeds, fine mid-sentence). `last_word_merge_cap` (20000) re-encodes such
   a last word under that cap (" crypt" + "oc" + "urrency"): 10/10;
   " cryptocurrencies" 0/10 -> 10/10, " Kubernetes" 3/10 -> 7/10. The pieces
   must be frequent (" cryptoc" + "urrency" is 3/10), 12000 breaks common
   words, and a last word the normal merges already split is left alone.
   A stream applies (c) only once `end()` is known and holds back a word
   followed by trailing whitespace until then, so stream and one-shot stay
   sample-identical. Benchmark: 24 sentences, 35 rare-word slots, 10 seeds,
   ASR judge; everyday-word WER does not show any of this.

10. **K-quants need row width % 256 == 0.** At hidden 896 only `ffn_down`
    (4864) takes one; everything else falls back to `--fallback-type`.
    Q4_0 LM weights cost ~6 points of rare-word pronunciation, Q8_0 is as
    good as F16. Sign off a quantization recipe on rare words, not on WER.

11. **Free models before the backend at exit.** ggml-metal asserts if a
    buffer is still resident when its device goes away. The backend frees
    itself in an atexit handler, and the C API frees leaked models in one
    registered after it (so it runs first).

12. **Generation is serialised process-wide** (`compute_mutex()`): graph
    intermediates come from one allocator. Model loading takes the same
    lock, since it uploads to the backend.

## Releasing models

1. `scripts/convert_model.py --tokenizer ... --dfn ...` from the upstream
   safetensors.
2. `livevoice-quantize --recipe attn-q8` (and `mixed`, `q4_k`, `q8_0`);
   check the per-family table, `--inspect` shows it again later.
3. Listen, and run the model tests against the result.
4. Upload to `nodcai/livevoice-0.5b-gguf` with the voices under `voices/`.

## Style

C++17. The C API returns `lv_status` codes and never throws. Comments
explain why, not what; the gotchas above are what that looks like. Keep the
C API thin: it forwards into `lv::`.
