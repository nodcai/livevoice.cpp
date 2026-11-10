# livevoice.cpp

Streaming text-to-speech with Microsoft's
[VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
in C++ on [ggml](https://github.com/ggml-org/ggml). It starts speaking while
the text is still being written, which is what a voice assistant needs: push
an LLM's tokens in as they arrive and play the audio chunks as they come out.

- one gguf per model: weights, tokenizer and a DeepFilterNet3 post-filter
- Metal, CUDA, Vulkan and CPU through ggml backends
- a small C API (`include/livevoice.h`) for embedding from any language
- about 70-120 ms to the first audio and ~11x real time on an M4 Max

## Quickstart

```bash
git clone --recursive https://github.com/nodcai/livevoice.cpp && cd livevoice.cpp
cmake -B build && cmake --build build -j
hf download nodcai/livevoice-0.5b-gguf --local-dir models

./build/bin/livevoice-cli --model models/livevoice-0.5b-ATTN_Q8.gguf \
    --voice models/voices/en-Clarissa_female.gguf \
    --text "Hello! This is a streaming voice." --out hello.wav
```

`--recursive` matters: ggml is a submodule (`git submodule update --init`
if you cloned without it). Add `--stream` to push the text word by word, the
way an LLM delivers it, and see when each chunk arrives:

```
chunk  0 at    72.1 ms:  13665 samples, 0.28 s of audio so far
chunk  1 at   124.3 ms:  25440 samples, 0.81 s of audio so far
chunk  2 at   195.0 ms:  38400 samples, 1.61 s of audio so far
...
5.12 s of audio at 48000 Hz in 475.0 ms
```

Without `--text` the CLI reads stdin, and with `--stream` it pushes stdin
as it is read, so `my-llm | livevoice-cli --stream ...` speaks as the
model writes.

## Models

[`nodcai/livevoice-0.5b-gguf`](https://huggingface.co/nodcai/livevoice-0.5b-gguf)
has four quantizations of the same model, each with the tokenizer and the
post-filter inside, and two voices:

| file | size | contents |
|---|---:|---|
| `livevoice-0.5b-ATTN_Q8.gguf` | 757 MB | q8_0 attention, token embedding and diffusion head; q4_0 gate / up; q4_k ffn_down; q6_k decoder FFN. **Recommended**: q4_k speed with the best rare-word pronunciation |
| `livevoice-0.5b-MIXED.gguf` | 862 MB | attn-q8 with q8_0 gate / up; no measurable gain, slower on phones |
| `livevoice-0.5b-Q4_K.gguf` | 667 MB | q4_0 attention, gate / up and embedding, q4_k ffn_down; smallest, worse on rare words |
| `livevoice-0.5b-Q8_0.gguf` | 1.1 GB | q8_0 throughout (F16 embedding) |
| `voices/en-Clarissa_female.gguf`, `voices/en-Clarion_male.gguf` | ~7 MB | voice prompts |

The realtime model has no audio encoder, so a voice is a pre-computed
prompt (the K/V caches of the model after hearing the voice), not a WAV.
`scripts/convert_voice.py` turns the `.pt` voice prompts that ship with
[microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) into voice
ggufs.

### Why attn-q8

K-quants pack 256 weights per block and the LM's hidden size is 896, so only
`ffn_down` can take q4_k; every other LM matrix and the embedding fall back
to a block-32 type. With q4_0 there, rare words suffer (24 sentences, 35
rare-word slots, 10 seeds, judged by an ASR model):

| LM attention / gate / up | rare-word slots hit |
|---|---:|
| q4_0 (the q4_k file) | 73.7 % |
| q8_0 | 79.7 % |
| f16 | 78.3 % |

The attention projections carry the gain and are only 47 MB at q8_0, while
gate / up are 222 MB at q8_0 against 118 MB at q4_0. Every LM matrix is read
once per speech frame, so on a phone GPU bandwidth decides: on an iPhone 18
Pro (Metal, 5 steps, CFG 1.5) q4_k runs at 8.0x real time, attn-q8 at 7.9x,
q8_0 gate / up at 6.3x. With the tokenizer rules below, attn-q8 hits 82.0 %
of the rare-word slots.

### The compact recipe

`livevoice-quantize --recipe compact` builds a 577 MB file that is closer to
the q8_0 model than attn-q8 is, by spending the bytes where they matter. Fed
the same text and speech latents, the TTS-LM's output drifts from the q8_0
model's by:

| change, everything else q8_0 | drift | size of that part |
|---|---:|---:|
| token embedding q4_0 | 0.7 % | 77 MB, from 145 |
| attention q4_0 | 10.3 % | 25 MB, from 47 |
| gate / up q4_0 | 10.4 % | 118 MB, from 222 |
| gate / up iq4_nl | 8.7 % | 118 MB |
| ffn_down q4_k | 7.8 % | 59 MB, from 111 |

So compact, against attn-q8:

- keeps only the 82,028 embedding rows the tokenizer can produce under its
  merge cap of 80000 (of 151,936), at q4_0: 41 MB instead of 145. The
  pruning alone changes no sample; such a file limits `merge_cap` to 80000
- stores gate / up as iq4_nl: less drift than q4_0 at the same size
- stores the decoder's upsampling kernels at q8_0 instead of F16 (0.4 %
  waveform difference), which the decoder then multiplies with directly
- takes the decoder FFN from q6_k to q5_k (3.5 % waveform difference from
  the q8_0 decoder, against 1.9 %)

| | attn-q8 | compact |
|---|---:|---:|
| file | 757 MB | 577 MB |
| TTS-LM drift from q8_0 | 13.0 % | 11.7 % |
| peak memory, M4 Max | 1.50 GB | 0.92 GB |
| time to 10.4 s of audio, M4 Max | 0.93 s | 0.94 s |

The diffusion head stays at q8_0: it is the most sensitive part of the
model, and at q5_0 its latents move by 10 % on identical inputs.

## C API

```c
#include "livevoice.h"

int play(const float* pcm, int n, void* user) {
    /* hand n samples at lv_sample_rate() to the audio device */
    return 0;                                   /* non-zero stops generation */
}

lv_model* model = lv_model_load("livevoice-0.5b-ATTN_Q8.gguf");
lv_voice* voice = lv_voice_load(model, "en-Clarissa_female.gguf");
lv_params p;
lv_params_init(&p);

/* the whole text at once, in chunks as they are generated */
lv_synthesize(model, voice, "Hello there!", &p, play, NULL);

/* or text pushed while it is being written */
lv_stream* s = lv_stream_begin(model, voice, &p, play, NULL);
lv_stream_push(s, "Hello th");                  /* any split, even mid-word */
lv_stream_push(s, "ere! How can I help?");
lv_stream_end(s);
lv_stream_free(s);                              /* waits for the last chunk */

lv_voice_free(voice);
lv_model_free(model);
```

A stream calls `on_audio` from its worker thread. For the same text and seed
it produces exactly the samples `lv_synthesize` does: text is only handed to
the model up to the last word boundary, since the tokenizer folds a space
into the following word. The output is 48 kHz with the post-filter and
24 kHz without; ask `lv_sample_rate()`.

Build a shared library with `-DLIVEVOICE_SHARED=ON` (for `dlopen`, purego,
ctypes and the like); the API takes and returns only opaque handles, plain
structs and C strings.

### Parameters

`lv_params_init` sets the defaults; the CLI flag is in brackets.

| field | default | |
|---|---:|---|
| `n_steps` [`--steps`] | 5 | DPM-Solver++ steps per speech frame |
| `cfg_scale` [`--cfg`] | 1.5 | classifier-free guidance; <= 1 turns it off and halves the LM work |
| `seed` [`--seed`] | 0 | 0 = random |
| `max_frames` [`--max-frames`] | 1500 | safety stop, 7.5 frames per second of audio |
| `max_text_tokens` | 1024 | text capacity of a stream |
| `first_chunk_frames` [`--first-chunk`] | 3 | first chunk, ~400 ms of audio; sets the time to first audio |
| `max_chunk_frames` | 32 | chunks grow by half up to this |
| `lead_chunk_frames` [`--lead-chunk`] | 0 | hold chunks this small until one carried speech |
| `flush_on_silence` | 1 | emit pending audio when the latents go silent |
| `neg_anchor` | 0.2 | keeps the guidance condition from drifting over long replies |
| `postfilter` [`--no-postfilter`] | 1 | DeepFilterNet3 at 48 kHz |
| `trim_lead` [`--no-trim`] | 1 | drop the lead-in silence before the first sound |
| `merge_cap` [`--merge-cap`] | 80000 | tokenizer rule for rare words, 0 = off |
| `canonical_first_word` [`--no-canonical-first-word`] | 1 | tokenizer rule for a text-initial word |
| `last_word_merge_cap` [`--last-word-merge-cap`] | 20000 | tokenizer rule for the last word, 0 = off |

## How it runs

Text enters five tokens at a time: the lower 4 LM layers, then the upper 20
TTS-LM layers. After each text window up to six speech frames follow, until
the EOS head fires. Each frame is **one ggml graph**: the full DPM-Solver++
solve with the guided pair batched through the diffusion head, the acoustic
connector, the TTS-LM step for the positive and the negative sequence as two
columns of the same matmuls (each with its own K/V cache resident on the
device), and the EOS classifier.

Latents are decoded as soon as a chunk is due by a **stateful acoustic
decoder** built from `mul_mat` and an in-graph overlap-add, which keeps the
left context of every convolution between calls; nothing is decoded twice,
and it runs on Metal, where `conv_transpose_1d` at this 3200x upsample does
not. The first chunk is 3 frames; later ones grow by half so each plays
longer than the next takes to make, and pending frames are flushed when the
latents go silent, because the model sometimes pads half a second before
its EOS.

The realtime model now and then puts a short musical artefact in front of
replies that open with "Welcome!" or "Hello there". A **DeepFilterNet3
post-filter** (a hand-written 10 ms-hop forward on the CPU, at utility QoS
on Apple chips) removes it inside the stream, and the lead-in silence is
trimmed after it, so a host only ever receives speech.

The model mispronounces some rare words for **tokenization** reasons, so the
text encoding deviates from canonical Qwen2 BPE in three measured ways:
merges into very high token ids are refused (" cryptography" becomes
" crypt" + "ography"), a text-initial word is split as it would be
mid-sentence, and a single-token rare word at the very end is re-split into
frequent pieces, because the model rushes the final word. See
[AGENTS.md](AGENTS.md) for the numbers.

Per speech frame on an M4 Max (Metal, attn-q8, 5 steps, CFG 1.5) the graph
takes about 5.5 ms; an 11.5 s paragraph is generated, decoded and
post-filtered in about 1 s, with the first chunk after 120 ms.

## Building

```bash
cmake -B build -DLIVEVOICE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Backends follow ggml's own options: Metal is on by default on Apple,
`-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` elsewhere. At run time
`LIVEVOICE_BACKEND=cpu` (or `metal`, `cuda`, `vulkan`) picks a device.
A parent CMake project that already has a `ggml` target keeps its own.

The tests that need a model read `LIVEVOICE_MODEL` and `LIVEVOICE_VOICE`
and are skipped without them.

Other environment switches: `LIVEVOICE_BENCH=1` prints per-phase timings,
`LIVEVOICE_FLASH_ATTN=0` turns flash attention off, `LIVEVOICE_DUMP_LATENTS=<file>`
writes the raw latents.

## Converting from upstream

```bash
python -m pip install numpy gguf
hf download Qwen/Qwen2.5-0.5B --local-dir models/qwen2.5
hf download microsoft/VibeVoice-Realtime-0.5B --local-dir models/VibeVoice-Realtime-0.5B
python scripts/convert_tokenizer.py --src models/qwen2.5 --out models/tokenizer.gguf
# DeepFilterNet3's ONNX export (pip install deepfilternet onnx), optional
python scripts/convert_deepfilternet.py --input <DeepFilterNet3 dir> --output models/deepfilternet3.gguf

python scripts/convert_model.py --src models/VibeVoice-Realtime-0.5B \
    --tokenizer models/tokenizer.gguf --dfn models/deepfilternet3.gguf \
    --out models/livevoice-0.5b-F16.gguf
./build/bin/livevoice-quantize --src models/livevoice-0.5b-F16.gguf \
    --out models/livevoice-0.5b-ATTN_Q8.gguf --recipe attn-q8
```

This reproduces the published attn-q8 file tensor for tensor. The other
recipes are `compact`, `mixed`, `q4_k` and `q8_0`; per-family flags override them, and
`livevoice-quantize --inspect <gguf>` prints what a file contains. Voices:

```bash
python -m pip install torch
curl -LO https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/en-Carter_man.pt
python scripts/convert_voice.py --src en-Carter_man.pt --out models/voices/en-Carter_man.gguf
```

## License

MIT, see [LICENSE](LICENSE). The model weights are under their own licenses:
[VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
(MIT) and [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
(MIT / Apache-2.0).
