#!/usr/bin/env python3
"""Convert microsoft/VibeVoice-Realtime-0.5B (safetensors) to a model gguf.

    hf download microsoft/VibeVoice-Realtime-0.5B --local-dir models/VibeVoice-Realtime-0.5B
    python scripts/convert_model.py --src models/VibeVoice-Realtime-0.5B \\
        --tokenizer models/tokenizer.gguf --dfn models/deepfilternet3.gguf \\
        --out models/livevoice-0.5b-F16.gguf

then quantize with `livevoice-quantize --recipe attn-q8`.

The file carries the weights under the names below, the model's
hyperparameters as `vibevoice.*` keys, the tokenizer (`tokenizer.*`, from
convert_tokenizer.py) and, with --dfn, the DeepFilterNet3 post-filter
(`dfn.*`, from convert_deepfilternet.py). Matrices are written as F16 (or
F32 with --dtype f32); scalars stay F32. The TTS-LM's token embedding is
never read and is left out.

    model.language_model.*           lm.*
    model.tts_language_model.*       tlm.*
    model.tts_input_types            tts.input_types
    model.speech_{scaling,bias}_factor  speech.{scaling,bias}
    model.acoustic_connector.*       ac.*
    model.prediction_head.*          dh.*
    model.acoustic_tokenizer.decoder.*  at.dec.*
    tts_eos_classifier.*             eos.*

Needs numpy and gguf (pip install numpy gguf); no torch.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

_L = r"model\.language_model"
_T = r"model\.tts_language_model"
_D = r"model\.acoustic_tokenizer\.decoder"
_H = r"model\.prediction_head"

# (pattern, replacement); the first match wins. `__SUFFIX__` becomes "" for
# a weight and "_bias" for a bias.
REWRITES: list[tuple[re.Pattern, str]] = [(re.compile(p), r) for p, r in [
    (rf"^{_L}\.embed_tokens\.weight$",                                   "lm.tok_embd.weight"),
    (rf"^{_L}\.layers\.(\d+)\.self_attn\.([qkv])_proj\.(weight|bias)$",  r"lm.blk.\1.attn_\2.\3"),
    (rf"^{_L}\.layers\.(\d+)\.self_attn\.o_proj\.weight$",              r"lm.blk.\1.attn_o.weight"),
    (rf"^{_L}\.layers\.(\d+)\.input_layernorm\.weight$",                r"lm.blk.\1.attn_norm.weight"),
    (rf"^{_L}\.layers\.(\d+)\.post_attention_layernorm\.weight$",       r"lm.blk.\1.ffn_norm.weight"),
    (rf"^{_L}\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$",       r"lm.blk.\1.ffn_\2.weight"),
    (rf"^{_L}\.norm\.weight$",                                           "lm.output_norm.weight"),
    (rf"^{_T}\.embed_tokens\.weight$",                                   "tlm.tok_embd.weight"),
    (rf"^{_T}\.layers\.(\d+)\.self_attn\.([qkv])_proj\.(weight|bias)$",  r"tlm.blk.\1.attn_\2.\3"),
    (rf"^{_T}\.layers\.(\d+)\.self_attn\.o_proj\.weight$",              r"tlm.blk.\1.attn_o.weight"),
    (rf"^{_T}\.layers\.(\d+)\.input_layernorm\.weight$",                r"tlm.blk.\1.attn_norm.weight"),
    (rf"^{_T}\.layers\.(\d+)\.post_attention_layernorm\.weight$",       r"tlm.blk.\1.ffn_norm.weight"),
    (rf"^{_T}\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$",       r"tlm.blk.\1.ffn_\2.weight"),
    (rf"^{_T}\.norm\.weight$",                                           "tlm.output_norm.weight"),
    (r"^model\.tts_input_types\.weight$",                                "tts.input_types.weight"),
    (r"^model\.speech_scaling_factor$",                                  "speech.scaling"),
    (r"^model\.speech_bias_factor$",                                     "speech.bias"),
    (r"^model\.acoustic_connector\.fc([12])\.(weight|bias)$",            r"ac.fc\1.\2"),
    (r"^model\.acoustic_connector\.norm\.weight$",                       "ac.norm.weight"),
    (rf"^{_H}\.noisy_images_proj\.weight$",                              "dh.noisy_proj"),
    (rf"^{_H}\.cond_proj\.weight$",                                      "dh.cond_proj"),
    (rf"^{_H}\.t_embedder\.mlp\.0\.weight$",                             "dh.t_embed_lin1"),
    (rf"^{_H}\.t_embedder\.mlp\.2\.weight$",                             "dh.t_embed_lin2"),
    (rf"^{_H}\.layers\.(\d+)\.norm\.weight$",                            r"dh.layer_\1.norm"),
    (rf"^{_H}\.layers\.(\d+)\.adaLN_modulation\.1\.weight$",             r"dh.layer_\1.adaln"),
    (rf"^{_H}\.layers\.(\d+)\.ffn\.(gate|up|down)_proj\.weight$",        r"dh.layer_\1.ffn_\2"),
    (rf"^{_H}\.final_layer\.linear\.weight$",                            "dh.final.proj"),
    (rf"^{_H}\.final_layer\.adaLN_modulation\.1\.weight$",               "dh.final.adaln"),
    (rf"^{_D}\.upsample_layers\.0\.0\.conv\.conv\.(weight|bias)$",        r"at.dec.stem.\1"),
    (rf"^{_D}\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.(weight|bias)$", r"at.dec.up_\1.\2"),
    (rf"^{_D}\.head\.conv\.conv\.(weight|bias)$",                        r"at.dec.head.\1"),
    (rf"^{_D}\.stages\.(\d+)\.(\d+)\.norm\.weight$",                     r"at.dec.stage_\1_block_\2.weight.norm"),
    (rf"^{_D}\.stages\.(\d+)\.(\d+)\.ffn_norm\.weight$",                 r"at.dec.stage_\1_block_\2.weight.ffn_norm"),
    (rf"^{_D}\.stages\.(\d+)\.(\d+)\.gamma$",                            r"at.dec.stage_\1_block_\2.weight.gamma"),
    (rf"^{_D}\.stages\.(\d+)\.(\d+)\.ffn_gamma$",                        r"at.dec.stage_\1_block_\2.weight.ffn_gamma"),
    (rf"^{_D}\.stages\.(\d+)\.(\d+)\.mixer\.conv\.conv\.conv\.(weight|bias)$", r"at.dec.stage_\1_block_\2.weight.mixer_\3"),
    (rf"^{_D}\.stages\.(\d+)\.(\d+)\.ffn\.linear([12])\.(weight|bias)$", r"at.dec.stage_\1_block_\2.weight.ffn_linear\3__SUFFIX__\4"),
    (r"^tts_eos_classifier\.fc([12])\.(weight|bias)$",                   r"eos.fc\1.\2"),
]]
DROPPED = {"tlm.tok_embd.weight"}


def remap(name: str) -> str | None:
    for pat, repl in REWRITES:
        if pat.match(name):
            out = pat.sub(repl, name)
            return out.replace("__SUFFIX__weight", "").replace("__SUFFIX__bias", "_bias")
    return None


def read_safetensors(path: Path):
    """Yields (name, float32 array) without torch; handles BF16 / F16 / F32."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            lo, hi = info["data_offsets"]
            f.seek(base + lo)
            raw = f.read(hi - lo)
            dt = info["dtype"]
            if dt == "BF16":
                arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                raise SystemExit(f"error: {name} has unsupported dtype {dt}")
            yield name, arr.reshape(info["shape"])


def copy_keys(w: gguf.GGUFWriter, path: str, prefix: str) -> gguf.GGUFReader:
    r = gguf.GGUFReader(path)
    for name, field in r.fields.items():
        if not name.startswith(prefix):
            continue
        if field.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_array(name, field.contents())
        else:
            w.add_key_value(name, field.contents(), field.types[0])
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="model directory with config.json and model*.safetensors")
    ap.add_argument("--tokenizer", required=True, help="tokenizer gguf from convert_tokenizer.py")
    ap.add_argument("--dfn", help="deepfilternet3 gguf from convert_deepfilternet.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    src = Path(args.src)
    cfg = json.loads((src / "config.json").read_text())
    if not cfg.get("tts_backbone_num_hidden_layers"):
        sys.exit("error: not a VibeVoice-Realtime checkpoint (no tts_backbone_num_hidden_layers)")
    files = sorted(src.glob("model*.safetensors"))
    if not files:
        sys.exit(f"error: no model*.safetensors under {src}")

    tensors, unmapped = [], []
    for f in files:
        for name, arr in read_safetensors(f):
            new = remap(name)
            if new is None:
                unmapped.append(name)
            elif new not in DROPPED:
                if args.dtype == "f16" and arr.size > 1:
                    arr = arr.astype(np.float16)
                tensors.append((new, np.ascontiguousarray(arr)))
    if unmapped:
        sys.exit("error: unmapped tensors:\n  " + "\n  ".join(unmapped[:20]))

    dec = cfg["decoder_config"]
    ac = cfg["acoustic_tokenizer_config"]
    dh = cfg.get("diffusion_head_config") or {}
    n_tlm = cfg["tts_backbone_num_hidden_layers"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), arch="vibevoice")
    w.add_string ("vibevoice.variant",      "realtime-0.5b")
    w.add_uint32 ("vibevoice.hidden",       dec["hidden_size"])
    w.add_uint32 ("vibevoice.n_layers_lm",  dec["num_hidden_layers"] - n_tlm)
    w.add_uint32 ("vibevoice.n_layers_tlm", n_tlm)
    w.add_uint32 ("vibevoice.n_heads",      dec["num_attention_heads"])
    w.add_uint32 ("vibevoice.n_kv_heads",   dec["num_key_value_heads"])
    w.add_uint32 ("vibevoice.head_dim",     dec["hidden_size"] // dec["num_attention_heads"])
    w.add_uint32 ("vibevoice.vocab_size",   dec["vocab_size"])
    w.add_float32("vibevoice.rope_theta",   float(dec["rope_theta"]))
    w.add_float32("vibevoice.rms_norm_eps", float(dec["rms_norm_eps"]))
    w.add_uint32 ("vibevoice.acoustic.vae_dim", ac["vae_dim"])
    w.add_array  ("vibevoice.acoustic.encoder_ratios", list(ac["encoder_ratios"]))
    enc_d = ac["encoder_depths"]
    enc_d = [int(x) for x in enc_d.split("-")] if isinstance(enc_d, str) else list(enc_d)
    dec_d = ac.get("decoder_depths") or list(reversed(enc_d))
    dec_d = [int(x) for x in dec_d.split("-")] if isinstance(dec_d, str) else list(dec_d)
    w.add_array  ("vibevoice.acoustic.encoder_depths", enc_d)
    w.add_array  ("vibevoice.acoustic.decoder_depths", dec_d)
    w.add_uint32 ("vibevoice.diffusion.head_layers", dh.get("head_layers", 4))
    w.add_float32("vibevoice.diffusion.ffn_ratio",   float(dh.get("head_ffn_ratio", 3.0)))
    w.add_uint32 ("vibevoice.diffusion.latent",      dh.get("latent_size", 64))
    w.add_uint32 ("vibevoice.sample_rate",           24000)
    for name, arr in tensors:
        w.add_tensor(name, arr)

    tok = copy_keys(w, args.tokenizer, "tokenizer.")
    if "tokenizer.tokens" not in tok.fields:
        sys.exit("error: --tokenizer has no tokenizer.tokens key")
    n_dfn = 0
    if args.dfn:
        dfn = copy_keys(w, args.dfn, "dfn.")
        arch = dfn.fields.get("general.architecture")
        if arch and arch.contents() != "deepfilternet3":
            sys.exit(f"error: --dfn architecture is {arch.contents()!r}, expected 'deepfilternet3'")
        w.add_string("vibevoice.postfilter", "deepfilternet3")
        for t in dfn.tensors:
            if t.name.startswith("dfn."):
                w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
                n_dfn += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}: {len(tensors)} tensors + {n_dfn} post-filter tensors, "
          f"layers {dec['num_hidden_layers'] - n_tlm}+{n_tlm}, hidden {dec['hidden_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
