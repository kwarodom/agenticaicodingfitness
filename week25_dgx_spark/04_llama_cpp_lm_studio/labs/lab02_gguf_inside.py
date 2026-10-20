#!/usr/bin/env python3
"""Lab 04-2 · Inside a GGUF file: what "Q4_K_M" really stores.

Reads the header of a real GGUF file (metadata + the table of tensors, never the weights)
and counts, for every tensor, its quantization type, its number of weights and its bytes.
Then it checks the arithmetic: the sum of all tensor bytes, computed from the block sizes
of each GGML type, must match the size of the file. That proves the bits-per-weight table
in Section 3 of the tutorial.

With no argument it opens a GGUF that Ollama already downloaded on THIS machine (Ollama's
model blobs are GGUF files). On the Spark, point it at the playbook's download:
  --path ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/*/*.gguf

Run: .venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab02_gguf_inside.py [--path FILE.gguf]
"""
import glob
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, bar, check, note, result, step, table, warn  # noqa: E402

# GGML tensor types: id → (name, weights per block, bytes per block). From ggml's block structs,
# e.g. block_q4_K = 2×fp16 scales + 12 bytes of 6-bit sub-scales + 128 bytes of 4-bit values = 144 B / 256 weights.
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
        7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
        20: ("IQ4_NL", 32, 18), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
        28: ("F64", 1, 8), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17)}
# GGUF metadata value types → struct format (8 = string, 9 = array are handled separately)
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
FILE_TYPE = {7: "Q8_0", 15: "Q4_K_M", 14: "Q4_K_S", 17: "Q5_K_M", 18: "Q6_K", 12: "Q3_K_M", 10: "Q2_K",
             1: "F16", 32: "BF16", 2: "Q4_0"}  # general.file_type → the name people use


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(size))[0]

    def string(self) -> str:
        return self.f.read(self.take("Q")).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in SCALAR:
            return self.take(SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, count = self.take("I"), self.take("Q")
            if etype in SCALAR:                              # skip numeric arrays fast, keep the length
                self.f.seek(count * struct.calcsize("<" + SCALAR[etype]), 1)
            else:
                for _ in range(count):
                    self.value(etype)
            return f"[array of {count}]"
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf(path: str) -> dict:
    with open(path, "rb") as f:
        rd = Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file (magic bytes differ)")
        version, n_tensors, n_kv = rd.take("I"), rd.take("Q"), rd.take("Q")
        meta = {}
        for _ in range(n_kv):
            key = rd.string()
            meta[key] = rd.value(rd.take("I"))
        tensors = []
        for _ in range(n_tensors):
            name = rd.string()
            dims = [rd.take("Q") for _ in range(rd.take("I"))]
            ttype, _offset = rd.take("I"), rd.take("Q")
            tensors.append((name, dims, ttype))
        header_end = f.tell()
    return {"version": version, "meta": meta, "tensors": tensors, "header_bytes": header_end,
            "file_bytes": os.path.getsize(path)}


def find_local_gguf() -> tuple[str, str]:
    """A GGUF that Ollama already downloaded here: the 'model' layer of a manifest is the GGUF blob."""
    home = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    best = ("", "")
    for mf in glob.glob(str(home / "manifests" / "**" / "*"), recursive=True):
        if not os.path.isfile(mf):
            continue
        try:
            layers = json.loads(Path(mf).read_text()).get("layers") or []
        except Exception:  # noqa: BLE001
            continue
        for layer in layers:
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                blob = home / "blobs" / layer["digest"].replace(":", "-")
                parts = Path(mf).parts
                label = f"{parts[-2]}:{parts[-1]}"
                # prefer the smallest real file: the header is at the front either way
                if blob.is_file() and (not best[0] or blob.stat().st_size < os.path.getsize(best[0])):
                    best = (str(blob), label)
    return best


banner("Lab 04-2 · inside a GGUF file", "read the header, count the quant types, check the arithmetic", status=False)

args = sys.argv[1:]
if "--path" in args:
    matches = sorted(glob.glob(os.path.expanduser(args[args.index("--path") + 1])))
    path, label = (matches[0], Path(matches[0]).name) if matches else ("", "")
else:
    path, label = find_local_gguf()
if not path:
    warn("no GGUF found. Pass --path FILE.gguf, or pull a small model with Ollama first (ollama pull gemma3:4b).")
    sys.exit(0)

step(1, f"read the header of {label}")
g = read_gguf(path)
meta = g["meta"]
arch = meta.get("general.architecture", "?")
ftype = meta.get("general.file_type")
print(f"│ file          {path}")
print(f"│ GGUF version  {g['version']} · {len(meta)} metadata keys · {len(g['tensors'])} tensors · "
      f"header {g['header_bytes'] / 1e6:.1f} MB (tokenizer included)")
print(f"│ architecture  {arch} · layers {meta.get(f'{arch}.block_count', '?')} · "
      f"context {meta.get(f'{arch}.context_length', '?')}")
print(f"│ file_type     {ftype} → {FILE_TYPE.get(ftype, 'see ggml LLAMA_FTYPE list')}   (what the file calls itself)")

step(2, "what each tensor is actually stored as")
by_type = defaultdict(lambda: [0, 0, 0])            # type → [tensors, weights, bytes]
unknown = set()
examples = {}
for name, dims, ttype in g["tensors"]:
    n = 1
    for d in dims:
        n *= d
    if ttype not in GGML:
        unknown.add(ttype)
        continue
    tname, per_block, block_bytes = GGML[ttype]
    by_type[tname][0] += 1
    by_type[tname][1] += n
    by_type[tname][2] += n // per_block * block_bytes
    examples.setdefault(tname, name)
total_w = sum(v[1] for v in by_type.values())
total_b = sum(v[2] for v in by_type.values())
rows = []
for tname, (nt, nw, nb) in sorted(by_type.items(), key=lambda kv: -kv[1][2]):
    per_block, block_bytes = next((pb, bb) for n_, pb, bb in GGML.values() if n_ == tname)
    rows.append([tname, nt, f"{nw / 1e9:.3f} B", f"{nb / 1e9:.3f} GB", f"{8 * block_bytes / per_block:.3f}",
                 f"{100 * nb / total_b:4.1f}% {bar(nb, total_b, 14)}", examples[tname]])
table(rows, ["type", "tensors", "weights", "bytes", "bits/weight", "share of file", "e.g."])
if unknown:
    warn(f"tensor types this lab does not know: {sorted(unknown)} — their bytes are not counted")

step(3, "does the arithmetic match the file?")
data = g["file_bytes"] - g["header_bytes"]
print(f"│ sum of tensor bytes from block sizes   {total_b:>15,}")
print(f"│ file size − header                     {data:>15,}   (includes ≤ 32-byte alignment padding per tensor)")
check(not unknown and 0 <= data - total_b < 32 * (len(g["tensors"]) + 1),
      "block sizes × tensor shapes = the file, to within alignment padding",
      "the numbers disagree — an unknown tensor type, or a newer GGUF layout")
eff = 8 * total_b / total_w
note(f"effective bits per weight for the whole file: {eff:.2f} ({total_w / 1e9:.2f} B weights, {total_b / 1e9:.2f} GB)")

step(4, "where the bits go: transformer layers vs embeddings vs vision")
groups = defaultdict(lambda: [0, 0])                # group → [weights, bytes]
for name, dims, ttype in g["tensors"]:
    if ttype not in GGML:
        continue
    n = 1
    for d in dims:
        n *= d
    _, per_block, block_bytes = GGML[ttype]
    grp = ("transformer layers (blk.*)" if name.startswith("blk.") else
           "vision tower (v.*, mm.*)" if name.startswith(("v.", "mm.")) else "embeddings, output, norms")
    groups[grp][0] += n
    groups[grp][1] += n // per_block * block_bytes
table([[k, f"{w / 1e9:.3f} B", f"{b / 1e9:.3f} GB", f"{8 * b / w:.2f}"] for k, (w, b) in
       sorted(groups.items(), key=lambda kv: -kv[1][1])], ["group", "weights", "bytes", "bits/weight"])
note("A K-quant recipe keeps most layer weights at its base type (Q4_K = 4.5 bits for Q4_K_M) and gives the most "
     "sensitive ones more (Q6_K = 6.56 bits). Tiny tensors (norms) stay F32.")
if any(k.startswith("vision") for k in groups):
    note("This file also carries a vision encoder, stored at F16: it counts toward the download and memory, "
         "but text generation does not read it.")
k_w = sum(by_type[t][1] for t in by_type if t.endswith("_K"))
if k_w < 0.5 * sum(by_type[t][1] for t in by_type if t != "F32"):
    note("Most weights here are NOT K-quants: K-quants pack 256 weights per super-block, and when a tensor's rows "
         "are not a multiple of 256 the quantizer falls back to a 32-weight type (Q5_0, Q8_0). Small models hit this.")

result(f"'{FILE_TYPE.get(ftype, ftype)}' is a recipe, not one format: a mix of types chosen per tensor. "
       "Judge a quant by the bits/weight of the transformer layers, and size a download by the whole file.")
