---
base_model: google/embeddinggemma-2
language:
- multilingual
library_name: mlx
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- mlx
- mlx-vlm
- embedding
- feature-extraction
- sentence-similarity
- multimodal
- image-feature-extraction
- audio-feature-extraction
- video-feature-extraction
---
# mlx-community/embeddinggemma-2-4bit

MLX conversion of [Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) in **4-bit affine** format, retaining text, image, audio, and video encoders. The model produces normalized 768-dimensional embeddings. See the [original model card](https://huggingface.co/google/embeddinggemma-2) for intended use, training information, evaluation, and limitations.

Quantization mode: `affine`, 4 bits, group size 64. The standard MLX-VLM policy quantizes the text encoder and audio projection. The vision and audio towers and vision projection remain BF16.

Weight storage: 1.098 GB (decimal). Source revision: [`914f7f89142e33e77833254d9c9b90c3cef7303b`](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b). Converted with [MLX-VLM revision `3d87e884`](https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0), from branch `pc/embeddinggemma-2`, and MLX 0.32.3.

## Text embeddings

Install the implementation revision that includes EmbeddingGemma 2 support:

```sh
pip install "git+https://github.com/Blaizzy/mlx-vlm.git@3d87e88402f307efbf68e568971aa887ee7d9ed0"
pip install -U "mlx>=0.32.3" "transformers>=5.18.0"
```

```python
import mlx.core as mx
from transformers import AutoTokenizer
from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.utils import get_model_path

path = get_model_path("mlx-community/embeddinggemma-2-4bit")
model = load_embedding_model(path)
tokenizer = AutoTokenizer.from_pretrained(path)
inputs = tokenizer([
    "task: search result | query: Which planet is known as the Red Planet?",
    "title: none | text: Mars is known as the Red Planet.",
    "title: none | text: Venus is often called Earth's twin.",
], padding=True, return_tensors="np")
embeddings = model(**{key: mx.array(value) for key, value in inputs.items()}).text_embeds
print(embeddings[:1] @ embeddings[1:].T)

# Optional Matryoshka truncation: use 128, 256, 512, or 768 dimensions.
embeddings = embeddings[:, :256]
embeddings /= mx.linalg.norm(embeddings, axis=-1, keepdims=True)
```

Apply the appropriate task prefixes from `config_sentence_transformers.json`. Queries and documents must use the same embedding dimension. Keep non-quantized weights and activations in BF16; do not cast this model to float16.

## Multimodal preprocessing

All modality weights and processor configuration files are included. Image, audio, and video preprocessing requires a Transformers build that exposes `EmbeddingGemma2Processor`. Validation used the model's supplied upstream Transformers 5.18.0.dev0 build; stock PyPI Transformers 5.18.0 does not yet expose this processor. The text-only example above uses `AutoTokenizer` and does not require that processor. With a compatible processor build:

```python
from mlx_vlm import load

model, processor = load("mlx-community/embeddinggemma-2-4bit")
inputs = processor(images=[image], return_tensors="np")  # image is a PIL image
embedding = model(**{key: mx.array(value) for key, value in inputs.items()}).text_embeds
```

## Conversion checks

This low-bit variant shows measurable embedding drift. For applications where embedding fidelity matters, compare against BF16 or 8-bit on your own retrieval data.

All checked outputs were finite, unit-normalized, and had 768 dimensions. The text retrieval smoke test ranked the Mars passage above Venus. Checks cover six multilingual text inputs and synthetic image, audio, two-frame video, and text+image inputs; they are numerical smoke checks, not MTEB or a retrieval-quality benchmark. Cosine and absolute errors compare against the original checkpoint running in PyTorch FP32.

| Input | Minimum cosine vs FP32 | Maximum absolute error |
| --- | ---: | ---: |
| audio | 0.974435 | 0.037149 |
| image | 0.990386 | 0.016959 |
| text | 0.980877 | 0.022356 |
| text_image | 0.986436 | 0.019337 |
| video | 0.986146 | 0.021160 |

The full measurements, including truncated-vector comparisons, are in `validation.json`.

## Reproduce conversion

Using the implementation and compatible processor environment described above:

```sh
python -m mlx_vlm convert --hf-path google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --mlx-path embeddinggemma-2-4bit --dtype bfloat16 -q --q-mode affine --q-bits 4 --q-group-size 64
```

The source model declares Apache-2.0 licensing. This conversion preserves that license and attribution to Google.
