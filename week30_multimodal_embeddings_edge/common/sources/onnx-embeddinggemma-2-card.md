---
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- embedding
- feature-extraction
- multimodal-embedding
- multimodal
- vision
- audio
- video
- image-feature-extraction
- audio-feature-extraction
- video-feature-extraction
- sentence-similarity
language:
- multilingual
library_name: transformers.js
base_model:
- google/embeddinggemma-2
---

<div align="center">
  <img src=https://ai.google.dev/gemma/images/embeddinggemma2_banner.png>
</div>

<p align="center">
    <a href="https://huggingface.co/google/embeddinggemma-2" target="_blank">Hugging Face</a> |
    <a href="https://github.com/google-gemma" target="_blank">GitHub</a> |
    <a href="https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2" target="_blank">Launch Blog</a> |
    <a href="https://ai.google.dev/gemma/docs/embeddinggemma" target="_blank">Documentation</a> |
    <br>
    <b>License</b>: <a href="https://ai.google.dev/gemma/docs/gemma_4_license" target="_blank">Apache 2.0</a> | <b>Authors</b>: <a href="https://deepmind.google/models/gemma/" target="_blank">Google DeepMind</a>
</p>

**EmbeddingGemma 2** is an open multimodal embedding model built by Google DeepMind which maps text (incl. code), images, video, and audio inputs—and combinations thereof—into a single, unified 768-dimensional vector space. The model has 740M total parameters, combining a 270M parameter text model with modular vision (170M) and audio (300M) encoders.

Designed to run on consumer hardware such as mobile devices and laptops, EmbeddingGemma 2 delivers low-latency semantic representations for on-device applications, like search, retrieval-augmented generation (RAG), classification, and clustering.

EmbeddingGemma 2 builds upon the architectural and capability advancements of Gemma 4, offering several core features:

* **Native multimodality:** Unifies 4 modalities (text, images, video, and audio) in a single shared 768-dimensional embedding space.
* **Multilinguality and code:** EmbeddingGemma 2 understands 100+ languages, and achieves a ~14% improvement on code tasks relative to its predecessor.
* **Flexible footprint:** Combines a 270M parameter text backbone (130M transformer + 140M embedder) with selectively loadable vision (170M) and audio (300M) encoders, allowing developers to load only the modalities required for their use case.
* **Matryoshka Representation Learning (MRL):** Native support for truncated embeddings across 128d, 256d, 512d, and 768d, enabling up to a **6x reduction** in vector storage costs with minimal impact on quality.
* **Context length:** 8K token context window, capable of processing minutes of audio or video.
* **Task-steered representations:** Uses lightweight text instruction prefixes to optimize embeddings for different tasks (search, classification, clustering, semantic similarity, etc.).

### Model Overview

| Parameters | Total | 740M |
| :---- | :---- | :---- |
|  | **Backbone** | 130M |
|  | **Embedder** | 140M |
|  | **Modality Encoders** | *Vision:* 170M  *Audio:* 300M |
| **Architecture** | **Layers** | 24 |
|  | **Model Dimension** | 512 |
|  | **Hidden Dimension** | 2048 |
|  | **Sliding Window** | 1024 tokens |
|  | **Vocabulary Size** | 262,144 |
|  | **\# Heads** | 4 |
|  | **\# KV-Heads (Local/Global)** | 2/1 |
|  | **Local:Global** | 5:1 |
|  | **Attention** | GQA/MQA |
|  | **Activation** | Gated FFN with GELU |
|  | **Pooling** | Mean Pooling |
|  | **Projection Layer** | 512→768 |
| **Input/Output** | **Supported Modalities** | Text, Images, Video, Audio |
|  | **Context Window** | 8,192 tokens |
|  | **Native Output Dimension** | 768 |
|  | **MRL Truncation Dimensions** | 128, 256, 512 |

## **Benchmark Results**

EmbeddingGemma 2 was evaluated across text, code, vision, visual document, video, and audio embedding benchmarks. All results reported below use the full-precision checkpoint.

### Overall Evaluation Results (768d)

| Modality | Benchmark | Metric | EmbeddingGemma 2 | EmbeddingGemma 1 |
| :---- | :---- | :---- | :---- | :---- |
| *Text* | Massive Text Embedding Benchmark (MTEB, multilingual, v2) | Mean(Task), Multiple | 61.36 | 61.15 |
|  | Massive Text Embedding Benchmark (MTEB, code, v1) | Mean(Task), NDCG@10 | 78.68 | 68.76 |
| *Image* | Massive Image Embedding Benchmark (MIEB, lite) | Mean(TaskType), Multiple | 64.64 | - |
|  | Massive Multimodal Embedding Benchmark (MMEB v2 - Image) | Mean(Task), Hit@1 | 57.28 | - |
|  | Massive Multimodal Embedding Benchmark (MMEB v2 - VisDoc) | Mean(Task), NDCG@5 | 67.84 | - |
| *Video* | Massive Multimodal Embedding Benchmark (MMEB v2 - Video) | Mean(Task), Hit@1 | 50.67 | - |
| *Audio* | Massive Sound Embedding Benchmark (MSEB, Retrieval) | Mean(Task), MRR@10 | 69.54 | - |
|  | Massive Audio Embedding Benchmark (MAEB) *Hugging Face* | Mean(Task), Multiple | 49.39 | - |

### Evaluation Results with Vector Truncation

With MRL, EmbeddingGemma 2 representations can be truncated below the native 768d to 128d, 256d, and 512d representations and re-normalized.
With this, model users can reduce storage requirements, with minimal quality impact down to 256d. 128d is best suited to text-only workloads.

| Output Dimension | Compression Ratio | MTEB *(multilingual, v2) Mean(Task)* | MTEB  *(eng, v2) Mean(Task)* | MTEB *(code, v1) Mean(Task)* | MIEB *(lite) Mean(TaskType)* | MMEB (v2) *Overall* | MSEB  *(Retrieval) Mean(Task)* | MAEB   *Mean(Task)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 768d (Full) | 1:1 | 61.36 | 68.46 | 78.68 | 64.64 | 59.01 | 69.54 | 49.39 |
| 512d | 1:1.5 | 61.17 | 68.41 | 77.24 | 64.32 | 58.38 | 69.18 | 49.21 |
| 256d | 1:3 | 60.41 | 67.78 | 76.18 | 63.13 | 56.24 | 66.76 | 48.91 |
| 128d | 1:6 | 57.89 | 65.68 | 71.41 | 59.06 | 45.65 | 56.71 | 46.92 |

## **Quick Start**

Install [Transformers.js](https://huggingface.co/docs/transformers.js) from [NPM](https://www.npmjs.com/package/@huggingface/transformers):

```bash
npm i @huggingface/transformers
```

### Text search

The `feature-extraction` pipeline returns normalized embeddings, so the dot product of two embeddings is their cosine similarity:

```js
import { pipeline, matmul } from "@huggingface/transformers";

const extractor = await pipeline("feature-extraction", "onnx-community/embeddinggemma-2-ONNX", {
  device: "webgpu", // or "wasm" (browser) / "cpu" (Node.js)
  dtype: "q4", // see "Choosing a dtype" below
});

const query = "task: search result | query: Which planet is known as the Red Planet?";
const documents = [
  "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
  "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
  "title: none | text: Jupiter, the largest planet in our solar system, has a prominent red spot.",
  "title: none | text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
];

const embeddings = await extractor([query, ...documents], { pooling: "mean", normalize: true });
const query_embedding = embeddings.slice([0, 1]);
const document_embeddings = embeddings.slice([1, null]);

const scores = (await matmul(query_embedding, document_embeddings.transpose(1, 0))).tolist()[0];
const ranking = scores.map((score, i) => ({ score, document: documents[i] })).sort((a, b) => b.score - a.score);
console.log(ranking);
// [
//   { score: 0.854, document: "title: none | text: Mars, known for its reddish appearance, ..." },
//   { score: 0.783, document: "title: none | text: Saturn, famous for its rings, ..." },
//   { score: 0.752, document: "title: none | text: Jupiter, the largest planet in our solar system, ..." },
//   { score: 0.684, document: "title: none | text: Venus is often called Earth's twin, ..." },
// ]
```

### Images, audio and video

For other modalities, use the processor and the model directly. Every input maps into the same embedding space, so any embedding can be compared with any other: here, text queries against an image, an audio clip and a video.

```js
import { AutoModel, AutoProcessor, load_image, load_audio, load_video, cat, matmul } from "@huggingface/transformers";

const model_id = "onnx-community/embeddinggemma-2-ONNX";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModel.from_pretrained(model_id, { device: "webgpu", dtype: "q4" });

// The processor takes (text, images, audio, videos)
const embed = async (...inputs) => (await model(await processor(...inputs))).sentence_embedding;

// Text queries, with a task prefix (see "Task Instruction Prefixes" below)
const queries = [
  "task: search result | query: cats sleeping on a couch",
  "task: search result | query: a president's speech about serving your country",
  "task: search result | query: a turtle swimming in the ocean",
];
const query_embeddings = await embed(queries);

// An image, an audio clip (mono, 16 kHz) and a video (1 frame per second)
const url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main";
const image = await load_image(`${url}/cats.jpg`);
const audio = await load_audio(`${url}/jfk.wav`, 16000);
const video = await load_video(`${url}/sea-turtle.mp4`, { fps: 1 });

const media_embeddings = cat([
  await embed(null, image),
  await embed(null, null, audio),
  await embed(null, null, null, video),
]);

// Embeddings are normalized: the dot product is the cosine similarity
const scores = (await matmul(media_embeddings, query_embeddings.transpose(1, 0))).tolist();
["image", "audio", "video"].forEach((name, i) => console.log(name, scores[i].map((x) => x.toFixed(3))));
// Each input scores highest with its own query:
//          cats     speech   turtle
// image ["0.740", "0.462", "0.508"]
// audio ["0.504", "0.763", "0.487"]
// video ["0.505", "0.500", "0.731"]
```

To embed several items of a modality at once, pass one list per input: `processor(null, [[image1], [image2]])` returns two image embeddings, while a flat list of images, `processor(null, [image1, image2])`, is a single input made of both images.

> [!NOTE]
> `load_audio` and `load_video` decode media with browser APIs. In Node.js, decode the media yourself and pass the audio as a mono 16 kHz `Float32Array`, and a video as `new RawVideo(frames, duration)`, where `frames` is an array of `RawImage` and `duration` is in seconds.

## **Best Practices**

For optimal embedding quality and runtime efficiency, follow these configurations and best practices:

### 1. Task Instruction Prefixes

EmbeddingGemma 2 is trained with short task instruction prefixes prepended to text inputs. Using the right prefix improves quality; omitting it still works but reduces precision. Prefixes apply to text only. Pass images, video, and audio without any prefix.

Documents with a real title should be formatted as `title: {title} | text: {content}`. Use `title: none` when no title is available.

**Prefix Notation & Usage**

We offer two types of task prefixes, depending on how embeddings are used in the task. There are two types of tasks:

* **Asymmetric Tasks (e.g. retrieval):** Use a query prefix for queries and a document prefix for corpus items.
* **Symmetric Tasks (e.g. classification, similarity)**: Apply the same task prefix to all inputs being compared.

| Use Case | Task Type | Query Task Instruction | Document Task Instruction *(use `none` if no title)* |
| :---- | :---- | :---- | :---- |
| Web / document search | Asymmetric | `task: search result \| query: {query}` | `title: {title} \| text: {content}` |
| Question answering | Asymmetric | `task: question answering \| query: {question}` | `title: {title} \| text: {passage}` |
| Fact checking | Asymmetric | `task: fact checking \| query: {claim}` | `title: {title} \| text: {evidence}` |
| Code search | Asymmetric | `task: code retrieval \| query: {query}` | `title: {title or filename} \| text: {code}` |
| Text classification | Symmetric | `task: classification \| query: {content}` | N/A |
| Clustering | Symmetric | `task: clustering \| query: {content}` | N/A |
| Measuring similarity | Symmetric | `task: sentence similarity \| query: {content}` | N/A |

For example:

```js
const query = (text) => `task: search result | query: ${text}`;
const document = (text, title = "none") => `title: ${title} | text: ${text}`;

const texts = [
  query("What causes the northern lights?"),
  document("Charged particles from the sun excite gases in the upper atmosphere.", "Aurora"),
];
console.log(texts);
// ["task: search result | query: What causes the northern lights?",
//  "title: Aurora | text: Charged particles from the sun excite gases in the upper atmosphere."]
```

### 2. Selective Encoder Loading

The vision and audio encoders are independent components. To reduce memory consumption and download size when deploying text-only or single-modality pipelines, remove the configs of the unused encoders before loading the model:

```js
import { AutoConfig, AutoModel, AutoTokenizer } from "@huggingface/transformers";

const model_id = "onnx-community/embeddinggemma-2-ONNX";
const config = await AutoConfig.from_pretrained(model_id);
config.vision_config = config.audio_config = null; // text only
const model = await AutoModel.from_pretrained(model_id, { config, device: "webgpu", dtype: "q4" });

const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const { sentence_embedding } = await model(await tokenizer(["task: search result | query: What causes the northern lights?"]));
console.log(sentence_embedding.dims); // [1, 768]
```

| Active Modalities | Configs to remove | Effective Size |
| :---- | :---- | :---- |
| Text only | `vision_config`, `audio_config` | 270M |
| Text and image | `audio_config` | 440M |
| Text and audio | `vision_config` | 570M |
| Full multimodal | none | 740M |

Video frames go through the vision encoder, so video needs `vision_config`. The `feature-extraction` pipeline loads every encoder.

### 3. Matryoshka Dimension Truncation

EmbeddingGemma 2 is trained with Matryoshka Representation Learning, so the 768-dimensional output vector can be shortened by keeping only its leading dimensions. The supported dimensions are 768, 512, 256, and 128. Shorter vectors reduce storage and speed up similarity search at some cost to quality.

At runtime, please adhere to the following guidelines:

* **Re-normalize after truncating:** slicing a unit-length vector does not preserve unit length. The shortened vector must be L2-normalized before it is used for cosine similarity. Skipping this step degrades ranking quality silently—it produces plausible-looking scores rather than an error.
* **Queries and documents must share a dimension.** A 768-dimensional query cannot be scored against a 128-dimensional corpus.

For example, continuing the text search example above:

```js
const dim = 256; // or 512, 128
const truncated = embeddings.slice(null, [0, dim]).normalize(2, -1);

const truncated_scores = (await matmul(truncated.slice([0, 1]), truncated.slice([1, null]).transpose(1, 0))).tolist()[0];
console.log(truncated_scores);
// [0.706, 0.871, 0.775, 0.807]: Mars is still first (Venus, Mars, Jupiter, Saturn)
```

Model quality at each dimension is reported in the truncation table in the **Benchmark Results** section above. Quality is close to lossless down to 256 dimensions. 128 dimensions degrades multimodal quality substantially and should be validated against your own workload before adoption.

### 4. Choosing a dtype

The model is available in several precisions. Pick one with `dtype`, either for every component or per component (`model` is the text model):

```js
import { AutoModel } from "@huggingface/transformers";

const model = await AutoModel.from_pretrained("onnx-community/embeddinggemma-2-ONNX", {
  device: "webgpu",
  dtype: { model: "q4", vision_encoder: "q4", audio_encoder: "q8" },
});
```

| dtype | Text model | Vision encoder | Audio encoder | Total | Cosine similarity to fp32 (worst case) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| `fp32` | 1085 MB | 671 MB | 1173 MB | 2929 MB | 1 |
| `fp16` | 543 MB | 336 MB | 587 MB | 1465 MB | 0.9998 |
| `q8` | 314 MB | 195 MB | 340 MB | 850 MB | 0.9997 |
| `q4` | 175 MB | 109 MB | 189 MB | 473 MB | 0.975 (text: 0.988) |
| `q4f16` | 157 MB | 98 MB | 171 MB | 426 MB | 0.975 (text: 0.988) |

In the browser, we recommend `q4` on WebGPU: it is a sixth of the full-precision download, with good quality. Use `q8` (or `fp16` on WebGPU) when quality matters most, for example for audio, the modality that 4-bit quantization affects the most.

### 5. Multimodal Input

#### Interleaving

A single input may mix text with images, video, and audio, using a single shared 8,192-token context.

The position of each media item within the sequence is marked in the text using placeholder tokens from the model's vocabulary:

* `<|image|>` marks the position of image input
* `<|video|>` marks the position of video input
* `<|audio|>` marks the position of audio input

For example, a social media post with a photo and a video might be encoded as:

```js
import { AutoModel, AutoProcessor, load_image, load_video, matmul } from "@huggingface/transformers";

const model_id = "onnx-community/embeddinggemma-2-ONNX";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModel.from_pretrained(model_id, { device: "webgpu", dtype: "q4" });

const url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main";
const text = ["My two cats while I'm away <|image|> and what I'm up to on my diving trip: <|video|>"];
const images = [[await load_image(`${url}/cats.jpg`)]];
const videos = [await load_video(`${url}/sea-turtle.mp4`, { fps: 1 })];
const { sentence_embedding: post_embedding } = await model(await processor(text, images, null, videos));

const queries = [
  "task: search result | query: vacation in the ocean",
  "task: search result | query: pets at home",
  "task: search result | query: a cooking recipe",
];
const { sentence_embedding: query_embeddings } = await model(await processor(queries));
console.log((await matmul(post_embedding, query_embeddings.transpose(1, 0))).tolist()[0]);
// [0.734, 0.672, 0.603]: the post matches both the vacation and the pets, and not the recipe
```

Each placeholder in text is filled from the corresponding media list, in order. The call returns a single embedding that represents the text, image and video together. This embedding can be compared directly against any other EmbeddingGemma 2 embedding, such as the text-only queries above.

#### Context Limits

All modalities share a single 8,192-token context window. Each modality consumes that budget at a fixed rate:

| Modality | Token Cost | Max Input |
| :---- | :---- | :---- |
| *Text* | 1 token per subword | 8,192 tokens |
| *Image* | 280 tokens per image  *(default)* | ~29 images |
| *Video* | 140 tokens per frame  *(default)* | ~58 frames |
| *Audio* | 25 tokens per second | ~327 seconds |

The maximums above assume a single modality with no accompanying text. Interleaved inputs draw from the same budget, so mixing modalities reduces the amount of each that fits. Note also max input for images and video can be as high as ~114 images or frames when using a lower vision token budget (described below).

> [!WARNING]
> On WebGPU, keep each batch under about 2,700 tokens in total, counting the tokens of images, video frames and audio. Above that, some ONNX Runtime WebGPU kernels exceed a GPU dispatch limit. This includes video longer than about 19 seconds at 1 frame per second: sample fewer frames with `load_video(url, { num_frames: 16 })`, or run on `wasm` or `cpu`.

#### Vision Token Budget

Model users can forgo the default sequence length for input images / video frames to represent images with “soft token” amounts ranging from 70 to 1120. Increasing the vision budget for input images trades latency and token count for quality.

In general, higher input sequence lengths capture more information. Scaling up input sequence length means more expressive input representations; so, increasing input sequence length will improve fine-grained visual understanding, uplifting embedding quality and performance in downstream use cases.

The budget is a setting of the image processor (280 by default; 70, 140, 280, 560 or 1120):

```js
import { AutoProcessor, load_image } from "@huggingface/transformers";

const processor = await AutoProcessor.from_pretrained("onnx-community/embeddinggemma-2-ONNX");
const image = await load_image("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg");

for (const budget of [70, 280, 1120]) {
  processor.image_processor.max_soft_tokens = budget;
  const { num_soft_tokens_per_image } = await processor(null, image);
  console.log(budget, num_soft_tokens_per_image);
}
// 70 [63]
// 280 [266]
// 1120 [1064]
// The tokens used: at most the budget, depending on the image's aspect ratio
```

#### Input Sampling Defaults

Audio and video inputs have the following default sampling rates:

* By default, video is processed as sampled frames through the vision encoder at **1 frame per second**, as with `load_video(url, { fps: 1 })`. Videos with more than 32 frames are subsampled uniformly to 32 frames.
* Audio should be supplied as **mono at 16 kHz**, as with `load_audio(url, 16000)`.

## **Model Data**

Our pre-training dataset is a large-scale, diverse collection of data encompassing a wide range of domains and modalities, which includes web documents, code, images, video, and audio, with a cutoff date of January 2025. These include:

* **Web Documents:** A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. The training dataset includes content in over 140 languages.
* **Code:** Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to understand code-related semantics.
* **Images:** A wide range of images enables the model to perform image analysis and visual data extraction tasks.
* **Video:** Video sequences and clips covering diverse visual scenes, human actions, and temporal dynamics.
* **Audio:** Speech recordings across multiple languages, environmental sounds, and acoustic events.
* **Cross-modality Samples:** Paired text, image, video, and audio data to align representations across different modalities.

### Data Processing

Several data cleaning and filtering methods were applied to the training data:

* **CSAM Filtering**: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.
* **Sensitive Data Filtering**: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.
* **Additional Methods**: Filtering based on content quality and safety in line with [our policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

### **Ethics & Safety**

EmbeddingGemma 2 is a pre-trained embedding model. Unlike generative models, it does not undergo post-training alignment, safety tuning, or output-level moderation. Safety mitigations during development were focused on pre-training data filtering to reduce exposure to harmful content and severe biases in the learned embedding space, in alignment with [Google's AI Principles](https://ai.google/principles/).

Since embedding models produce representations rather than user-facing text, safety risks manifest downstream in how those representations are used.

Developers and deployers are responsible for evaluating and implementing application-level safeguards, like retrieval filtering and fairness testing, appropriate to their specific production use case.

Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## **Usage and Limitations**

EmbeddingGemma 2 has certain limitations that users should be aware of.

### Intended Usage

EmbeddingGemma 2 generates embeddings from input content, which can be used for a number of downstream applications.

The following list of potential uses is not comprehensive. The purpose of this list is to provide contextual information about the possible use-cases that the model creators considered as part of model training and development.

* **Retrieval:** Embeddings for semantic search across text, code, images, video, and audio, such as document search, RAG over enterprise knowledge bases, code search from natural language queries, or spoken-query search over audio archives.
* **Classification:** Embeddings to classify inputs according to preset labels, like sentiment analysis, content moderation, audio event detection, or image categorization.
* **Clustering:** Embeddings to group inputs based on semantic similarity, like organizing document collections, clustering customer feedback by theme, or discovering near-duplicate content across modalities.
* **Semantic Similarity:** Embeddings to measure pairwise similarity between inputs, such as recommendation systems, duplicate detection, paraphrase identification, or cross-lingual content alignment.
* **Fact Verification:** Embeddings for retrieving evidence documents given a claim, like automated fact-checking systems or source attribution pipelines.

### Limitations

* **Training Data:** The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.
  * The scope of the training dataset determines the subject areas the model can handle effectively.
  * For example, while EmbeddingGemma 2 supports 100+ languages, the model may not exhibit equal performance across languages.
* **Context and Task Complexity:** Models perform well on tasks that can be framed with clear prompts and instructions.
  * Open-ended or highly complex tasks might be challenging.
  * A model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).
* **Language Ambiguity and Nuance:** Natural language is inherently complex. Models might struggle to grasp subtle nuances, sarcasm, or figurative language.
* **Task Instruction Usage:** For text tasks, omitting the recommended task prefix may lead to sub-optimal embedding quality.

### Ethical Considerations and Risks

In creating an open embedding model, we have carefully considered the following:

* **Bias and Fairness**
  * Models trained on large-scale, real-world text and image data can reflect socio-cultural biases embedded in the training material.
  * Training data used for EmbeddingGemma 2 underwent safety filtering to mitigate the risk of these biases.
* **Misinformation and Misuse**
  * Embedding representations can be misused to retrieve, classify, or otherwise organize embedded content in false, misleading or harmful ways.
  * Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).
* **Transparency and Accountability**
  * This model card summarizes details on the model's architecture, capabilities, limitations, and evaluation processes.
  * A responsibly developed open model offers the opportunity to share innovation by making Vision-Language Model (VLM) technology accessible to developers and researchers across the AI ecosystem.

### Risks Identified and Mitigations

* **Misuse for malicious purposes:** Technical limitations and developer and end-user education can help mitigate against malicious applications of embedding models. Educational resources and reporting mechanisms for users to flag misuse are provided.
* **Privacy violations:** Models were trained on data filtered for removal of certain personal information and other sensitive data. Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.
* **Perpetuation of biases:** It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.

### Benefits

EmbeddingGemma 2 is among the strongest multimodal embedding models under 1B parameters. We’re excited to see how developers will use and adapt this model for on-device or edge AI applications.

The model is designed from the ground up for responsible AI development, like other Gemma-family models.
