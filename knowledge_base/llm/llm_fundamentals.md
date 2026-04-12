# Large Language Models — Fundamentals

## What is a Large Language Model?

A **Large Language Model (LLM)** is a neural network trained on massive amounts of text data to understand and generate human language. LLMs are built on the **Transformer** architecture and learn statistical patterns across billions of parameters to predict, generate, and reason about text.

Key characteristics:
- **Scale**: Billions to trillions of parameters (GPT-4 ~1.8T, LLaMA-3 70B, Mistral 7B)
- **Pretraining**: Self-supervised learning on large text corpora (books, web, code)
- **Emergent abilities**: Capabilities that appear at scale — reasoning, in-context learning, code generation — not explicitly trained for
- **Generality**: One model handles translation, summarization, QA, coding, and more

---

## The Transformer Architecture

The Transformer (Vaswani et al., 2017 — "Attention is All You Need") is the backbone of all modern LLMs.

### Core Components

#### Tokenization
Text is first split into **tokens** (subword units) using algorithms like:
- **BPE (Byte Pair Encoding)**: Merges frequent character pairs iteratively
- **WordPiece**: Used by BERT
- **SentencePiece**: Language-agnostic; used by LLaMA, T5

Example: "unhappiness" → ["un", "happi", "ness"] → token IDs [1043, 22567, 1097]

The vocabulary size is typically 32K–100K tokens.

#### Token Embeddings
Each token ID maps to a dense vector (embedding dimension d_model, typically 768–12288). These embeddings are learned during training.

#### Positional Encoding
Transformers have no inherent notion of order — positional encodings inject sequence position information.

- **Sinusoidal (original)**: Fixed trigonometric functions: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- **Learned positional embeddings**: Absolute positions learned as parameters (GPT-2)
- **RoPE (Rotary Position Embedding)**: Encodes relative positions via rotation matrices; used by LLaMA, GPT-NeoX — extends context length better than absolute
- **ALiBi**: Adds position bias to attention scores; enables extrapolation beyond training length

#### Self-Attention Mechanism
The heart of the Transformer. Each token attends to every other token.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```
- **Q (Query)**: What the current token is looking for
- **K (Key)**: What each token offers as a match
- **V (Value)**: The actual information to aggregate
- **√d_k scaling**: Prevents softmax saturation for large d_k

**Multi-Head Attention (MHA):**
Run h independent attention heads in parallel, each with d_k = d_model / h.
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```
Each head learns different relationship types (syntax, coreference, semantics).

#### Feed-Forward Network (FFN)
Applied independently to each token after attention:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```
Typically 4× wider than d_model. Uses GELU or SwiGLU activation in modern models.

#### Layer Normalization
Applied **before** (Pre-LN, used in GPT-2+) or **after** (Post-LN, original paper) each sub-layer. Pre-LN is more stable for large-scale training.

#### Residual Connections
Each sub-layer output is x + Sublayer(x), enabling gradient flow through deep networks.

### Decoder-Only vs. Encoder-Decoder

| Architecture | Examples | Use Case |
|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Classification, embedding, NLU |
| **Decoder-only** | GPT, LLaMA, Mistral | Text generation, chat, reasoning |
| **Encoder-Decoder** | T5, BART, Flan-T5 | Translation, summarization |

Modern chat LLMs (GPT-4, LLaMA, Claude, Gemini) are all **decoder-only** — they generate text autoregressively, one token at a time, left to right.

**Causal masking**: Decoder attention masks future tokens so position i can only attend to positions ≤ i.

---

## Attention Variants (Efficiency)

Standard MHA has O(n²) memory w.r.t. sequence length n — expensive for long contexts.

### Grouped Query Attention (GQA)
Multiple query heads share a smaller number of key/value heads. Used in LLaMA-3, Mistral, Gemma.
- MHA: h Q-heads, h K-heads, h V-heads
- GQA: h Q-heads, g K/V-heads where g < h
- MQA (Multi-Query Attention): g = 1 (single K/V head) — maximum efficiency

### FlashAttention
A hardware-aware exact attention algorithm that avoids materializing the full n×n attention matrix in HBM (GPU memory). Uses tiling and recomputation.
- **FlashAttention-2/3**: Further optimized for modern GPU architectures
- Result: Same output as standard attention, 2–4× faster, O(n) memory

### Sliding Window Attention (SWA)
Each token attends only to a window of w nearby tokens instead of all n. Used in Mistral to extend context efficiently.

---

## Pretraining

### Objective: Next Token Prediction
Given tokens x_1, ..., x_{t-1}, predict x_t. The loss is cross-entropy:
```
L = -Σ log P(x_t | x_1, ..., x_{t-1})
```
This is **self-supervised** — no labels needed; the text itself provides supervision.

### Training Data
Modern LLMs train on trillions of tokens from:
- **Web crawl** (Common Crawl, C4, RefinedWeb)
- **Books** (Books3, Project Gutenberg)
- **Code** (GitHub, The Stack)
- **Wikipedia, arXiv, StackOverflow**

Data quality matters more than raw quantity — careful deduplication and filtering significantly improves downstream performance.

### Training at Scale
- **Optimizer**: AdamW with weight decay, gradient clipping
- **Learning rate schedule**: Warmup + cosine decay
- **Mixed precision**: bfloat16 or float16 to reduce memory
- **Gradient checkpointing**: Recompute activations during backward pass to save memory
- **Distributed training**: Tensor parallelism, pipeline parallelism, data parallelism across thousands of GPUs
- **Chinchilla scaling law**: Optimal training tokens ≈ 20× model parameters (Hoffmann et al., 2022)

### Key Pretraining Models Timeline
- **GPT-2** (2019): 1.5B params, first large-scale demonstration
- **GPT-3** (2020): 175B params, in-context learning emergence
- **PaLM** (2022): 540B, chain-of-thought reasoning
- **LLaMA-1/2/3** (2023–2024): Open-weight models, competitive with proprietary
- **Mistral 7B** (2023): Efficient small model, GQA + SWA
- **Gemma / Phi**: Small, high-quality models for efficiency

---

## Context Window and Memory

### Context Window
The maximum number of tokens a model can process at once (input + output).
- GPT-3: 4K tokens
- GPT-4: 128K tokens
- Gemini 1.5: 1M tokens
- LLaMA-3.1: 128K tokens

Larger context enables longer documents, multi-turn conversations, and in-context learning with many examples.

### KV Cache
During autoregressive generation, Key and Value tensors for past tokens are cached to avoid recomputation. This makes generation O(n) per new token instead of O(n²).

**Memory**: KV cache grows linearly with sequence length and batch size — a major bottleneck for serving.

---

## Tokenization and Vocabulary

### Byte Pair Encoding (BPE)
1. Start with character vocabulary
2. Iteratively merge most frequent adjacent pair
3. Repeat until target vocabulary size reached

Rare words decompose into known subwords; common words stay whole.

### Tokenization Quirks
- LLMs count tokens, not words — "ChatGPT" ≈ 3 tokens
- Numbers are often split by digit: "12345" → ["123", "45"]
- Spaces are part of tokens: " hello" ≠ "hello"
- Code tokenization: indentation whitespace can be expensive

### Vocabulary Size vs. Context
Larger vocabularies mean fewer tokens per text (more efficient context use) but larger embedding tables.

---

## Perplexity and Evaluation

### Perplexity (PPL)
Standard intrinsic measure of language model quality:
```
PPL = exp(-1/N · Σ log P(x_t | x_<t))
```
Lower perplexity = model assigns higher probability to held-out text = better language model.

### Benchmark Suites
- **MMLU**: 57 subjects, multiple-choice — knowledge breadth
- **HellaSwag**: Commonsense reasoning
- **HumanEval / MBPP**: Code generation
- **GSM8K / MATH**: Mathematical reasoning
- **TruthfulQA**: Hallucination / truthfulness
- **BIG-Bench**: 200+ diverse tasks

---

## Emergent Abilities

Capabilities that appear at large scale but are absent in smaller models:

- **In-context learning**: Adapting from examples in the prompt without weight updates
- **Chain-of-thought reasoning**: Step-by-step logical reasoning
- **Instruction following**: Following complex multi-step instructions
- **Multi-step arithmetic**: Solving math word problems
- **Code generation**: Producing functional code from natural language descriptions

Emergence is debated — some argue it's an artifact of discrete metrics (Wei et al., 2022).
