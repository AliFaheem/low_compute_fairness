# Compute, time and cost reference

Measured numbers for planning runs. Source: `meta.json` of every run in the HF results repo (`alifaheem94/pakbbq-results`) and the Vast logs, 25 Oct 2026. Update this file when new hardware, models or variants are measured.

## Setup measured

- **GPU:** 1× RTX 3090 Ti 24 GB (Ampere, native BF16), PCIe 4.0, Vast.ai instance 57582615.
- **Software:** torch 2.14.1+cu126, transformers 5.20.2, bitsandbytes 0.50.2.
- **Scoring:** one forward pass per prompt, reading the logits of the 3 label tokens. No generation.
- **Batching:** batches sized by a token budget of 16,384 tokens. Batches shrink automatically on out-of-memory; this never happened in these runs.

## Workload size

- One run = model × variant × language = **6,432 items × 3 option orders = 19,296 rows** (forward passes).
- Exp 1 grid: 3 models × 5 variants × 3 languages = 45 runs, or 30 with Llama skipped for `ur`/`roman_ur`.
- **Cost scales with tokens, not items.** Prompt length depends on the language and the tokenizer.

| Tokens per prompt (Qwen3 tokenizer, mean of 200 items) | en | roman_ur | ur |
|---|---:|---:|---:|
| Mean | 131 | 226 (1.7×) | 315 (2.4×) |
| Max over the 6,432 items | 205 | 383 | 551 |

Gemma 4's tokenizer is much more efficient on Urdu script (max 261 tokens for `ur`), so for Gemma `ur` is *faster* than `roman_ur` (max 334).

## Measured throughput, full runs (19,296 rows each)

| Model | Variant | Weights on GPU | Peak GPU memory | Load time | en rows/s (min) | roman_ur rows/s (min) | ur rows/s (min) | 3 languages |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | bf16 | 16.4 GB | 20.6 GB | 16 s | 39.9 (8.1) | 22.6 (14.2) | 16.1 (20.0) | **42 min** |
| Qwen3-8B | nf4 | 6.0 GB | 10.7 GB | 13 s | 39.6 (8.1) | 22.3 (14.4) | 16.0 (20.1) | **43 min** |
| Qwen3-8B | int8 | 9.4 GB | 14.6 GB | 37 s | 52.9 (6.1) | 29.4 (11.0) | 20.7 (15.5) | **33 min** |
| Qwen3-8B | wanda50 | 16.4 GB | 20.7 GB | 13 s | 39.8 (8.1) | 22.5 (14.3) | 16.1 (20.0) | **42 min** |
| Qwen3-8B | wanda25 | 16.4 GB | 20.7 GB | 12 s | 39.8 (8.1) | 22.5 (14.3) | 16.1 (20.0) | **42 min** |
| Gemma-4-E4B | bf16 | 15.9 GB | 19.2 GB | 16 s | 54.1 (6.0) | 36.3 (8.9) | 44.0 (7.3) | **22 min** |
| Gemma-4-E4B | nf4 | 9.2 GB | 12.8 GB | 16 s | 53.9 (6.0) | 36.1 (8.9) | 44.1 (7.3) | **22 min** |
| Gemma-4-E4B | int8 | 11.4 GB | 15.5 GB | 32 s | 55.4 (5.8) | 37.0 (8.7) | 45.0 (7.1) | **22 min** |

Pilot speeds for Llama-3.1-8B bf16 (500 items): en 40.1, roman_ur 23.4, ur 17.6 rows/s, similar to Qwen.

### Observations

- **Quantization did not slow scoring down.** NF4 ran at BF16 speed and INT8 was faster (Qwen +33% en). Before measuring we assumed bitsandbytes would be 1.5–4× slower, but that applies to token-by-token generation. Our workload is large prefill batches, where the GPU is compute-bound and dequantization overhead is hidden.
- **Wanda runs at exactly BF16 speed.** Zeroed weights are still stored and multiplied densely; unstructured sparsity gives no speedup on GPU.
- **Memory:** NF4 cuts Qwen's weights to 36% of BF16 and INT8 to 58%. Gemma shrinks less (NF4 58%) because a large part of it (per-layer embeddings) is not a linear layer and stays BF16.
- **Peak memory is about 4 GB above the weights.** The batch activations take this, since the token budget fills free memory. All variants fit 24 GB with room to spare.
- **Load times** are 12–37 s from local cache. The first download of an 8B model took about 165–172 s on this host.

## Wanda pruning

| Model | Sparsity | Wall time | C4 validation perplexity (16 × 2048 tokens) | Sparsity in pruned layers |
|---|---:|---:|---|---:|
| Qwen3-8B | 50% | 19.2 min | 7.163 → 8.566 (+20%) | 0.50 exactly |
| Qwen3-8B | 25% | 19.1 min | 7.163 → 7.252 (+1%) | 0.25 exactly |
| Gemma-4-E4B | 50% | about 20 min (finished; host went offline while its runs were loading) | not recorded | not recorded |

- Calibration: 128 × 2048-token windows from C4 (`en/c4-train.00000-of-01024`), seed 0, as in the paper. It downloads about 300 MB of C4 the first time.
- **Algorithmic cost:** `wanda_prune.py` collects the inputs of block *l* by running the model from the start and stopping after block *l*.
  - That is about L²/2 block-forwards per calibration window: 36 blocks for Qwen means ~666 instead of 36 in the official implementation, which caches block inputs.
  - Despite this it took about 19 min, because each partial pass is cheap and the GPU is saturated. It was estimated at 2–3 h before measuring.
  - Fine for 8B models; for much deeper models, consider caching block inputs instead.
- A pruned copy is the same size as the original (about 16 GB in BF16), since zeros are stored densely. The queue deletes each copy after its runs.

## Time per model for the whole of Exp 1 (5 variants × 3 languages), 3090 Ti

| Model | Runs | Pruning | Total |
|---|---:|---:|---:|
| Qwen3-8B | 42 + 43 + 33 + 42 + 42 = 202 min | 2 × 19 min | **about 4.0 h** |
| Gemma-4-E4B | 22 + 22 + 22 + about 22 + about 22 = about 110 min | about 2 × 20 min | **about 2.5 h** |

Planning rule of thumb on a 3090 Ti: **minutes per run ≈ 19,296 rows ÷ rows/s ÷ 60.** Expect rows/s of about 40 (en) / 23 (roman_ur) / 16 (ur) for an 8B model with a Llama/Qwen-style tokenizer. A 4090 is roughly 1.1–1.3× faster.

## Cost on Vast.ai (lessons)

- **GPU time was cheap:** the RTX 3090 Ti host was about $0.2–0.3/h. The whole Exp 1 for Qwen + Gemma (about 6.5 h) was about $1.5–2 of GPU time.
- **Bandwidth was the expensive part.** The host charged per TB of traffic ("$/hr plus bandwidth"). About 50 GB of downloads (three 8B models, torch, data) cost about $2.5 in under an hour; that was more than all the GPU time afterwards.
  - **Filter Vast offers by bandwidth price** ($/TB) and compare total cost.
  - Keep an instance (stop, don't destroy) while its cached models are needed.
  - Avoid uploading pruned checkpoints from expensive hosts.
- Download sizes: an 8B model in BF16 is about 16 GB; Ministral-3-8B about 18 GB; 3–4B students about 7–9 GB; torch and packages about 3 GB; data a few MB.
- **Reliability:** the host went offline after about 6.5 h with no warning. Results were safe because each run uploads to HF as it goes and the queue skips finished runs on restart.

## Estimates for planned work

| Work | GPU time (3090 Ti) | Download |
|---|---:|---:|
| Gemma-4-E4B wanda50 + wanda25 (prune and run) | about 1.3 h | 16 GB (if new instance) |
| Pilot: Ministral-3-8B, 500 items × 3 languages | about 5 min | about 18 GB |
| Pilot: Qalb-1.0-8B (Llama-3.1-8B architecture), 500 items × 3 languages | about 5 min | about 16 GB |
| Full Exp 1 for one more 8B model (5 variants × 3 languages) | about 4 h | about 16–18 GB |
| Exp 2 students (3 × bf16 × 3 languages; 3–4B models) | about 1–1.5 h | about 25 GB |
