# Related work and our contribution

Compiled 25 Oct 2026 from web search results and abstracts. **Verify every claim against the full paper before citing.** Several entries are 2026 preprints.

## 1. Compression and social bias (closest work; all English)

| Paper | Findings | Not covered |
|---|---|---|
| Hua, Lotfi & Chen, *Investigating Social Bias Changes in Quantized Language Models*, COLM 2026 (arXiv 2602.06181) | 50 quantized models, 13 bias datasets (PostTrainingBiasBench). "Bias flipping" in up to 21% of responses while aggregate scores stay flat. Flips are 3–11× more likely for high-uncertainty responses. 4-bit changes behaviour 4–6× more than 8-bit. Group shifts are up to +18.6% / −14.1%. Larger models are not consistently more robust | quantization only; no non-English evaluation stated |
| *The Asymmetric Harms of LLM Compression*, 2026 (arXiv 2608.19670) | GPTQ/AWQ/OmniQuant/AQLM (2–4 bit) and magnitude/Wanda/SparseGPT (30–70%, 2:4, 4:8) plus structured pruning, on BBQ and WinoBias, with Llama-3.1-8B-Instruct, Qwen-3-8B and Gemma-2-9B-it. Small overall bias change hides large opposing subgroup shifts (e.g. −53.1 subgroup vs −2.2 overall) | English; no distillation; instruction-tuned models only |
| Marcuzzi, Ning, Schwartz & Gurevych, *How Quantization Shapes Bias in LLMs*, arXiv 2508.18088 (2025/26) | 13 benchmarks; probability- and generation-based metrics. Quantization slightly increases stereotypes and unfairness in generation (more when aggressive), reduces toxicity, and leaves sentiment unchanged | no multilingual coverage stated |
| Paxton, Wuraola & Thakker, *Effects of Model Scale and Quantization on Social Bias and Abstention in QA*, arXiv 2610.29657 (Oct 2026) | BBQ; Qwen3, Gemma-3, SmolLM2. In 20 of 26 configurations, lower bias came with more over-abstention; forced-choice and prompt-variation checks | English BBQ; quantization only |
| *Quantization Undoes Alignment*, arXiv 2605.15208 (2026) | 3-bit PTQ on BBQ: 6–21% of items show new stereotypical behaviour; willingness to choose "unknown" falls by 17.4% | English |
| *Weight Pruning Amplifies Bias*, arXiv 2605.08137 (2026); Kirsten et al. (2025) | Pruning amplifies bias; structured pruning reduces bias while AWQ increases it | English |
| Gonçalves & Strubell, EMNLP 2023 (arXiv 2312.05662) | Early study of compression and social bias | English |

## 2. Compression and multilingual performance (accuracy, not bias)

- Marchisio et al., *How Does Quantization Affect Multilingual LLMs?*, Findings EMNLP 2024: non-Latin scripts degrade more, and degradation correlates with less training data.
- Long-context quantization, EMNLP 2025 (2025.emnlp-main.479): 4-bit loses more, especially for non-English and low-resource inputs.
- *The Multilingual Quantization Tax*, arXiv 2608.09941 (2026); *The Uneven Impact of PTQ in MT*, arXiv 2508.20893; *Language-Conditional Dequantization*, arXiv 2608.11786.

## 3. Calibration language for pruning and quantization (accuracy, not bias)

- *Investigating Language-Specific Calibration for Pruning Multilingual LLMs* (arXiv 2408.14398; TACL): Wanda on Spanish Belebele ranges from 47.8 (Swahili calibration) to 57.3 (German). Target-language calibration is best in about 58% of Llama-3 cases.
- Chimoto et al., *Calibrating Beyond English*, EACL 2026 (arXiv 2601.18306): multilingual calibration for quantization.
- Williams & Aletras, *On the Impact of Calibration Data in PTQ and Pruning* (arXiv 2311.09755): pruning is more sensitive to calibration data than quantization.
- *Calibration Language Shapes Block-Influence Pruning* (arXiv 2610.21383, 2026).

## 4. Culturally adapted bias benchmarks (never tested under compression)

PakBBQ (Hashmat et al., EMNLP 2025), KoBBQ, MBBQ (Neplenbroek et al., 2024; Dutch/Spanish/Turkish), BharatBBQ (8 Indian languages; check whether it covers Urdu), JBBQ, EsBBQ/CaBBQ, GG-BBQ.
Roman Urdu: only idiom translation (arXiv 2510.17460; native script beats Roman), UrduBench (arXiv 2601.21000; native script only) and transliteration work (LoResMT 2025). **No Roman Urdu bias benchmark found.**

## Our contribution (bias-focused)

1. **First compression → bias study in a culturally adapted, low-resource setting:** Pakistan-specific groups (regional, religious, language formality) in Urdu and Roman Urdu. All compression-bias work above is English BBQ/WinoBias.
2. **Script as a dimension of bias:** identical items in Urdu script vs Roman Urdu. Plus the **Roman Urdu PakBBQ release**, the first Roman Urdu bias benchmark.
3. **Calibration language → bias:** the C4 vs mixed en/ur/roman_ur Wanda ablation. The calibration literature measures accuracy only, and the bias literature uses English calibration.
4. **Pruning + quantization + distillation in one paired design** (each variant vs its own BF16; distillation via official base students).
5. **Bias separated from abstention and accuracy** (per Paxton et al.): report bias conditional on the Unknown rate and accuracy, from stored per-item probabilities.
6. **Replicating 2026 bias-flip findings in a new setting:** flip rates, group asymmetries and flips vs uncertainty (Hua et al.), in Urdu and Roman Urdu. Early data: 2–3× more flips in Urdu/Roman Urdu than English.
7. **Language adaptation × compression:** Qalb (Urdu-adapted Llama-3.1-8B) vs its base.
8. **Methodology:**
   - corrected PakBBQ scoring (polarity, tag matching, unknown handling);
   - the script-specific letter artifact (dead `ا` slot in 3 model families) and `ur_abc` robustness;
   - option-order averaging with a noise floor.

## Implications for the analysis plan

- Must match the 2026 papers: per-group bias change (Asymmetric Harms), abstention-aware bias (Paxton), flips vs uncertainty (Hua). All are analysis on existing data.
- AWQ/GPTQ (Tier 2) is used by nearly all competing papers, so adding it makes our results directly comparable, and it extends the calibration-language question to quantization.
