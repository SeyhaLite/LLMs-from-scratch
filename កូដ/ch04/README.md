# ជំពូក ៤៖ អនុវត្តម៉ូដែល GPT ពីមូលដ្ឋាន ដើម្បីបង្កើតអត្ថបទ (Implementing a GPT Model from Scratch to Generate Text)

> **ប្រភព៖** ឯកសារណែនាំនេះសរសេរឡើងដោយសហគមន៍ខ្មែរ ដើម្បីធ្វើជាសន្ទស្សន៍ថត
> សម្រាប់កូដដែលចម្លងពី [`ch04/`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) ក្នុងឃ្លាំង upstream
> [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0, (c) Sebastian Raschka)។
> អត្ថបទ និងរូបភាពរបស់សៀវភៅ *Build a Large Language Model (From Scratch)* (Manning)
> **មិនត្រូវបានចម្លងមកទីនេះទេ** — សូមមើល [LICENSE](../../LICENSE)។

&nbsp;

## ខ្លឹមសារក្នុងថត `ch04` នៃឃ្លាំងនេះ

| ថត | ខ្លឹមសារ |
|---|---|
| [`01_main-chapter-code`](01_main-chapter-code) | កូដមេរបស់ជំពូក |
| [`02_performance-analysis`](02_performance-analysis) | វិភាគសមត្ថភាពម៉ូដែល GPT ដែលអនុវត្តក្នុងជំពូកមេ |
| [`03_kv-cache`](03_kv-cache) | អនុវត្ត KV cache ដើម្បីបង្កើនល្បឿន text generation ពេល inference |
| [`04_gqa`](04_gqa) | សេចក្តីផ្តើមអំពី Grouped-Query Attention (GQA) — ប្រើក្នុង Llama 4, gpt-oss, Qwen3, Gemma 3 |
| [`05_mla`](05_mla) | សេចក្តីផ្តើមអំពី Multi-Head Latent Attention (MLA) — ប្រើក្នុង DeepSeek V3 |
| [`06_swa`](06_swa) | សេចក្តីផ្តើមអំពី Sliding Window Attention (SWA) — ប្រើក្នុង Gemma 3 |
| [`07_moe`](07_moe) | ការពន្យល់ និងការអនុវត្ត Mixture-of-Experts (MoE) |
| [`08_deltanet`](08_deltanet) | ការពន្យល់ Gated DeltaNet — វ៉ារ្យ៉ង់ linear attention (ប្រើក្នុង Qwen3-Next, Kimi Linear) |
| [`09_dsa`](09_dsa) | អនុវត្តម៉ូដែល GPT ជាមួយ KV cache + DSA (មើល `gpt_with_kv_dsa.py` និង `test_dsa.py`) |
| [`10_kv-sharing`](10_kv-sharing) | សេចក្តីផ្តើមអំពី cross-layer KV sharing — ប្រើក្នុង Gemma 4 E2B/E4B ដើម្បីកាត់បន្ថយអង្គចងចាំ KV-cache |

&nbsp;

## ជំពូកបន្ទាប់

[ជំពូក ៥ — Pretraining on Unlabeled Data](../ch05)

&nbsp;

## កូដមេរបស់ជំពូក

កូដមេស្ថិតក្នុងថត [`01_main-chapter-code`](01_main-chapter-code)។
កំណែ notebook ជាភាសាខ្មែរ (កូដដូច upstream ១០០% ព្រមទាំង comments និងការពន្យល់ជាខ្មែរ)
ស្ថិតក្នុងថត [`ណូតប៊ុក-ខ្មែរ/`](https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/%E1%9E%8E%E1%9E%BC%E1%9E%8F%E1%9E%94%E1%9E%BB%E1%9E%80-%E1%9E%81%E1%9F%92%E1%9E%98%E1%9F%82%E1%9E%9A)។

&nbsp;

## វីដេអូ code-along

អ្នកនិពន្ធមានវីដេអូ code-along ដែលគ្របដណ្តប់ខ្លឹមសារមួយចំនួននៃជំពូកនេះ
ជាសម្ភារៈបន្ថែម៖ <https://www.youtube.com/watch?v=YSAkgEarBGE>

&nbsp;

## ទំព័រឯកសារខ្មែរ

- មេរៀនពេញលេញជាខ្មែរសម្រាប់ជំពូកនេះ៖ [គេហទំព័រឯកសារ](https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/source/%E1%9E%87%E1%9E%98%E1%9E%96%E1%9E%BD%E1%9E%80)
- សន្ទស្សន៍កូដទាំងមូល៖ [`../README.md`](../README.md)
