# ជំពូក ៣៖ សរសេរកូដយន្តការ Attention (Coding Attention Mechanisms)

> **ប្រភព៖** ឯកសារណែនាំនេះសរសេរឡើងដោយសហគមន៍ខ្មែរ ដើម្បីធ្វើជាសន្ទស្សន៍ថត
> សម្រាប់កូដដែលចម្លងពី [`ch03/`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03) ក្នុងឃ្លាំង upstream
> [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0, (c) Sebastian Raschka)។
> អត្ថបទ និងរូបភាពរបស់សៀវភៅ *Build a Large Language Model (From Scratch)* (Manning)
> **មិនត្រូវបានចម្លងមកទីនេះទេ** — សូមមើល [LICENSE](../../LICENSE)។

&nbsp;

## ខ្លឹមសារក្នុងថត `ch03` នៃឃ្លាំងនេះ

| ថត | ខ្លឹមសារ |
|---|---|
| [`01_main-chapter-code`](01_main-chapter-code) | កូដមេរបស់ជំពូក |
| [`02_bonus_efficient-multihead-attention`](02_bonus_efficient-multihead-attention) | អនុវត្ត និងប្រៀបធៀបកំណែ multihead-attention ផ្សេងៗ |

&nbsp;

## ថតដែលមានតែនៅ upstream (មិនបានចម្លងមកទីនេះ)

ថតខាងក្រោមមាននៅឃ្លាំង upstream ប៉ុន្តែ **មិនត្រូវបានចម្លង** មកឃ្លាំងខ្មែរនេះទេ
(មើលគោលការណ៍ក្នុង [`../README.md`](../README.md))។ សូមមើលផ្ទាល់នៅ upstream៖

| ថត | ខ្លឹមសារ |
|---|---|
| [`03_understanding-buffers`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/03_understanding-buffers) | ពន្យល់គំនិត PyTorch buffers ដែលប្រើអនុវត្ត causal attention |

&nbsp;

## ជំពូកបន្ទាប់

[ជំពូក ៤ — Implementing a GPT Model from Scratch](../ch04)

&nbsp;

## កូដមេរបស់ជំពូក

កូដមេស្ថិតក្នុងថត [`01_main-chapter-code`](01_main-chapter-code)។
កំណែ notebook ជាភាសាខ្មែរ (កូដដូច upstream ១០០% ព្រមទាំង comments និងការពន្យល់ជាខ្មែរ)
ស្ថិតក្នុងថត [`ណូតប៊ុក-ខ្មែរ/`](https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/%E1%9E%8E%E1%9E%BC%E1%9E%8F%E1%9E%94%E1%9E%BB%E1%9E%80-%E1%9E%81%E1%9F%92%E1%9E%98%E1%9F%82%E1%9E%9A)។

&nbsp;

## វីដេអូ code-along

អ្នកនិពន្ធមានវីដេអូ code-along ដែលគ្របដណ្តប់ខ្លឹមសារមួយចំនួននៃជំពូកនេះ
ជាសម្ភារៈបន្ថែម៖ <https://www.youtube.com/watch?v=-Ll8DtpNtvk>

&nbsp;

## ទំព័រឯកសារខ្មែរ

- មេរៀនពេញលេញជាខ្មែរសម្រាប់ជំពូកនេះ៖ [គេហទំព័រឯកសារ](https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/source/%E1%9E%87%E1%9E%98%E1%9E%96%E1%9E%BD%E1%9E%80)
- សន្ទស្សន៍កូដទាំងមូល៖ [`../README.md`](../README.md)
