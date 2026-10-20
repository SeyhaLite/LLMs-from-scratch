# ជំពូក ២៖ ការធ្វើការជាមួយទិន្នន័យអត្ថបទ (Working with Text Data)

> **ប្រភព៖** ឯកសារណែនាំនេះសរសេរឡើងដោយសហគមន៍ខ្មែរ ដើម្បីធ្វើជាសន្ទស្សន៍ថត
> សម្រាប់កូដដែលចម្លងពី [`ch02/`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02) ក្នុងឃ្លាំង upstream
> [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0, (c) Sebastian Raschka)។
> អត្ថបទ និងរូបភាពរបស់សៀវភៅ *Build a Large Language Model (From Scratch)* (Manning)
> **មិនត្រូវបានចម្លងមកទីនេះទេ** — សូមមើល [LICENSE](../../LICENSE)។

&nbsp;

## ខ្លឹមសារក្នុងថត `ch02` នៃឃ្លាំងនេះ

| ថត | ខ្លឹមសារ |
|---|---|
| [`01_main-chapter-code`](01_main-chapter-code) | កូដមេរបស់ជំពូក និងចម្លើយលំហាត់ |
| [`02_bonus_bytepair-encoder`](02_bonus_bytepair-encoder) | កូដ bonus សម្រាប់ benchmark ការអនុវត្ត byte pair encoder ផ្សេងៗ |
| [`05_bpe-from-scratch`](05_bpe-from-scratch) | អនុវត្ត និងបណ្តុះបណ្តាល GPT-2 BPE tokenizer ពីមូលដ្ឋាន |

&nbsp;

## ថតដែលមានតែនៅ upstream (មិនបានចម្លងមកទីនេះ)

ថតខាងក្រោមមាននៅឃ្លាំង upstream ប៉ុន្តែ **មិនត្រូវបានចម្លង** មកឃ្លាំងខ្មែរនេះទេ
(មើលគោលការណ៍ក្នុង [`../README.md`](../README.md))។ សូមមើលផ្ទាល់នៅ upstream៖

| ថត | ខ្លឹមសារ |
|---|---|
| [`03_bonus_embedding-vs-matmul`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/03_bonus_embedding-vs-matmul) | បង្ហាញថា embedding layer និង fully connected layer លើ one-hot vector គឺសមមូលគ្នា |
| [`04_bonus_dataloader-intuition`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/04_bonus_dataloader-intuition) | ពន្យល់ data loader ដោយប្រើលេខសាមញ្ញ ជំនួសឱ្យអត្ថបទ |
| [`06_bonus_simple-tokenizer-v3`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/06_bonus_simple-tokenizer-v3) | កំណែ SimpleTokenizerV3 |

&nbsp;

## ជំពូកបន្ទាប់

[ជំពូក ៣ — Coding Attention Mechanisms](../ch03)

&nbsp;

## កូដមេរបស់ជំពូក

កូដមេស្ថិតក្នុងថត [`01_main-chapter-code`](01_main-chapter-code)។
កំណែ notebook ជាភាសាខ្មែរ (កូដដូច upstream ១០០% ព្រមទាំង comments និងការពន្យល់ជាខ្មែរ)
ស្ថិតក្នុងថត [`ណូតប៊ុក-ខ្មែរ/`](https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/%E1%9E%8E%E1%9E%BC%E1%9E%8F%E1%9E%94%E1%9E%BB%E1%9E%80-%E1%9E%81%E1%9F%92%E1%9E%98%E1%9F%82%E1%9E%9A)។

&nbsp;

## វីដេអូ code-along

អ្នកនិពន្ធមានវីដេអូ code-along ដែលគ្របដណ្តប់ខ្លឹមសារមួយចំនួននៃជំពូកនេះ
ជាសម្ភារៈបន្ថែម៖ <https://www.youtube.com/watch?v=341Rb8fJxY0>

&nbsp;

## ទំព័រឯកសារខ្មែរ

- មេរៀនពេញលេញជាខ្មែរសម្រាប់ជំពូកនេះ៖ [គេហទំព័រឯកសារ](https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/source/%E1%9E%87%E1%9E%98%E1%9E%96%E1%9E%BD%E1%9E%80)
- សន្ទស្សន៍កូដទាំងមូល៖ [`../README.md`](../README.md)
