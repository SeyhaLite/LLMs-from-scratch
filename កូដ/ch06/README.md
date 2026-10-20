# ជំពូក ៦៖ Finetuning សម្រាប់ការចាត់ថ្នាក់ (Finetuning for Classification)

> **ប្រភព៖** ឯកសារណែនាំនេះសរសេរឡើងដោយសហគមន៍ខ្មែរ ដើម្បីធ្វើជាសន្ទស្សន៍ថត
> សម្រាប់កូដដែលចម្លងពី [`ch06/`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06) ក្នុងឃ្លាំង upstream
> [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0, (c) Sebastian Raschka)។
> អត្ថបទ និងរូបភាពរបស់សៀវភៅ *Build a Large Language Model (From Scratch)* (Manning)
> **មិនត្រូវបានចម្លងមកទីនេះទេ** — សូមមើល [LICENSE](../../LICENSE)។

&nbsp;

## ខ្លឹមសារក្នុងថត `ch06` នៃឃ្លាំងនេះ

| ថត | ខ្លឹមសារ |
|---|---|
| [`01_main-chapter-code`](01_main-chapter-code) | កូដមេរបស់ជំពូក |
| [`02_bonus_additional-experiments`](02_bonus_additional-experiments) | ការពិសោធន៍បន្ថែម (បណ្តុះបណ្តាលលើ token ចុងក្រោយ vs token ដំបូង, បង្រីកប្រវែង input ជាដើម) |
| [`03_bonus_imdb-classification`](03_bonus_imdb-classification) | ប្រៀបធៀប LLM របស់ជំពូក ៦ ជាមួយម៉ូដែលផ្សេង លើ dataset IMDb ៥០ ពាន់ movie review sentiment |
| [`04_user_interface`](04_user_interface) | អនុវត្ត user interface ដើម្បីប្រើប្រាស់ LLM ដែល pretrain រួច |

&nbsp;

## ជំពូកបន្ទាប់

[ជំពូក ៧ — Finetuning to Follow Instructions](../ch07)

&nbsp;

## កូដមេរបស់ជំពូក

កូដមេស្ថិតក្នុងថត [`01_main-chapter-code`](01_main-chapter-code)។
កំណែ notebook ជាភាសាខ្មែរ (កូដដូច upstream ១០០% ព្រមទាំង comments និងការពន្យល់ជាខ្មែរ)
ស្ថិតក្នុងថត [`ណូតប៊ុក-ខ្មែរ/`](https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/%E1%9E%8E%E1%9E%BC%E1%9E%8F%E1%9E%94%E1%9E%BB%E1%9E%80-%E1%9E%81%E1%9F%92%E1%9E%98%E1%9F%82%E1%9E%9A)។

&nbsp;

## វីដេអូ code-along

អ្នកនិពន្ធមានវីដេអូ code-along ដែលគ្របដណ្តប់ខ្លឹមសារមួយចំនួននៃជំពូកនេះ
ជាសម្ភារៈបន្ថែម៖ <https://www.youtube.com/watch?v=5PFXJYme4ik>

&nbsp;

## ទំព័រឯកសារខ្មែរ

- មេរៀនពេញលេញជាខ្មែរសម្រាប់ជំពូកនេះ៖ [គេហទំព័រឯកសារ](https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/source/%E1%9E%87%E1%9E%98%E1%9E%96%E1%9E%BD%E1%9E%80)
- សន្ទស្សន៍កូដទាំងមូល៖ [`../README.md`](../README.md)
