# ជំពូក ៧៖ Finetuning ដើម្បីធ្វើតាមការណែនាំ (Finetuning to Follow Instructions)

> **ប្រភព៖** ឯកសារណែនាំនេះសរសេរឡើងដោយសហគមន៍ខ្មែរ ដើម្បីធ្វើជាសន្ទស្សន៍ថត
> សម្រាប់កូដដែលចម្លងពី [`ch07/`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07) ក្នុងឃ្លាំង upstream
> [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0, (c) Sebastian Raschka)។
> អត្ថបទ និងរូបភាពរបស់សៀវភៅ *Build a Large Language Model (From Scratch)* (Manning)
> **មិនត្រូវបានចម្លងមកទីនេះទេ** — សូមមើល [LICENSE](../../LICENSE)។

&nbsp;

## ខ្លឹមសារក្នុងថត `ch07` នៃឃ្លាំងនេះ

| ថត | ខ្លឹមសារ |
|---|---|
| [`01_main-chapter-code`](01_main-chapter-code) | កូដមេរបស់ជំពូក និងចម្លើយលំហាត់ |
| [`02_dataset-utilities`](02_dataset-utilities) | កូដជំនួយសម្រាប់រៀបចំ instruction dataset |
| [`03_model-evaluation`](03_model-evaluation) | កូដជំនួយសម្រាប់វាយតម្លៃ instruction responses ដោយប្រើម៉ូដែល Llama 3 ក្នុងតំបន់ និង GPT-4 API |
| [`04_preference-tuning-with-dpo`](04_preference-tuning-with-dpo) | អនុវត្ត preference finetuning ជាមួយ Direct Preference Optimization (DPO) |
| [`05_dataset-generation`](05_dataset-generation) | កូដបង្កើត និងកែលម្អ synthetic dataset សម្រាប់ instruction finetuning |
| [`06_user_interface`](06_user_interface) | អនុវត្ត user interface ដើម្បីប្រើប្រាស់ LLM ដែល pretrain រួច |

&nbsp;

## បន្ទាប់ពីជំពូកនេះ

ជំពូក ៧ ជាជំពូកចុងក្រោយនៃផ្នែកស្នូល។ សូមបន្តទៅ **ភាគបន្ថែម (Appendix A–E)**
ដែលមានការពន្យល់អំពី PyTorch, វិធីសាស្ត្រឆ្លើយតបលំហាត់, training loop និង LoRA។

&nbsp;

## កូដមេរបស់ជំពូក

កូដមេស្ថិតក្នុងថត [`01_main-chapter-code`](01_main-chapter-code)។
កំណែ notebook ជាភាសាខ្មែរ (កូដដូច upstream ១០០% ព្រមទាំង comments និងការពន្យល់ជាខ្មែរ)
ស្ថិតក្នុងថត [`ណូតប៊ុក-ខ្មែរ/`](https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/%E1%9E%8E%E1%9E%BC%E1%9E%8F%E1%9E%94%E1%9E%BB%E1%9E%80-%E1%9E%81%E1%9F%92%E1%9E%98%E1%9F%82%E1%9E%9A)។

&nbsp;

## វីដេអូ code-along

អ្នកនិពន្ធមានវីដេអូ code-along ដែលគ្របដណ្តប់ខ្លឹមសារមួយចំនួននៃជំពូកនេះ
ជាសម្ភារៈបន្ថែម៖ <https://www.youtube.com/watch?v=4yNswvhPWCQ>

&nbsp;

## ទំព័រឯកសារខ្មែរ

- មេរៀនពេញលេញជាខ្មែរសម្រាប់ជំពូកនេះ៖ [គេហទំព័រឯកសារ](https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/source/%E1%9E%87%E1%9E%98%E1%9E%96%E1%9E%BD%E1%9E%80)
- សន្ទស្សន៍កូដទាំងមូល៖ [`../README.md`](../README.md)
