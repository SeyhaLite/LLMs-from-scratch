# ជំពូក ៥៖ Pretraining លើទិន្នន័យគ្មានស្លាក (Pretraining on Unlabeled Data)

> **ប្រភព៖** ឯកសារណែនាំនេះសរសេរឡើងដោយសហគមន៍ខ្មែរ ដើម្បីធ្វើជាសន្ទស្សន៍ថត
> សម្រាប់កូដដែលចម្លងពី [`ch05/`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05) ក្នុងឃ្លាំង upstream
> [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0, (c) Sebastian Raschka)។
> អត្ថបទ និងរូបភាពរបស់សៀវភៅ *Build a Large Language Model (From Scratch)* (Manning)
> **មិនត្រូវបានចម្លងមកទីនេះទេ** — សូមមើល [LICENSE](../../LICENSE)។

&nbsp;

## ខ្លឹមសារក្នុងថត `ch05` នៃឃ្លាំងនេះ

| ថត | ខ្លឹមសារ |
|---|---|
| [`01_main-chapter-code`](01_main-chapter-code) | កូដមេរបស់ជំពូក |
| [`03_bonus_pretraining_on_gutenberg`](03_bonus_pretraining_on_gutenberg) | កូដ pretrain LLM បន្ថែមលើ corpus សៀវភៅទាំងមូលពី Project Gutenberg |
| [`05_bonus_hparam_tuning`](05_bonus_hparam_tuning) | script សម្រាប់ hyperparameter tuning |
| [`06_user_interface`](06_user_interface) | អនុវត្ត user interface ដើម្បីប្រើប្រាស់ LLM ដែល pretrain រួច |
| [`07_gpt_to_llama`](07_gpt_to_llama) | មគ្គុទ្ទេសក៍ម្តងមួយជំហាន បំលែង GPT ទៅ Llama 3.2 និងផ្ទុក pretrained weights ពី Meta AI |
| [`08_memory_efficient_weight_loading`](08_memory_efficient_weight_loading) | ផ្ទុក model weights តាម `load_state_dict` ដោយសន្សំសំចៃអង្គចងចាំជាងមុន |
| [`10_llm-training-speed`](10_llm-training-speed) | គន្លឹះ PyTorch performance ដើម្បីបង្កើនល្បឿនការបណ្តុះបណ្តាល LLM |
| [`11_qwen3`](11_qwen3) | អនុវត្ត Qwen3 0.6B និង Qwen3 30B-A3B (MoE) ពីមូលដ្ឋាន ព្រមទាំងកូដផ្ទុក pretrained weights |
| [`12_gemma3`](12_gemma3) | អនុវត្ត Gemma 3 270M ពីមូលដ្ឋាន និងកំណែប្រើ KV cache ព្រមទាំងកូដផ្ទុក pretrained weights |
| [`13_olmo3`](13_olmo3) | អនុវត្ត Olmo 3 7B និង 32B (Base, Instruct, Think) ពីមូលដ្ឋាន និងកំណែប្រើ KV cache |
| [`15_tiny-aya`](15_tiny-aya) | អនុវត្ត Tiny Aya ពីមូលដ្ឋាន (រួមមាន tests និង layer debugger) |
| [`16_qwen3.5`](16_qwen3.5) | អនុវត្ត Qwen3.5 ពីមូលដ្ឋាន ព្រមទាំងកំណែប្រើ KV cache |
| [`17_gemma4`](17_gemma4) | អនុវត្តកំណែ dense E2B និង E4B នៃ Gemma 4 ពីមូលដ្ឋាន |
| [`18_muon`](18_muon) | ការពន្យល់ពីរបៀបប្រើ Muon optimizer ជាមួយការរៀបចំបណ្តុះបណ្តាលម៉ូដែល GPT |

&nbsp;

## ថតដែលមានតែនៅ upstream (មិនបានចម្លងមកទីនេះ)

ថតខាងក្រោមមាននៅឃ្លាំង upstream ប៉ុន្តែ **មិនត្រូវបានចម្លង** មកឃ្លាំងខ្មែរនេះទេ
(មើលគោលការណ៍ក្នុង [`../README.md`](../README.md))។ សូមមើលផ្ទាល់នៅ upstream៖

| ថត | ខ្លឹមសារ |
|---|---|
| [`02_alternative_weight_loading`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/02_alternative_weight_loading) | កូដផ្ទុក GPT weights ពីប្រភពជំនួស ក្នុងករណី weights មិនអាចទាញពី OpenAI បាន |
| [`04_learning_rate_schedulers`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/04_learning_rate_schedulers) | training function កម្រិតខ្ពស់ រួមមាន learning rate scheduler និង gradient clipping |
| [`09_extending-tokenizers`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/09_extending-tokenizers) | អនុវត្ត GPT-2 BPE tokenizer ពីមូលដ្ឋាន |
| [`14_ch05_with_other_llms`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/14_ch05_with_other_llms) | អនុវត្តខ្លឹមសារជំពូក ៥ ជាមួយ LLM ផ្សេង (Qwen3, Llama 3.2) |
| [`19_cpu_mps_differences`](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/19_cpu_mps_differences) | ប្រមូលឧទាហរណ៍ដែលឧបករណ៍ CPU និង MPS មានឥរិយាបថខុសគ្នា |

&nbsp;

## ជំពូកបន្ទាប់

[ជំពូក ៦ — Finetuning for Classification](../ch06)

&nbsp;

## កូដមេរបស់ជំពូក

កូដមេស្ថិតក្នុងថត [`01_main-chapter-code`](01_main-chapter-code)។
កំណែ notebook ជាភាសាខ្មែរ (កូដដូច upstream ១០០% ព្រមទាំង comments និងការពន្យល់ជាខ្មែរ)
ស្ថិតក្នុងថត [`ណូតប៊ុក-ខ្មែរ/`](https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/%E1%9E%8E%E1%9E%BC%E1%9E%8F%E1%9E%94%E1%9E%BB%E1%9E%80-%E1%9E%81%E1%9F%92%E1%9E%98%E1%9F%82%E1%9E%9A)។

&nbsp;

## វីដេអូ code-along

អ្នកនិពន្ធមានវីដេអូ code-along ដែលគ្របដណ្តប់ខ្លឹមសារមួយចំនួននៃជំពូកនេះ
ជាសម្ភារៈបន្ថែម៖ <https://www.youtube.com/watch?v=Zar2TJv-sE0>

&nbsp;

## ទំព័រឯកសារខ្មែរ

- មេរៀនពេញលេញជាខ្មែរសម្រាប់ជំពូកនេះ៖ [គេហទំព័រឯកសារ](https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/source/%E1%9E%87%E1%9E%98%E1%9E%96%E1%9E%BD%E1%9E%80)
- សន្ទស្សន៍កូដទាំងមូល៖ [`../README.md`](../README.md)
