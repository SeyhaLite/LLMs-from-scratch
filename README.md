# LLMs-from-scratch — ឯកសារសហគមន៍ភាសាខ្មែរ

ឯកសារសម្រាប់សិក្សាអំពី **Large Language Models (LLMs)** ពីមូលដ្ឋាន ជាភាសាខ្មែរ ដោយផ្អែកលើឃ្លាំងកូដ [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) របស់ Sebastian Raschka និងសៀវភៅ *Build a Large Language Model (From Scratch)* របស់ Manning។

> **សេចក្តីបញ្ជាក់៖** នេះជាគម្រោងឯកសាររបស់សហគមន៍ មិនមែនជាឯកសារផ្លូវការរបស់អ្នកនិពន្ធ ឬ Manning Publications ទេ។ ខ្លឹមសារពន្យល់ជាភាសាខ្មែរត្រូវបានរៀបចំឡើងដោយសហគមន៍ ហើយមិនចម្លងអត្ថបទ ឬរូបភាពដែលមានកម្មសិទ្ធិរបស់សៀវភៅឡើយ។

## តំណភ្ជាប់សំខាន់ៗ

| ធនធាន | តំណភ្ជាប់ |
|---|---|
| ឯកសារសិក្សាជាខ្មែរ (Read the Docs) | [llms-from-scratch.readthedocs.io](https://llms-from-scratch.readthedocs.io/km/latest/) |
| គេហទំព័រ GitHub Pages | [seyhalite.github.io/LLMs-from-scratch](https://seyhalite.github.io/LLMs-from-scratch/) |
| ឃ្លាំងឯកសារខ្មែរ | [SeyhaLite/LLMs-from-scratch](https://github.com/SeyhaLite/LLMs-from-scratch) |
| កូដផ្លូវការរបស់អ្នកនិពន្ធ (Upstream) | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) |
| សៀវភៅនៅ Manning | [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) |

## គម្រោងនេះមានអ្វីខ្លះ?

គម្រោងនេះជួយឲ្យអ្នករៀនយល់ពីរបៀបបង្កើតម៉ូដែលភាសាធំស្រដៀង GPT ដោយរៀនតាមជំហាន និងសាកល្បងកូដជាក់ស្តែង។ ប្រធានបទសំខាន់ៗរួមមាន៖

1. **ស្វែងយល់ពី LLM** — គោលគំនិត និងរបៀបដែលម៉ូដែលភាសាធំដំណើរការ។
2. **ទិន្នន័យអត្ថបទ** — Tokenization, Token IDs, និងការរៀបចំទិន្នន័យ។
3. **Attention** — យន្តការ Self-Attention និង Multi-Head Attention។
4. **ម៉ូដែល GPT** — ការសង់ស្ថាបត្យកម្ម GPT ពីផ្នែកតូចៗ។
5. **Pretraining** — ការបណ្តុះបណ្តាលម៉ូដែលលើទិន្នន័យ។
6. **Finetuning សម្រាប់ Classification** — ការកែសម្រួលម៉ូដែលសម្រាប់ការចាត់ថ្នាក់។
7. **Finetuning សម្រាប់ Instructions** — ការកែសម្រួលម៉ូដែលឲ្យឆ្លើយតបតាមការណែនាំ។

ឯកសារបន្ថែមមានភាគបន្ថែម A, D និង E, វចនានុក្រមពាក្យបច្ចេកទេស, សម្ភារៈសិក្សា និងឯកសារយោង។

## ចាប់ផ្តើមរៀន

អ្នកអាចរៀនតាមលំដាប់នេះ៖

| ជំហាន | អ្វីដែលត្រូវធ្វើ |
|---|---|
| ១. មើលទិដ្ឋភាពទូទៅ | ស្វែងយល់ថាគម្រោងនេះបង្រៀនអ្វី និងត្រូវត្រៀមអ្វីខ្លះ។ |
| ២. រៀបចំបរិស្ថាន | ដំឡើង Python និងបណ្ណាល័យដែលត្រូវការ។ |
| ៣. រៀនតាមជំពូក | អានការពន្យល់ជាខ្មែរតាមជំពូក ១–៧។ |
| ៤. សាកល្បង Notebook | ដំណើរការកូដ និងលំហាត់នៅក្នុង JupyterLab ឬ VS Code។ |
| ៥. សិក្សាបន្ថែម | ពិនិត្យភាគបន្ថែម វចនានុក្រម និងសម្ភារៈផ្សេងៗ។ |

- [ចាប់ផ្តើមពីទិដ្ឋភាពទូទៅ](https://seyhalite.github.io/LLMs-from-scratch/ការចាប់ផ្តើម/ទិដ្ឋភាពទូទៅ.html)
- [ការដំឡើងបរិស្ថាន](https://seyhalite.github.io/LLMs-from-scratch/ការចាប់ផ្តើម/ការដំឡើងបរិស្ថាន.html)
- [ការទាញយកកូដ](https://seyhalite.github.io/LLMs-from-scratch/ការចាប់ផ្តើម/ការទាញយកកូដ.html)

**ចំណេះដឹងមុនចាប់ផ្តើម៖** គួរមានមូលដ្ឋាន Python។ ការយល់ដឹងខ្លះអំពី Neural Networks និង PyTorch នឹងជួយឲ្យរៀនបានងាយជាងមុន ប៉ុន្តែអាចរៀនបន្ថែមតាមឯកសារបាន។

## ឯកសារ និងកូដដែលមានក្នុងឃ្លាំង

### ១. ឯកសារពន្យល់ជាភាសាខ្មែរ

ថត **source/** ផ្ទុកឯកសារដើម Sphinx/RST ដែលប្រើបង្កើតគេហទំព័រ។ វាមានទំព័រចាប់ផ្តើម ៣ ទំព័រ ជំពូកស្នូល ៧ ទំព័រ និងទំព័រយោង ៥ ទំព័រ។

### ២. Notebook ជាភាសាខ្មែរ

ថត **ណូតប៊ុក-ខ្មែរ/** មាន Notebook សរុប **៦៨ ឯកសារ**៖

- **១៩ Notebook ស្នូល** នៅថតខាងក្រៅ សម្រាប់ជំពូក និងលំហាត់សំខាន់ៗ។
- **៤៩ Notebook បន្ថែម** នៅក្នុងថត **ណូតប៊ុក-ខ្មែរ/បន្ថែម/** ដែលរៀបតាមជំពូក និងប្រធានបទ។

Notebook ទាំងនេះមានកូដសម្រាប់អនុវត្ត ព្រមទាំងការពន្យល់ និង comments ជាភាសាខ្មែរ។ ព័ត៌មានអំពីឯកសារនីមួយៗមាននៅក្នុង [ទំព័រ Notebook ខ្មែរ](https://seyhalite.github.io/LLMs-from-scratch/យោង/ណូតប៊ុក-ខ្មែរ.html)។

### ៣. កូដ និងទិន្នន័យពី Upstream

ថត **កូដ/** មានកូដ និងឯកសារទិន្នន័យដែលបានចម្លងពីឃ្លាំង upstream ដោយរក្សារចនាសម្ព័ន្ធថត ដើម្បីជួយឲ្យ Notebook អាចយោងទៅកាន់ឯកសារជំនួយបាន។

- **កូដ/ch01/** ដល់ **កូដ/ch07/** — សន្ទស្សន៍តាមជំពូក និងកូដដែលមាន។
- **កូដ/requirements.txt**, **កូដ/pyproject.toml**, **កូដ/pixi.toml** — ការកំណត់បរិស្ថានសម្រាប់កូដ។
- **កូដ/README.md** — ព័ត៌មានអំពីវិធីចម្លងកូដ និងអាជ្ញាបណ្ណ។

**ចំណាំ៖** កូដក្នុងថតនេះមិនត្រូវបានអះអាងថាជាកំណែ upstream ថ្មីបំផុតទេ។ សូមមើលផ្នែកស្ថានភាពសមកាលកម្មខាងក្រោម។

## តម្រូវការ និងការដំឡើង

### សម្រាប់បង្កើតគេហទំព័រឯកសារ

គម្រោង Read the Docs កំណត់ Python 3.12 និងប្រើ Sphinx។ ដើម្បីបង្កើតគេហទំព័រនៅលើកុំព្យូទ័ររបស់អ្នក៖

1. Clone ឃ្លាំង៖

   ~~~bash
   git clone https://github.com/SeyhaLite/LLMs-from-scratch.git
   cd LLMs-from-scratch
   ~~~

2. បង្កើត Virtual Environment៖

   ~~~bash
   python -m venv .venv
   ~~~

   Linux/macOS៖

   ~~~bash
   source .venv/bin/activate
   ~~~

   Windows PowerShell៖

   ~~~powershell
   .venv\Scripts\Activate.ps1
   ~~~

3. ដំឡើងបណ្ណាល័យសម្រាប់បង្កើតឯកសារ៖

   ~~~bash
   python -m pip install -r requirements.txt
   ~~~

4. Build ឯកសារ HTML៖

   ~~~bash
   sphinx-build -W --keep-going source docs
   ~~~

   ពាក្យបញ្ជានេះបង្កើតលទ្ធផល HTML នៅក្នុងថត **docs/** ដែលជាថតគេហទំព័រ GitHub Pages របស់ឃ្លាំងនេះ។ ជម្រើស **-W** ធ្វើឲ្យ Warning ត្រូវបានចាត់ទុកជាកំហុស ដើម្បីជួយរកបញ្ហាក្នុងឯកសារ។

5. មើលគេហទំព័រលើម៉ាស៊ីនខ្លួនឯង៖

   ~~~bash
   python -m http.server 8000 --directory docs
   ~~~

   បន្ទាប់មកបើក **http://localhost:8000** ក្នុងកម្មវិធីរុករក។

> **សំខាន់៖** **requirements.txt** នៅថត root គឺសម្រាប់បង្កើតគេហទំព័រឯកសារប៉ុណ្ណោះ។ វាមិនដំឡើង PyTorch ឬបណ្ណាល័យសម្រាប់ដំណើរការកូដ LLM ទេ។ សម្រាប់ dependencies របស់កូដ សូមពិនិត្យ **កូដ/requirements.txt** ឬ **កូដ/pyproject.toml** និងការណែនាំក្នុងទំព័រដំឡើងបរិស្ថាន។

### សម្រាប់ដំណើរការ Notebook

អ្នកអាចប្រើ JupyterLab ឬ VS Code ដើម្បីបើកឯកសារ **.ipynb**។ សូមអាន [ការណែនាំអំពី Notebook](https://seyhalite.github.io/LLMs-from-scratch/យោង/ណូតប៊ុក-ខ្មែរ.html) ជាមុន ព្រោះ Notebook មួយចំនួនត្រូវការឯកសារជំនួយ ទិន្នន័យ ឬ pretrained weights។ ការទាញយក weights អាចត្រូវការអ៊ីនធឺណិត និងទំហំផ្ទុកគ្រប់គ្រាន់។

## រចនាសម្ព័ន្ធឃ្លាំង

~~~text
LLMs-from-scratch/
├── source/              # ឯកសារដើម Sphinx/RST ជាភាសាខ្មែរ
│   ├── _static/         # CSS, រូបភាព និងធនធានគេហទំព័រ
│   ├── ការចាប់ផ្តើម/    # ទិដ្ឋភាពទូទៅ និងការដំឡើង
│   ├── ជំពូក/           # ជំពូក ១–៧
│   └── យោង/             # Notebook, វចនានុក្រម និងឯកសារយោង
├── កូដ/                 # កូដ និងទិន្នន័យពី upstream
├── ណូតប៊ុក-ខ្មែរ/        # Notebook ជាភាសាខ្មែរ (៦៨)
├── docs/                # គេហទំព័រ HTML ដែលបាន build រួច
├── requirements.txt     # Dependencies សម្រាប់ឯកសារ Sphinx
├── .readthedocs.yaml    # ការកំណត់ Read the Docs
├── Makefile             # ពាក្យបញ្ជា build របស់ Sphinx
├── LICENSE              # CC BY 4.0 សម្រាប់ឯកសារខ្មែរ
└── LICENSE-CODE         # MIT សម្រាប់កូដ/config របស់គម្រោង
~~~

## ស្ថានភាពសមកាលកម្មជាមួយ Upstream

ដើម្បីកុំឲ្យច្រឡំរវាងកំណែកូដដែលបានចម្លង និងកូដថ្មីបំផុត៖

| ព័ត៌មាន | កំណែ |
|---|---|
| Snapshot ដែល README នេះកត់ត្រាសម្រាប់ថត **កូដ/** | **cdbd33e6** — 2026-10-02 |
| Upstream **main** ដែលបានពិនិត្យចុងក្រោយ | **0311973e** — 2026-11-09 |
| ភាពខុសគ្នា | Upstream មាន ១៣ commits ថ្មីជាង snapshot |

មានន័យថា កូដក្នុងថត **កូដ/** នៅតាម snapshot ដែលបានកត់ត្រា មិនទាន់រួមបញ្ចូលការផ្លាស់ប្ដូរ upstream ទាំង ១៣ commits ដែលកើតឡើងក្រោយថ្ងៃទី ២ ខែតុលា ឆ្នាំ ២០២៦ ទេ។ សម្រាប់កូដថ្មីបំផុត សូមពិនិត្យ [ឃ្លាំង upstream](https://github.com/rasbt/LLMs-from-scratch) ដោយផ្ទាល់។ មុនពេលធ្វើសមកាលកម្មលើកក្រោយ គួរប្រៀបធៀបកូដ និងសាកល្បងឡើងវិញ។

## ការដាក់ពង្រាយ

- **Read the Docs:** កំណត់រចនាសម្ព័ន្ធនៅក្នុង **.readthedocs.yaml** និងប្រើ **source/conf.py** ជា Sphinx configuration។
- **GitHub Pages:** ឯកសារ HTML ដែលបាន build រក្សាទុកនៅក្នុង **docs/** និងមាន **.nojekyll**។

ការកែប្រែឯកសារដើមត្រូវធ្វើនៅក្នុង **source/**។ បន្ទាប់ពីកែប្រែ ត្រូវ build ឡើងវិញ ហើយពិនិត្យលទ្ធផលក្នុង **docs/** មុនពេល commit។

## អាជ្ញាបណ្ណ និងកម្មសិទ្ធិបញ្ញា

| ផ្នែក | អាជ្ញាបណ្ណ / កម្មសិទ្ធិ |
|---|---|
| អត្ថបទពន្យល់ និងការរៀបចំជាភាសាខ្មែរ | [CC BY 4.0](LICENSE) |
| Theme, template និង build configuration ដែលសហគមន៍បង្កើត | [MIT](LICENSE-CODE) |
| កូដដែលបានចម្លងពី upstream | Apache License 2.0 — Sebastian Raschka |
| អត្ថបទ និងរូបភាពសៀវភៅ | ជាកម្មសិទ្ធិរបស់ Manning Publications និងមិនត្រូវបានចម្លងក្នុងគម្រោងនេះទេ |

សូមអានឯកសារអាជ្ញាបណ្ណនីមួយៗមុនពេលប្រើប្រាស់ ឬចែកចាយឡើងវិញ។ ការចម្លងកូដ upstream ត្រូវរក្សាការដាក់ប្រភព និងលក្ខខណ្ឌអាជ្ញាបណ្ណដើម។

## ការចូលរួមចំណែក

អ្នកអាចចូលរួមកែលម្អគម្រោងនេះតាមរយៈ Issue ឬ Pull Request៖

- រាយការណ៍កំហុសក្នុងការពន្យល់ តំណភ្ជាប់ ឬការបង្ហាញអក្សរខ្មែរ។
- កែលម្អការពន្យល់ឲ្យងាយយល់ ដោយរក្សាឈ្មោះបច្ចេកទេសអង់គ្លេសនៅកន្លែងដែលចាំបាច់។
- ពិនិត្យថា Notebook និងឯកសារយោងនៅតែត្រូវនឹងកូដ upstream។
- Build ដោយប្រើ **sphinx-build -W --keep-going source docs** ហើយដោះស្រាយ Warning មុនពេលផ្ញើ Pull Request។
- កុំបញ្ចូលអត្ថបទ ឬរូបភាពពីសៀវភៅដែលមានកម្មសិទ្ធិដោយគ្មានសិទ្ធិប្រើប្រាស់។

សូមបើក [Issues](https://github.com/SeyhaLite/LLMs-from-scratch/issues) ឬ [Pull Requests](https://github.com/SeyhaLite/LLMs-from-scratch/pulls) ដើម្បីរាយការណ៍ ឬស្នើការកែលម្អ។
