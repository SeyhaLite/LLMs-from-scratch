# LLMs-from-scratch — ឯកសារសហគមន៍ភាសាខ្មែរ

គេហទំព័រឯកសារសហគមន៍ (community documentation) ភាសាខ្មែរ សម្រាប់គម្រោង
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) —
ឃ្លាំងកូដផ្លូវការនៃសៀវភៅ *Build a Large Language Model (From Scratch)*
របស់ Sebastian Raschka (Manning)។ ឯកសារនេះសង្ខេបព័ត៌មានពិតពីឃ្លាំង upstream
ដោយមិនចម្លងអត្ថបទសៀវភៅ ឬរូបភាពដែលមានកម្មសិទ្ធិបញ្ញាទេ។

## តំណភ្ជាប់

- ឯកសារចម្បង (Read the Docs): https://llms-from-scratch.readthedocs.io/
- គេហទំព័រកញ្ចក់ (GitHub Pages): https://seyhalite.github.io/LLMs-from-scratch/
- ឃ្លាំង upstream: https://github.com/rasbt/LLMs-from-scratch
- upstream version ដែលតាមដាន: `main` @ `cdbd33e` (2026-10-02)

## មាតិកា

| ផ្នែក | ទំព័រ |
|---|---|
| ការចាប់ផ្តើម | ទិដ្ឋភាពទូទៅ · ការដំឡើងបរិស្ថាន · ការទាញយកកូដ |
| ជំពូកស្នូល | ជំពូក ១–៧ (Understanding LLMs, Text Data, Attention, GPT Model, Pretraining, Classification, Instructions) — ទំព័រនីមួយៗមាន **មេរៀនពេញលេញ** ជាខ្មែរ មិនមែនគ្រាន់តែសង្ខេបទេ |
| ឯកសារយោង | ភាគបន្ថែម A–E · សម្ភារៈបន្ថែម · Notebooks ខ្មែរ · វចនានុក្រម · ឯកសារយោង |
| Notebooks | ថត `ណូតប៊ុក-ខ្មែរ/` — companion notebooks ១៩ (កូដដូច upstream ១០០% + comments កូដជាខ្មែរ + ការពន្យល់ខ្មែរដើម) |
| សន្ទស្សន៍កូដ | ថត `កូដ/ch01` – `កូដ/ch07` — ឯកសារ `README.md` ជាខ្មែរសម្រាប់ជំពូកនីមួយៗ (បញ្ជីថតដែលមានក្នុងឃ្លាំង និងតំណទៅ upstream សម្រាប់ថតដែលមិនបានចម្លង) |

## លក្ខណៈពិសេស

- ខ្លឹមសារទាំងស្រុងជាភាសាខ្មែរ (រក្សាពាក្យបច្ចេកទេសអង់គ្លេសដដែល) ដោយប្រើពុម្ពអក្សរ Noto Sans Khmer + Noto Serif Khmer (fallback Khmer OS)
- រូបរាងបែប Read the Docs (Sphinx + sphinx_rtd_theme) ស្អាត សាមញ្ញ គ្មានភាពរញ៉េរញ៉ៃ
- ~~គាំទ្រទាំងស្បែកងងឹត (default) និងស្បែកភ្លឺ ដោយស្វ័យប្រវត្តិតាមការកំណត់ពណ៌របស់ប្រព័ន្ធ/ប្រៅស៊ឺ (prefers-color-scheme) ដោយគ្មានប៊ូតុងរំខានភ្នែក~~
- ឈ្មោះថត និងឈ្មោះឯកសារជាភាសាខ្មែរ
- តំណ Edit on GitHub លើគ្រប់ទំព័រ · sitemap.xml · robots.txt · Open Graph/Twitter/JSON-LD
- ការ build ឆ្លងកាត់ `sphinx-build -W --keep-going` ដោយគ្មាន warning

## រចនាសម្ព័ន្ធឃ្លាំង

```
source/           ប្រភពឯកសារ RST (conf.py, index.rst, _templates/, _static/, _extra/)
  ការចាប់ផ្តើម/   ទំព័រចាប់ផ្តើម
  ជំពូក/         ទំព័រជំពូក ១–៧
  យោង/           ភាគបន្ថែម សម្ភារៈបន្ថែម វចនានុក្រម ឯកសារយោង
កូដ/              ចម្លងកូដពី upstream (Apache-2.0) + README.md សន្ទស្សន៍ខ្មែរសម្រាប់ជំពូកនីមួយៗ
ណូតប៊ុក-ខ្មែរ/       companion notebooks ជាភាសាខ្មែរ
docs/             HTML ដែល build រួច (បម្រើតាម GitHub Pages)
Makefile          បញ្ជា build
requirements.txt  dependencies នៃឧបករណ៍ឯកសារ
.readthedocs.yaml ការកំណត់រចនាសម្ព័ន្ធ Read the Docs
```

## ការ build

```bash
pip install -r requirements.txt
make html          # លទ្ធផលនៅ docs/html ឬ
sphinx-build -W --keep-going source docs
```

## ការដាក់ពង្រាយ (Deploy)

- GitHub Pages: សាខា `main` ថត `/docs` (មាន `.nojekyll`)
- Read the Docs: កំពុងដំណើរការនៅ https://llms-from-scratch.readthedocs.io/
  (`.readthedocs.yaml`: ubuntu-24.04, Python 3.12, `source/conf.py`)

## អាជ្ញាបណ្ណ

- ខ្លឹមសារ/ការបកប្រែ: [CC BY 4.0](LICENSE) — មានកត់ត្រាកម្មសិទ្ធិ upstream
- កូដ/theme/config: [MIT](LICENSE-CODE)
- Upstream: កូដ Apache-2.0; អត្ថបទ និងរូបភាពសៀវភៅជាកម្មសិទ្ធិរបស់ Manning
  (ដកចេញពីវិសាលភាព Apache) — មិនត្រូវបានចម្លងមកទីនេះទេ

## ការចូលរួម

សូមបើក issue ឬ pull request។ គោលការណ៍៖ ព័ត៌មានពិតប៉ុណ្ណោះ (ផ្ទៀងផ្ទាត់ជាមួយ upstream)
រក្សាពាក្យបច្ចេកទេសអង់គ្លេសដដែល និង commit ជា conventional commits ភាសាអង់គ្លេស។
