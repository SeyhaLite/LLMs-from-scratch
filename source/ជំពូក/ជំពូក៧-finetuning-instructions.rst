.. meta::
   :description: ជំពូក ៧ Finetuning ដើម្បីធ្វើតាមការណែនាំ — instruction dataset, Ollama, DPO និង preference tuning។ Chapter 7 Finetuning to Follow Instructions — instruction dataset, Ollama, DPO and preference tuning.
   :keywords: ជំពូក ៧, instruction finetuning, ollama_evaluate.py, Ollama, DPO, preference tuning, dataset generation, Llama 3.1

ជំពូក ៧៖ Finetuning ដើម្បីធ្វើតាមការណែនាំ
====================================================

.. note::

   ប្រភព៖ ថត `ch07 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch07>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកចុងក្រោយនៃជំពូកមេ — ការ finetune ម៉ូដែល GPT ឱ្យធ្វើតាមការណែនាំ
(instruction following) ដូចជាជំនួយការសន្ទនា។ បណ្ណាល័យដែល ``requirements.txt``
កំណត់សម្រាប់ជំពូកនេះ៖ ``matplotlib`` · ``tensorflow`` · ``tqdm`` · ``psutil``។

មេរៀនពេញលេញ (សិក្សាងាយ)
---------------------------

ការពន្យល់ខាងក្រោមជា **អត្ថបទដើមរបស់សហគមន៍ខ្មែរ** គ្របដណ្តប់គំនិតទាំងអស់នៃជំពូក ៧។
នេះជាជំពូកចុងក្រោយនៃផ្នែកស្នូល — វាបំលែង **base model** ទៅជា **assistant**
ដែលឆ្លើយតបតាមបទបញ្ជា។

១. ភាពខុសគ្នារវាង base model និង instruction model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ប្រភេទ
     - ឥរិយាបថ
   * - **Base model**
     - បន្តអត្ថបទ។ បើសួរ *"តើអ្នកជាអ្នកណា?"* វាអាចឆ្លើយ *"តើអ្នកជាអ្នកណា? តើខ្ញុំជាអ្នកណា?"* — ព្រោះវារៀនតែពីការបន្តអត្ថបទ
   * - **Instruction-tuned model**
     - ឆ្លើយតប។ សួរដូចគ្នា វាឆ្លើយ *"ខ្ញុំជាជំនួយការ AI"*

ការ finetune នេះហៅថា **Supervised Finetuning (SFT)** ឬ **instruction finetuning**។

២. ទម្រង់ទិន្នន័យ instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ឧទាហរណ៍នីមួយៗមានផ្នែកបី (ទម្រង់ Alpaca-style)៖

.. code-block:: text

   {
     "instruction": "បំលែងសីតុណ្ហភាព ២៥ អង្សាសេ ទៅជា Fahrenheit",
     "input": "",                       # បរិបទបន្ថែម (អាចទទេ)
     "output": "២៥ អង្សាសេ ស្មើនឹង ៧៧ អង្សា Fahrenheit"
   }

ថត ``02_dataset-utilities`` ផ្តល់កូដជំនួយសម្រាប់រៀបចំ dataset បែបនេះ
ហើយ ``05_dataset-generation`` បង្កើត និងកែលម្អ **synthetic dataset** ដោយប្រើ LLM ផ្ទាល់។

៣. Prompt formatting — ការបំលែងទៅជាអត្ថបទតែមួយ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LLM ទទួលតែអត្ថបទ ដូច្នេះយើងត្រូវវេចខ្ចប់រចនាសម្ព័ន្ធជាមួយ **special tokens**៖

.. code-block:: text

   <|user|>
   បំលែងសីតុណ្ហភាព ២៥ អង្សាសេ ទៅជា Fahrenheit
   <|assistant|>
   ២៥ អង្សាសេ ស្មើនឹង ៧៧ អង្សា Fahrenheit
   <|endoftext|>

ហេតុអ្វីត្រូវការ tokens ពិសេសទាំងនេះ? ព្រោះពេល inference យើងឈប់បង្កើតអត្ថបទ
នៅពេលម៉ូដែលបញ្ចេញ ``<|endoftext|>`` ហើយយើងដឹងថាត្រូវចាប់ផ្តើមអានចម្លើយ
នៅកន្លែងណាដោយមើលរក ``<|assistant|>``។

.. note::

   Token ពិសេសទាំងនេះ **មិនមាន** ក្នុង vocabulary ដើមរបស់ GPT-2 ទេ។
   អ្នកត្រូវបន្ថែមវាចូល tokenizer រួចពង្រីក embedding table ឱ្យធំតាម។

៤. Instruction masking — ចំណុចសំខាន់បំផុតនៃជំពូកនេះ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**បញ្ហា៖** បើយើងគណនា loss លើ token ទាំងអស់ ម៉ូដែលនឹងចំណាយសមត្ថភាព
រៀន **ចងចាំសំណួរ** ជំនួសឱ្យរៀន **ឆ្លើយសំណួរ**។

**ដំណោះស្រាយ៖** គណនា loss តែលើ token នៃផ្នែក **output** ប៉ុណ្ណោះ។

.. code-block:: text

   tokens:      <|user|> បំលែង សីតុណ្ហ ...  <|assistant|> ២៥ ស្មើ នឹង ៧៧ ... <|endoftext|>
   loss mask:      0        0     0    0          0          1   1   1  1  1        1

ការអនុវត្តក្នុង PyTorch៖ ជំនួស label នៅកន្លែងដែលមិនចង់គណនាដោយ ``-100``
ព្រោះ ``F.cross_entropy`` មានប៉ារ៉ាម៉ែត្រ ``ignore_index=-100`` ដែលរំលងតម្លៃទាំងនោះ។

.. code-block:: text

   loss = F.cross_entropy(logits, labels, ignore_index=-100)

៥. ដំណើរការ finetuning
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ១. ផ្ទុក base model (ពីជំពូក ៥)
   ២. បន្ថែម instruction tokens ចូល tokenizer
   ៣. ពង្រីក embedding + output head ឱ្យត្រូវនឹង vocab ថ្មី
   ៤. រៀបចំ dataset ជាមួយ prompt formatting + instruction masking
   ៥. Finetune (ច្រើនតែប្រើ learning rate តូចជាង pretraining ១០ ដង)
   ៦. វាយតម្លៃលើ instruction dataset ដែលមិនធ្លាប់ឃើញ

ចំណាំសំខាន់៖ instruction finetuning ប្រើ **epoch តិច** (ច្រើនតែ ១–៣) និង learning rate តូច។
ការបណ្តុះបណ្តាលយូរពេកធ្វើឱ្យម៉ូដែល **catastrophic forgetting** — បាត់បង់ចំណេះដឹងភាសាទូទៅ។

៦. ការវាយតម្លៃ — ហេតុអ្វីពិបាកជាងចំណាត់ថ្នាក់
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

សម្រាប់ចំណាត់ថ្នាក់ ចម្លើយត្រូវ/ខុសច្បាស់លាស់។ សម្រាប់ instruction ចម្លើយអាចត្រូវ
តាមវិធីជាច្រើន។ វិធីសាស្ត្រវាយតម្លៃ៖

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - វិធីសាស្ត្រ
     - លក្ខណៈ
   * - មនុស្សវាយតម្លៃ
     - ត្រឹមត្រូវបំផុត ប៉ុន្តែយឺត និងថ្លៃ
   * - **LLM-as-a-judge**
     - ប្រើម៉ូដែលខ្លាំងជាង (GPT-4) ឬម៉ូដែលក្នុងតំបន់ (Llama 3) ដើម្បីវាយតម្លៃចម្លើយ។ ថត ``03_model-evaluation`` អនុវត្តទាំងពីរវិធី
   * - ការប្រៀបធៀបជាមួយ base model
     - វាស់ថា finetuning ជួយបានប៉ុន្មានពិតប្រាកដ

.. warning::

   **LLM-as-a-judge មានភាពលំអៀង** — ម៉ូដែលច្រើនតែផ្តល់ពិន្ទុខ្ពស់ដល់ចម្លើយវែងៗ
   និងចម្លើយដែលមានរចនាបទស្រដៀងខ្លួនឯង។ សូមប្រើវាជាឧបករណ៍ជំនួយ
   មិនមែនជាការវិនិច្ឆ័យចុងក្រោយទេ។

៧. Preference tuning ជាមួយ DPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ថត ``04_preference-tuning-with-dpo`` អនុវត្ត **Direct Preference Optimization**។

**គំនិត៖** ជំនួសឱ្យការបង្ហាញតែចម្លើយល្អ យើងបង្ហាញ **គូ**៖ ចម្លើយដែលពេញចិត្ត (chosen)
និងចម្លើយដែលមិនពេញចិត្ត (rejected) រួចបណ្តុះបណ្តាលម៉ូដែលឱ្យផ្តល់ប្រូបាបខ្ពស់ដល់ chosen។

DPO ជាជម្រើសជំនួស **RLHF** (Reinforcement Learning from Human Feedback) ដែលសាមញ្ញជាង
ព្រោះវាមិនត្រូវការ reward model ដាច់ដោយឡែក និងមិនត្រូវការការបណ្តុះបណ្តាល reinforcement learning។

៨. ការដាក់ពង្រាយ
~~~~~~~~~~~~~~~~~~~~

ថត ``06_user_interface`` អនុវត្តចំណុចប្រទាក់អន្តរកម្មសម្រាប់សន្ទនាជាមួយម៉ូដែល
ដែល finetune រួច — ជំហានចុងក្រោយដែលបិទបញ្ចប់វដ្តអភិវឌ្ឍន៍ទាំងមូល។

៩. វដ្តពេញលេញ — ពីជំពូក ១ ដល់ជំពូក ៧
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ជំពូក ១  គោលគំនិត
   ជំពូក ២  អត្ថបទ -> tokens -> tensors        (ទិន្នន័យ)
   ជំពូក ៣  attention                          (មេកានិចស្នូល)
   ជំពូក ៤  ផ្គុំជាម៉ូដែល GPT                   (ស្ថាបត្យកម្ម)
   ជំពូក ៥  pretraining + ផ្ទុក GPT-2 weights  (ចំណេះដឹងភាសា)
   ជំពូក ៦  finetune សម្រាប់ចំណាត់ថ្នាក់         (កិច្ចការជាក់លាក់)
   ជំពូក ៧  finetune សម្រាប់ instruction       (assistant)

១០. សំណួរពិនិត្យការយល់ដឹង
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ហេតុអ្វី base model មិនឆ្លើយសំណួរដោយផ្ទាល់?
2. តើ ``ignore_index=-100`` ធ្វើអ្វីក្នុង ``F.cross_entropy``?
3. ហេតុអ្វីត្រូវពង្រីក embedding table ពេលបន្ថែម special tokens?
4. តើអ្វីជា catastrophic forgetting ហើយធ្វើដូចម្តេចដើម្បីជៀសវាង?
5. តើ DPO ខុសពី RLHF ត្រង់ណា?

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch07/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch07/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch07.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-finetuning-instructions.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_instruction_finetuning.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch07/01_main-chapter-code/gpt_instruction_finetuning.py>`_
     - សង្ខេប (summary) ស្គ្រីប instruction finetuning
   * - `ollama_evaluate.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch07/01_main-chapter-code/ollama_evaluate.py>`_
     - សង្ខេប (summary) ការវាយតម្លៃតាមរយៈ Ollama
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៧-finetuning-instructions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-finetuning-instructions.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៧-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៧-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

សម្ភារៈបន្ថែម
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ថត
     - ខ្លឹមសារ
   * - ``02_dataset-utilities``
     - ឧបករណ៍ dataset សម្រាប់រក near duplicates និងបង្កើត passive voice entries
   * - ``03_model-evaluation``
     - ការវាយតម្លៃចម្លើយ instruction ដោយប្រើ OpenAI API និង Ollama
   * - ``04_preference-tuning-with-dpo``
     - Direct Preference Optimization (DPO) សម្រាប់ LLM Alignment (``dpo-from-scratch.ipynb``) និងការបង្កើត Preference Dataset ជាមួយ Llama 3.1 70B និង Ollama (``create-preference-data-ollama.ipynb``)
   * - ``05_dataset-generation``
     - ការបង្កើត dataset សម្រាប់ instruction finetuning (``llama3-ollama.ipynb``) និងការកែលម្អ dataset (``reflection-gpt4.ipynb``)
   * - ``06_user_interface``
     - ការសាងសង់ User Interface សម្រាប់ប្រើជាមួយម៉ូដែល GPT ដែលបាន instruction-finetune

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ សិក្សាភាគបន្ថែម A–E (PyTorch, LoRA ជាដើម) ក្នុងទំព័រ
   :doc:`../យោង/ភាគបន្ថែម`។
