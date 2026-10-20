.. meta::
   :description: ជំពូក ៥ Pretraining លើទិន្នន័យគ្មានស្លាក — training loop, weight loading, learning rate schedulers និងស្ថាបត្យកម្ម LLM ផ្សេងៗ។ Chapter 5 Pretraining on Unlabeled Data — training loop, weight loading, schedulers and LLM architectures.
   :keywords: ជំពូក ៥, pretraining, gpt_train.py, gpt_generate.py, Gutenberg, learning rate scheduler, Llama, Qwen3, Gemma, Olmo, Aya, hyperparameter, tiktoken

ជំពូក ៥៖ Pretraining លើទិន្នន័យគ្មានស្លាក
=================================================

.. note::

   ប្រភព៖ ថត `ch05 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch05>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះគ្របដណ្តប់ការបណ្តុះបណ្តាលមុន (pretraining) ម៉ូដែល GPT ដែលបានសាងសង់ក្នុងជំពូក ៤
លើទិន្នន័យគ្មានស្លាក (unlabeled data) រួមទាំងការបង្កើតបំណែកកូដសម្រាប់ train និង generate។
នេះជាជំពូកដែលមានថតបន្ថែមច្រើនជាងគេក្នុងឃ្លាំង — រួមទាំងការអនុវត្តស្ថាបត្យកម្ម LLM
ជំនាន់ថ្មីៗពីមូលដ្ឋាន។ បណ្ណាល័យដែល ``requirements.txt`` កំណត់សម្រាប់ជំពូកនេះ៖
``tiktoken`` · ``tensorflow`` · ``tqdm``។

មេរៀនពេញលេញ (សិក្សាងាយ)
---------------------------

ការពន្យល់ខាងក្រោមជា **អត្ថបទដើមរបស់សហគមន៍ខ្មែរ** គ្របដណ្តប់គំនិតទាំងអស់នៃជំពូក ៥។
ជំពូកនេះយកម៉ូដែលពីជំពូក ៤ មក **បណ្តុះបណ្តាលពិតប្រាកដ**។

១. ទិដ្ឋភាពទូទៅនៃដំណើរការ pretraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ១. រៀបចំទិន្នន័យ  -> ២. វាស់ loss  -> ៣. កំណត់ optimizer
        -> ៤. Training loop  -> ៥. វាយតម្លៃ  -> ៦. រក្សាទុក/ផ្ទុក weights

២. អនុគមន៍ loss — ការវាស់ថាម៉ូដែលខុសប៉ុណ្ណា
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

សម្រាប់ការព្យាករណ៍ token បន្ទាប់ យើងប្រើ **cross-entropy loss**។

.. code-block:: text

   Loss = - (1/N) * sum_i  log( p( token_ពិត_i | context_i ) )

ការបកស្រាយងាយយល់៖

- បើម៉ូដែលផ្តល់ប្រូបាបខ្ពស់ដល់ token ត្រឹមត្រូវ → ``log(p)`` ជិត ០ → loss តូច។
- បើម៉ូដែលផ្តល់ប្រូបាបតូចដល់ token ត្រឹមត្រូវ → ``log(p)`` អវិជ្ជមានខ្លាំង → loss ធំ។

**Perplexity** គឺជា exponential នៃ cross-entropy loss៖

.. code-block:: text

   perplexity = exp(loss)

ការបកស្រាយ៖ perplexity = ១០ មានន័យថាម៉ូដែល "ងងឹតងងល់" រវាងជម្រើសប្រហែល ១០។
perplexity ទាបជាង = ល្អជាង។ វាមានប្រយោជន៍ជាង loss ឆៅ ព្រោះវាងាយបកស្រាយ។

៣. Decoding strategies ក្នុងការវាយតម្លៃ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ដូចបានពន្យល់ក្នុងជំពូក ៤ — greedy, temperature scaling, និង top-k ទាំងអស់ត្រូវបាន
ប្រើក្នុងជំពូកនេះដើម្បីពិនិត្យគុណភាពអត្ថបទដែលម៉ូដែលបង្កើត អំឡុងពេលបណ្តុះបណ្តាល។

៤. Optimizer — AdamW
~~~~~~~~~~~~~~~~~~~~~~~~

**AdamW** គឺជា optimizer ស្តង់ដារសម្រាប់ Transformer។ វាផ្សំគំនិតពីរ៖

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - ធាតុ
     - តួនាទី
   * - Momentum
     - រក្សា "ល្បឿន" នៃការធ្វើបច្ចុប្បន្នភាព ដើម្បីឆ្លងកាត់រណ្តៅតូចៗ និងបន្ថយការញ័រ
   * - Adaptive learning rate
     - កែ learning rate សម្រាប់ parameter នីមួយៗដោយផ្អែកលើ gradient ពីមុន
   * - Weight decay (decoupled)
     - ការពារ overfitting ដោយរុញទម្ងន់ឱ្យតូច។ AdamW ធ្វើវា **ដាច់ពី gradient update** ដែលត្រឹមត្រូវជាង Adam ធម្មតា

ហេតុអ្វីមិនប្រើ SGD ធម្មតា? ព្រោះ Transformer មាន loss landscape ស្មុគស្មាញ
ហើយ SGD ត្រូវការការកំណត់ learning rate យ៉ាងប្រុងប្រយ័ត្នខ្លាំង។

៥. Training loop — បេះដូងនៃការបណ្តុះបណ្តាល
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

រង្វិលជុំស្តង់ដារមាន ៥ ជំហាន ដែលត្រូវចាំឱ្យបានត្រឹមត្រូវ៖

.. code-block:: text

   for epoch in range(num_epochs):
       for input_batch, target_batch in train_loader:

           optimizer.zero_grad()          # ១. សម្អាត gradient ចាស់
           logits = model(input_batch)    # ២. forward pass
           loss   = cross_entropy(logits, target_batch)   # ៣. គណនា loss
           loss.backward()                # ៤. backward pass (គណនា gradient)
           optimizer.step()               # ៥. ធ្វើបច្ចុប្បន្នភាពទម្ងន់

កំហុសដែលសិក្ខាកាមច្រើនតែធ្វើ៖

- **ភ្លេច ``zero_grad()``** → gradient កើនឡើងបន្តបន្ទាប់ → loss ផ្ទុះឡើង។
- **ហៅ ``loss.backward()`` ច្រើនដង** ដោយមិន ``retain_graph=True`` → error។
- **មិនដាក់ម៉ូដែលក្នុង ``eval()`` ពេលវាយតម្លៃ** → Dropout និង LayerNorm ឥរិយាបថខុស។
- **ភ្លេច ``torch.no_grad()`` ពេល evaluate** → ស៊ីអង្គចងចាំខ្លាំង និងយឺត។

៦. ការត្រួតពិនិត្យ overfitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ដំណើរការធម្មតាគឺត្រួតពិនិត្យ loss លើទាំង **training set** និង **validation set**៖

.. code-block:: text

   train loss ចុះ  +  val loss ចុះ     ->  ល្អ បណ្តុះបណ្តាលបន្ត
   train loss ចុះ  +  val loss ឡើង     ->  overfitting គួរបញ្ឈប់ ឬប្រើ regularization
   train loss មិនចុះ                   ->  ពិនិត្យ learning rate, data pipeline, gradient

ឧបករណ៍ទប់ស្កាត់ overfitting ក្នុងគម្រោងនេះ៖ **Dropout** និង **weight decay**។

៧. ការរក្សាទុក និងផ្ទុក weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   រក្សាទុក:  torch.save(model.state_dict(), "model.pth")
   ផ្ទុក:      model.load_state_dict(torch.load("model.pth"))

``state_dict`` គឺជា dictionary នៃ tensor ទាំងអស់ (parameter + buffer)។
ចំណាំថា buffer ក៏ត្រូវបានរក្សាទុកដែរ — នេះជាហេតុផលដែល causal mask ត្រូវចុះឈ្មោះជា buffer
(មើលជំពូក ៣)។

៨. ការផ្ទុក GPT-2 weights ពី OpenAI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ជំនួសឱ្យការបណ្តុះបណ្តាលពីសូន្យ (ដែលត្រូវការ GPU ច្រើន) គម្រោងនេះ **ផ្ទុកទម្ងន់ GPT-2 124M
ដែល OpenAI បានប្រកាសជាសាធារណៈ** ចូលទៅក្នុងស្ថាបត្យកម្មដែលយើងសរសេរដោយខ្លួនឯង។

នេះជាវិធីសាស្ត្រដ៏ល្អដើម្បីផ្ទៀងផ្ទាត់ថាកូដរបស់អ្នកត្រឹមត្រូវ៖ បើស្ថាបត្យកម្មរបស់អ្នក
ត្រូវគ្នានឹង GPT-2 ពិតប្រាកដ នោះទម្ងន់នឹងផ្ទុកបាន ហើយម៉ូដែលនឹងបង្កើតអត្ថបទមានន័យភ្លាម។

កំហុសធម្មតាពេលផ្ទុកទម្ងន់៖

- ឈ្មោះ key មិនត្រូវគ្នា → ត្រូវការការផ្គូផ្គងឈ្មោះ (name mapping)។
- ទំហំ tensor មិនត្រូវគ្នា → ស្ថាបត្យកម្មខុស ឬ config ខុស។
- ទម្រង់ ``QKV`` រួមបញ្ចូលគ្នា vs ដាច់ដោយឡែក → ត្រូវបំបែក/ផ្គុំឱ្យត្រូវ។

៩. Weight tying — ចែករំលែក embedding ជាមួយ output layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**គំនិត៖** token embedding table (``vocab_size x emb_dim``) និង output head
(``emb_dim x vocab_size``) មានទំហំដូចគ្នា ហើយតំណាងគំនិតដូចគ្នា។ ដូច្នេះយើងអាច
**ប្រើម៉ាទ្រីសតែមួយ** សម្រាប់ទាំងពីរ។

អត្ថប្រយោជន៍៖

- កាត់បន្ថយចំនួន parameter យ៉ាងច្រើន (សម្រាប់ GPT-2 small ប្រហែល ៣៨ លាន parameter)។
- ជារឿយៗជួយឱ្យម៉ូដែលរៀនបានល្អជាង ព្រោះ gradient ពីទាំងពីរផ្លូវពង្រឹងគ្នា។

១០. Pretraining លើទិន្នន័យធំជាង (Project Gutenberg)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ថត bonus ``03_bonus_pretraining_on_gutenberg`` បង្ហាញពីរបៀបបណ្តុះបណ្តាលម៉ូដែល
ឱ្យយូរជាងមុនលើ corpus សៀវភៅទាំងមូល។ ចំណុចសំខាន់ដែលអ្នកនឹងជួប៖

- ការដោះស្រាយអត្ថបទដែលធំជាងអង្គចងចាំ (ត្រូវការ streaming ឬ chunking)។
- ការរៀបចំ tokenization ជាមុន ដើម្បីកុំឱ្យ tokenize ម្តងទៀតរាល់ epoch។
- ការតាមដាន loss លើជំហានច្រើន និងការរក្សាទុក checkpoint ជាប្រចាំ។

១១. សំណួរពិនិត្យការយល់ដឹង
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. តើអ្វីជាភាពខុសគ្នារវាង cross-entropy loss និង perplexity?
2. ហេតុអ្វី AdamW ខុសពី Adam ធម្មតា?
3. តើអ្វីកើតឡើងបើអ្នកភ្លេច ``optimizer.zero_grad()``?
4. តើអ្នកដឹងថាម៉ូដែល overfit ដោយរបៀបណា ដោយមើលតែលើ loss?
5. ហេតុអ្វី weight tying ជួយកាត់បន្ថយ parameter?

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch05/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch05/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch05.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-pretraining.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_train.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch05/01_main-chapter-code/gpt_train.py>`_
     - សង្ខេប (summary) ស្គ្រីបបណ្តុះបណ្តាល
   * - `gpt_generate.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch05/01_main-chapter-code/gpt_generate.py>`_
     - សង្ខេប (summary) ស្គ្រីបបង្កើតអត្ថបទ
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៥-pretraining.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-pretraining.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៥-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៥-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

សម្ភារៈបន្ថែម
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ថត
     - ខ្លឹមសារ
   * - ``02_alternative_weight_loading``
     - វិធីសាស្ត្រផ្ទុក weights ជំនួស
   * - ``03_bonus_pretraining_on_gutenberg``
     - Pretraining GPT លើ dataset Project Gutenberg
   * - ``04_learning_rate_schedulers``
     - ការបន្ថែមមុខងារបន្ថែមទៅក្នុង training loop
   * - ``05_bonus_hparam_tuning``
     - ការកែលម្អ hyperparameters សម្រាប់ pretraining
   * - ``06_user_interface``
     - ការសាងសង់ User Interface សម្រាប់ប្រើជាមួយ LLM ដែលបាន pretrain
   * - ``07_gpt_to_llama``
     - ការបំលែង GPT ទៅជា Llama (រួមមាន ``standalone-llama32.ipynb`` សម្រាប់ Llama 3.2 From Scratch)
   * - ``08_memory_efficient_weight_loading``
     - ការផ្ទុក weights ដោយសន្សំសំចៃអង្គចងចាំ (``memory-efficient-state-dict.ipynb``)
   * - ``09_extending-tokenizers``
     - ការពង្រីក Tiktoken BPE Tokenizer ដោយ token ថ្មី (``extend-tiktoken.ipynb``)
   * - ``10_llm-training-speed``
     - គន្លឹះប្រសិទ្ធភាព PyTorch សម្រាប់បង្កើនល្បឿនការ train LLM
   * - ``11_qwen3``
     - Qwen3 Dense និង Mixture-of-Experts (MoE) From Scratch
   * - ``12_gemma3``
     - Gemma 3 From Scratch
   * - ``13_olmo3``
     - Olmo 3 From Scratch
   * - ``14_ch05_with_other_llms``
     - ជំពូក ៥ ជាមួយ LLM ផ្សេងជា Drop-In Replacement (ឧ. Llama 3, Qwen 3)
   * - ``15_tiny-aya``
     - Tiny Aya From Scratch
   * - ``16_qwen3.5``
     - Qwen3.5 From Scratch
   * - ``17_gemma4``
     - Gemma 4 E2B និង E4B From Scratch
   * - ``18_muon``
     - សម្ភារៈបន្ថែមក្នុងឃ្លាំង (មិនមានក្នុងបញ្ជី ``README.md`` របស់ upstream)
   * - ``19_cpu_mps_differences``
     - ភាពខុសគ្នារវាងឧបករណ៍ CPU និង MPS

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ ប្រើម៉ូដែលដែលបាន pretrain ដើម្បីចំណាត់ថ្នាក់អត្ថបទក្នុង
   :doc:`ជំពូក៦-finetuning-classification`។
