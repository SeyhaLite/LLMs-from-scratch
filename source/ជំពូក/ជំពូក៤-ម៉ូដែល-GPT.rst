.. meta::
   :description: ជំពូក ៤ ការអនុវត្តម៉ូដែល GPT ពីមូលដ្ឋាន — ស្ថាបត្យកម្មម៉ូដែល, KV cache, GQA, MLA, SWA, MoE។ Chapter 4 Implementing a GPT Model from Scratch — architecture, KV cache, GQA, MLA, SWA, MoE.
   :keywords: ជំពូក ៤, GPT, from scratch, gpt.py, KV cache, GQA, MLA, SWA, MoE, DeltaNet, DSA, FLOPs

ជំពូក ៤៖ ការអនុវត្តម៉ូដែល GPT ពីមូលដ្ឋាន
====================================================

.. note::

   ប្រភព៖ ថត `ch04 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch04>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះជាជំហានសំខាន់ — ការផ្គុំបំណែកទាំងអស់ចេញជាម៉ូដែល GPT ពេញលេញ
ដែលអនុវត្តពីមូលដ្ឋានដោយប្រើ PyTorch។ ក្រៅពីកូដមេ ជំពូកនេះមានសម្ភារៈបន្ថែម
ច្រើនជាងគេអំពីជម្រើសស្ថាបត្យកម្ម Attention និងការវិភាគប្រសិទ្ធភាព។

មេរៀនពេញលេញ (សិក្សាងាយ)
---------------------------

ការពន្យល់ខាងក្រោមជា **អត្ថបទដើមរបស់សហគមន៍ខ្មែរ** គ្របដណ្តប់គំនិតទាំងអស់នៃជំពូក ៤។
ជំពូកនេះយក multi-head attention ពីជំពូក ៣ មកផ្គុំជាម៉ូដែល GPT ពេញលេញ។

១. ផែនទីស្ថាបត្យកម្មទាំងមូល
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

មុនពេលចូលដល់លម្អិត សូមមើលរូបភាពធំនៃការផ្គុំ៖

.. code-block:: text

   Token IDs
      |
      v
   [ Token Embedding ] + [ Positional Embedding ]
      |
      v
   [ Dropout ]
      |
      v
   +--------------------------------------+
   |  TransformerBlock  x N layers        |
   |  +--------------------------------+  |
   |  | LayerNorm                      |  |
   |  | Multi-Head Causal Attention    |  |
   |  | Dropout                        |  |
   |  | + shortcut (residual) --------+|  |
   |  | LayerNorm                      |  |
   |  | FeedForward (Linear-GELU-Linear)|  |
   |  | Dropout                        |  |
   |  | + shortcut (residual) --------+|  |
   +--------------------------------------+
      |
      v
   [ Final LayerNorm ]
      |
      v
   [ Linear head -> vocab_size ]  (logits)

២. LayerNorm — ស្ថិរភាពនៃការបណ្តុះបណ្តាល
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**បញ្ហា៖** នៅពេលទិន្នន័យឆ្លងកាត់ស្រទាប់ច្រើន ការចែកចាយ (distribution) នៃតម្លៃអាចរសាត់
(mean ឃ្លាតពី ០ variance ប្រែប្រួល) ធ្វើឱ្យការបណ្តុះបណ្តាលមិនមានស្ថិរភាព។

**ដំណោះស្រាយ៖** LayerNorm ធ្វើឱ្យវ៉ិចទ័រនីមួយៗមាន mean ≈ ០ និង variance ≈ ១
រួចគុណ/បូកជាមួយ parameter ដែលរៀនបាន (``gamma`` scale និង ``beta`` shift)។

.. code-block:: text

   y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

ចំណាំ៖ LayerNorm ធ្វើ normalization **តាមលក្ខណៈនៃ sample នីមួយៗ** (across features)
ខុសពី BatchNorm ដែលធ្វើតាម batch។ នេះសំខាន់ណាស់សម្រាប់ NLP ព្រោះប្រវែង sequence
អាចខុសគ្នា និង batch អាចតូច។

៣. GELU — activation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPT ប្រើ **GELU** (Gaussian Error Linear Unit) ជំនួស ReLU៖

.. code-block:: text

   GELU(x) = x * Phi(x)      ដែល Phi = cumulative distribution នៃ Gaussian ធម្មតា

ភាពខុសគ្នាសំខាន់ពី ReLU៖ GELU **មិនកាត់តម្លៃអវិជ្ជមានទាំងស្រុងទេ** — វាអនុញ្ញាតឱ្យ
តម្លៃអវិជ្ជមានតូចៗឆ្លងកាត់បាន ដែលធ្វើឱ្យ gradient រលូនជាង និងជួយការបណ្តុះបណ្តាល។

៤. FeedForward network
~~~~~~~~~~~~~~~~~~~~~~~~~~

បន្ទាប់ពី attention ម៉ូដែលត្រូវការស្រទាប់ដែល **ដំណើរការព័ត៌មានលើ token នីមួយៗដាច់ដោយឡែក**។
នោះជាតួនាទីរបស់ FeedForward៖

.. code-block:: text

   x -> Linear(emb_dim -> 4 * emb_dim) -> GELU -> Linear(4 * emb_dim -> emb_dim) -> Dropout

ចំណុចគួរចាំ៖ កត្តាពង្រីក **4x** គឺជាទម្លាប់ស្តង់ដារ។ ស្រទាប់កណ្តាលធំជាង អនុញ្ញាតឱ្យ
ម៉ូដែលរៀនអនុគមន៍មិនលីនេអ៊ែរស្មុគស្មាញ។ FeedForward មាន parameter ច្រើនជាង attention
ក្នុង Transformer block មួយ។

៥. Shortcut connections (residual connections)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**បញ្ហា៖** បណ្តាញជ្រៅខ្លាំងធ្វើឱ្យ gradient រលាយបាត់ (vanishing gradient) ពេល backpropagation។

**ដំណោះស្រាយ៖** បូក input ដើមចូលទៅក្នុង output នៃស្រទាប់រង៖

.. code-block:: text

   output = x + Sublayer(Norm(x))

អត្ថប្រយោជន៍៖

- gradient អាចហូរត្រឡប់តាម "ផ្លូវកាត់" ដោយផ្ទាល់ ដូច្នេះម៉ូដែលជ្រៅនៅតែរៀនបាន។
- ស្រទាប់នីមួយៗគ្រាន់តែរៀន **ផលសង (delta)** លើ input មិនមែនរៀនតំណាងទាំងស្រុងទេ។

៦. TransformerBlock — ឯកតាដែលផ្គុំឡើងវិញបាន
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TransformerBlock គឺជាឯកតាដែលយកអ្វីៗខាងលើមកផ្គុំគ្នា ហើយត្រូវបាន **ដាក់ជួរគ្នា N ដង**។

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ធាតុផ្សំ
     - តួនាទី
   * - Multi-head causal attention
     - លាយព័ត៌មានរវាង token (ការទំនាក់ទំនង)
   * - FeedForward
     - ដំណើរការព័ត៌មានលើ token នីមួយៗដាច់ដោយឡែក
   * - LayerNorm x 2
     - ស្ថិរភាពនៃការបណ្តុះបណ្តាល
   * - Shortcut x 2
     - អនុញ្ញាតឱ្យ gradient ហូរបានល្អ

សម្រាប់ **GPT-2 small (124M)** ការកំណត់រចនាសម្ព័ន្ធគឺ៖ ``n_layers=12``, ``emb_dim=768``,
``n_heads=12``, ``context_length=1024``, ``vocab_size=50257``។

៧. ការគណនាទំហំ attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

អ្នកត្រូវចេះប៉ាន់ប្រមាណការគណនា និងអង្គចងចាំ៖

.. code-block:: text

   ចំនួន parameter attention ក្នុង 1 block ~ 4 * emb_dim^2
     (W_q, W_k, W_v, W_o)

   attention matrix មានទំហំ  context_length x context_length
     -> ការគណនា attention កើនជា O(n^2) ធៀបនឹងប្រវែង sequence

   អង្គចងចាំ KV cache ~ 2 * n_layers * n_heads * d_k * seq_len * batch * 4 bytes (float32)

នេះជាមូលហេតុដែល **context length វែង** ថ្លៃខ្លាំង និងជាមូលហេតុដែលមានស្ថាបត្យកម្ម
attention ជំនួស (GQA, MLA, SWA, DeltaNet) ក្នុងថត bonus នៃជំពូកនេះ។

៨. ការបង្កើតអត្ថបទ (text generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ម៉ូដែលដែលផ្គុំរួចបញ្ចេញ **logits** (ពិន្ទុឆៅសម្រាប់ token នីមួយៗក្នុង vocabulary)។
ដើម្បីបង្កើតអត្ថបទ យើងធ្វើរង្វិលជុំ autoregressive៖

1. ដាក់ prompt ចូលម៉ូដែល → ទទួល logits សម្រាប់ទីតាំងចុងក្រោយ។
2. បំលែង logits ទៅជាប្រូបាបដោយ softmax។
3. ជ្រើសរើស token មួយ (តាមយុទ្ធសាស្ត្រ decoding)។
4. បន្ថែម token នោះទៅ input រួចធ្វើម្តងទៀតពីជំហាន ១។

យុទ្ធសាស្ត្រជ្រើសរើស token៖

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - យុទ្ធសាស្ត្រ
     - លក្ខណៈ
   * - Greedy
     - យក token ដែលមានប្រូបាបខ្ពស់បំផុតជានិច្ច — ងាយជាប់គាំងក្នុងរង្វិលដដែលៗ
   * - **Temperature scaling**
     - ចែក logits នឹង ``T``។ ``T < 1`` ធ្វើឱ្យការចែកចាយមុតស្រួច (អត្ថបទច្បាស់លាស់ ប៉ុន្តែអាចជាន់ដដែល)។ ``T > 1`` ធ្វើឱ្យរាបស្មើ (ច្នៃប្រឌិត ប៉ុន្តែអាចមិនសមហេតុផល)
   * - **Top-k**
     - រក្សាតែ token ដែលមានប្រូបាបខ្ពស់បំផុត k ដំបូង រួចលុបផ្សេងទៀតចោល មុនពេលជ្រើសរើស។ ការពារម៉ូដែលមិនឱ្យជ្រើសពាក្យគ្មានន័យ

៩. KV cache — បង្កើនល្បឿន inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**បញ្ហា៖** នៅជំហានបង្កើត token ទី ``t`` យើងគណនា K និង V សម្រាប់ token ទាំងអស់ពី ១ ដល់ ``t``
ឡើងវិញ ទាំងដែល K/V នៃ token មុនៗ **មិនប្រែប្រួល** ទេ។

**ដំណោះស្រាយ៖** រក្សាទុក K និង V ដែលគណនារួច ហើយគណនាតែ token ថ្មីមួយប៉ុណ្ណោះ។

.. code-block:: text

   គ្មាន KV cache:  O(t^2) ការគណនាសរុបសម្រាប់បង្កើត t tokens
   មាន KV cache:  O(t)   ប៉ុន្តែត្រូវចំណាយអង្គចងចាំរក្សាទុក K, V

នេះជាអ្វីដែលថត ``03_kv-cache`` អនុវត្ត។ ការផ្លាស់ប្តូរនេះធ្វើឱ្យ inference លឿនជាងច្រើន
ប៉ុន្តែត្រូវការអង្គចងចាំបន្ថែម — នេះជា trade-off ដ៏សំខាន់បំផុតមួយក្នុងការបម្រើ LLM។

១០. សម្ភារៈបន្ថែមក្នុងជំពូកនេះ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ថត
     - អ្វីដែលអ្នកនឹងរៀន
   * - ``02_performance-analysis``
     - របៀបវាស់ FLOPs និងពេលវេលាជាក់ស្តែងនៃម៉ូដែល
   * - ``03_kv-cache``
     - អនុវត្ត KV cache ដោយផ្ទាល់
   * - ``04_gqa``
     - Grouped-Query Attention — កាត់បន្ថយអង្គចងចាំ KV cache ដោយចែករំលែក K/V រវាងក្បាល
   * - ``05_mla``
     - Multi-Head Latent Attention — បង្ហាប់ latent vector ជំនួសឱ្យការរក្សា KV ទាំងស្រុង
   * - ``06_swa``
     - Sliding Window Attention — កំណត់ attention ត្រឹមបង្អួចជិតខាង
   * - ``07_moe``
     - Mixture-of-Experts — មាន expert ច្រើន ប៉ុន្តែធ្វើឱ្យសកម្មតែមួយចំនួនក្នុងមួយ token
   * - ``08_deltanet``
     - Gated DeltaNet — វ៉ារ្យ៉ង់ linear attention ដែលជៀសវាង O(n^2)
   * - ``10_kv-sharing``
     - ចែករំលែក KV រវាងស្រទាប់ ដើម្បីកាត់បន្ថយអង្គចងចាំ

១១. សំណួរពិនិត្យការយល់ដឹង
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ហេតុអ្វី GPT ប្រើ LayerNorm ជំនួស BatchNorm?
2. តើ shortcut connection ដោះស្រាយបញ្ហាអ្វីពិតប្រាកដ?
3. បើ ``emb_dim=768`` តើ FeedForward មានទំហំស្រទាប់កណ្តាលប៉ុន្មាន?
4. តើ temperature = 0.5 និង temperature = 1.5 ផ្តល់អត្ថបទខុសគ្នាយ៉ាងដូចម្តេច?
5. ហេតុអ្វី KV cache ជួយបានតែពេល inference មិនមែនពេល training?

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch04/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch04/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch04.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ម៉ូដែល-GPT.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch04/01_main-chapter-code/gpt.py>`_
     - សង្ខេប (summary) ម៉ូដែល GPT ជាឯកសារ Python
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៤-ម៉ូដែល-GPT.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ម៉ូដែល-GPT.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៤-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៤-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

សម្ភារៈបន្ថែម និងជម្រើស Attention
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ថត
     - ខ្លឹមសារ
   * - ``02_performance-analysis``
     - ការវិភាគ FLOPs (``flops-analysis.ipynb``)
   * - ``03_kv-cache``
     - KV Cache
   * - ``04_gqa``
     - Grouped-Query Attention (GQA)
   * - ``05_mla``
     - Multi-Head Latent Attention (MLA)
   * - ``06_swa``
     - Sliding Window Attention (SWA)
   * - ``07_moe``
     - Mixture-of-Experts (MoE)
   * - ``08_deltanet``
     - Gated DeltaNet
   * - ``09_dsa``
     - DeepSeek Sparse Attention (DSA)
   * - ``10_kv-sharing``
     - Cross-Layer KV Sharing

.. note::

   បញ្ជីខាងលើឆ្លុះបញ្ចាំងពីរចនាសម្ព័ន្ធថតពិត និងបញ្ជីសម្ភារៈបន្ថែមក្នុង ``README.md``
   របស់ upstream។ លម្អិតបច្ចេកទេសនៃវិធីសាស្ត្រនីមួយៗ សូមអានក្នុង notebooks
   ផ្លូវការនៅឃ្លាំង upstream។

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ បណ្តុះបណ្តាលម៉ូដែលឱ្យដំណើរការពិតប្រាកដក្នុង
   :doc:`ជំពូក៥-pretraining`។
