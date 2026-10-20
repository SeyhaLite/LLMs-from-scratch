.. meta::
   :description: ជំពូក ៣ ការសរសេរកូដ Attention Mechanisms — self-attention, multi-head attention និងការអនុវត្តប្រកបដោយប្រសិទ្ធភាព។ Chapter 3 Coding Attention Mechanisms — self-attention, multi-head attention and efficient implementations.
   :keywords: ជំពូក ៣, Coding Attention Mechanisms, attention, multi-head attention, buffers, PyTorch

ជំពូក ៣៖ ការសរសេរកូដ Attention Mechanisms
=================================================

.. note::

   ប្រភព៖ ថត `ch03 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch03>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះណែនាំការសរសេរកូដយន្តការ Attention — សមាសភាគស្នូលដែលធ្វើឱ្យ LLM
អាចយកចិត្តទុកដាក់លើផ្នែកពាក់ព័ន្ធនៃទិន្នន័យបញ្ចូល រួមទាំងការអនុវត្ត Multi-Head Attention។

មេរៀនពេញលេញ (សិក្សាងាយ)
---------------------------

ការពន្យល់ខាងក្រោមជា **អត្ថបទដើមរបស់សហគមន៍ខ្មែរ** គ្របដណ្តប់គំនិតទាំងអស់នៃជំពូក ៣។
ជំពូកនេះជាចំណុចកណ្តាលនៃគម្រោងទាំងមូល — បើអ្នកយល់ attention ច្បាស់
ជំពូក ៤ ដល់ ៧ នឹងងាយស្រួលជាងច្រើន។

១. ហេតុអ្វីត្រូវការ attention?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

បញ្ហានៃភាសាគឺ **បរិបទ (context)**។ ន័យនៃពាក្យមួយអាស្រ័យលើពាក្យជុំវិញវា។

ឧទាហរណ៍៖ ក្នុងឃ្លា *"ខ្ញុំទិញកុំព្យូទ័រថ្មី ព្រោះវាលឿនណាស់"* — ពាក្យ **"វា"** សំដៅលើ
"កុំព្យូទ័រ" មិនមែន "ខ្ញុំ" ទេ។ ម៉ូដែលត្រូវតែអាចភ្ជាប់ទំនាក់ទំនងនេះបាន។

**Attention** អនុញ្ញាតឱ្យ token នីមួយៗ "សួរ" token ផ្សេងទៀតថា "តើអ្នកសំខាន់ប៉ុណ្ណា
សម្រាប់ខ្ញុំ?" រួចយកចម្លើយមកធ្វើជាទម្ងន់។

២. ជំហានទី ១ — Simplified self-attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

កំណែសាមញ្ញបំផុត៖ គណនាតំណាងថ្មីនៃ token នីមួយៗជា **មធ្យមភាគដែលមានទម្ងន់**
នៃ token ទាំងអស់ក្នុង sequence។

.. code-block:: text

   score(i,j)  = x_i . x_j          (dot product)
   weight(i,j) = softmax_j(score(i,j))
   z_i         = sum_j  weight(i,j) * x_j

កំណែនេះ **គ្មាន parameter ដែលត្រូវរៀន** ទេ — វាជាគណិតវិទ្យាសុទ្ធ។
វាមានប្រយោជន៍សម្រាប់យល់ពីមេកានិច ប៉ុន្តែមិនអាចបណ្តុះបណ្តាលបានទេ។

៣. ជំហានទី ២ — Self-attention ជាមួយទម្ងន់ដែលរៀនបាន
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

កំណែពិតប្រាកដណែនាំវ៉ិចទ័របីប្រភេទ ដែលកើតចេញពីការគុណ input ជាមួយម៉ាទ្រីសទម្ងន់៖

.. list-table::
   :header-rows: 1
   :widths: 18 15 67

   * - ឈ្មោះ
     - ម៉ាទ្រីស
     - តួនាទី (ការប្រៀបធៀបងាយយល់)
   * - **Query** (Q)
     - ``W_q``
     - "ខ្ញុំកំពុងស្វែងរកអ្វី?" — សំណួរដែល token នេះសួរ
   * - **Key** (K)
     - ``W_k``
     - "ខ្ញុំមានអ្វី?" — ស្លាកដែល token នីមួយៗផ្សព្វផ្សាយ
   * - **Value** (V)
     - ``W_v``
     - "ខ្ញុំផ្តល់ព័ត៌មានអ្វី?" — ខ្លឹមសារជាក់ស្តែងដែលត្រូវបញ្ជូន

រូបមន្ត attention ពេញលេញ (Scaled Dot-Product Attention)៖

.. code-block:: text

   Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V

**ហេតុអ្វីត្រូវចែកនឹង sqrt(d_k)?**

នៅពេល ``d_k`` (វិមាត្រនៃ key) ធំ dot product ក៏ធំទៅតាម។ តម្លៃធំៗនេះរុញ softmax ទៅក្នុង
តំបន់ដែល **gradient តូចខ្លាំង** (saturation) ធ្វើឱ្យការរៀនយឺត ឬជាប់គាំង។
ការចែកនឹង ``sqrt(d_k)`` ធ្វើឱ្យ variance នៃ score នៅប្រហែល ១ ដូច្នេះ gradient មានសុខភាពល្អ។

៤. ជំហានទី ៣ — Causal attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

សម្រាប់ការបង្កើតអត្ថបទ ម៉ូដែល **មិនត្រូវបានអនុញ្ញាតឱ្យមើលឃើញ token នៅអនាគត** ទេ —
បើមិនដូច្នោះទេវានឹង "លួចមើលចម្លើយ" អំឡុងពេលបណ្តុះបណ្តាល ហើយបរាជ័យពេលប្រើជាក់ស្តែង។

ដំណោះស្រាយគឺ **causal mask** (ឬ triangular mask)៖

- បង្កើតម៉ាទ្រីសការេដែលមានតម្លៃ ១ នៅត្រង់ និងខាងក្រោមអង្កត់ទ្រូង ហើយ ០ នៅលើអង្កត់ទ្រូង។
- កន្លែងដែល mask = ០ ជំនួស attention score ដោយ ``-inf``។
- បន្ទាប់ពី softmax តម្លៃ ``-inf`` ក្លាយជា **០** ដូច្នេះ token អនាគតមិនមានឥទ្ធិពលទេ។

.. code-block:: text

   mask =            scores (បន្ទាប់ពី mask)
   1 0 0 0           s00 -inf -inf -inf
   1 1 0 0    ->     s10  s11  -inf -inf
   1 1 1 0           s20  s21   s22  -inf
   1 1 1 1           s30  s31   s32   s33

ចំណាំជាក់ស្តែង៖ ក្នុង PyTorch គេច្រើនប្រើ ``torch.triu`` ដើម្បីបង្កើតម៉ាទ្រីសត្រីកោណ។
សម័យថ្មីៗនេះ ``F.scaled_dot_product_attention`` មានប៉ារ៉ាម៉ែត្រ ``is_causal=True``
ដែលធ្វើការងារនេះដោយស្វ័យប្រវត្តិ និងលឿនជាង។

៥. Multi-head attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**គំនិត៖** ជំនួសឱ្យការមាន attention តែមួយ យើងបង្កើត "ក្បាល" (head) ច្រើនដែលដំណើរការ
ស្របគ្នា ហើយរៀន **ប្រភេទទំនាក់ទំនងផ្សេងៗគ្នា**។ ក្បាលខ្លះអាចផ្តោតលើទំនាក់ទំនងវេយយាករណ៍
ខ្លះផ្តោតលើការយោង (coreference) ខ្លះផ្តោតលើពាក្យជិតខាង។

ដំណើរការ៖

1. បំបែកវ៉ិចទ័រ embedding ទំហំ ``d_model`` ជា ``n_heads`` ផ្នែក ដែលនីមួយៗមានវិមាត្រ
   ``d_k = d_model / n_heads``។
2. ដំណើរការ attention ដាច់ដោយឡែកលើក្បាលនីមួយៗ។
3. **Concatenate** លទ្ធផលទាំងអស់ចូលគ្នាវិញ។
4. គុណជាមួយម៉ាទ្រីស output projection ``W_o`` ដើម្បីលាយព័ត៌មានពីគ្រប់ក្បាល។

.. important::

   ចំណុចដែលច្រើនតែច្រឡំ៖ multi-head attention **មិនបង្កើនចំនួនការគណនា** ទេ បើប្រៀបធៀបនឹង
   single-head ដែលមានវិមាត្រដូចគ្នា — វាគ្រាន់តែ **បែងចែក** វិមាត្រដដែលទៅជាផ្នែកតូចៗជាច្រើន។
   អត្ថប្រយោជន៍គឺភាពចម្រុះនៃទំនាក់ទំនងដែលរៀនបាន។

៦. PyTorch buffers
~~~~~~~~~~~~~~~~~~~~~~

Causal mask គឺជា tensor ដែល **មិនមែនជា parameter** — វាមិនត្រូវបានកែប្រែដោយ gradient ទេ
ប៉ុន្តែវានៅតែត្រូវការធ្វើដំណើរជាមួយម៉ូដែលទៅ GPU ហើយត្រូវរក្សាក្នុង ``state_dict``។

ដំណោះស្រាយគឺ **register as buffer**៖ ``self.register_buffer('mask', tensor)``។
Buffer ខុសពី ``nn.Parameter`` ត្រង់ថាវាមិនស្ថិតក្នុងបញ្ជីដែល optimizer ធ្វើបច្ចុប្បន្នភាពទេ។

៧. ប្រសិទ្ធភាព — កំណែដែលអនុវត្តលឿនជាង
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ការអនុវត្ត attention ដោយប្រើ loop ឬការបង្កើតម៉ាទ្រីសធំៗគឺយឺត និងស៊ីអង្គចងចាំ។
ថត bonus ``02_bonus_efficient-multihead-attention`` ប្រៀបធៀបកំណែផ្សេងៗ៖

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - កំណែ
     - លក្ខណៈ
   * - Loop លើ head នីមួយៗ
     - សាមញ្ញបំផុត យឺតបំផុត
   * - Batched matrix multiplication
     - ប្រើ ``bmm`` ដំណើរការគ្រប់ head ក្នុងពេលតែមួយ
   * - ``F.scaled_dot_product_attention``
     - ប្រើ FlashAttention ខាងក្នុង — លឿនបំផុត និងសន្សំអង្គចងចាំបំផុត

៨. សំណួរពិនិត្យការយល់ដឹង
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. តើ Q, K, V ខុសគ្នាត្រង់ណា ហើយហេតុអ្វីត្រូវមានបីជំនួសមួយ?
2. បើលុបការចែកនឹង ``sqrt(d_k)`` តើអ្វីនឹងកើតឡើងចំពោះ gradient?
3. ហេតុអ្វី causal mask ត្រូវប្រើ ``-inf`` ជាជាង ``0`` មុនពេល softmax?
4. តើ ``n_heads=12`` និង ``d_model=768`` ផ្តល់ ``d_k`` ប៉ុន្មាន?
5. តើ buffer ខុសពី parameter ត្រង់ណា?

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch03/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch03/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch03.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-យន្តការ-attention.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `multihead-attention.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-សង្ខេប-multihead-attention.ipynb>`_
     - សង្ខេប (summary) នៃ multi-head attention
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៣-យន្តការ-attention.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-យន្តការ-attention.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៣-សង្ខេប-multihead-attention.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-សង្ខេប-multihead-attention.ipynb>`_ — សង្ខេប multi-head attention + data loader
- `ជំពូក៣-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៣-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

សូមមើលបញ្ជីពេញលេញ និងរបៀបដំណើរការក្នុងទំព័រ :doc:`../យោង/ណូតប៊ុក-ខ្មែរ`។

រចនាសម្ព័ន្ធថត
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ថត
     - ខ្លឹមសារ
   * - ``01_main-chapter-code``
     - កូដមេរបស់ជំពូក
   * - ``02_bonus_efficient-multihead-attention``
     - ប្រៀបធៀបការអនុវត្ត Multi-Head Attention ដែលមានប្រសិទ្ធភាព (``mha-implementations.ipynb``)
   * - ``03_understanding-buffers``
     - ស្វែងយល់អំពី PyTorch Buffers (``understanding-buffers.ipynb``)

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ យកចំណេះដឹង Attention ទៅសាងសង់ម៉ូដែល GPT ពេញលេញក្នុង
   :doc:`ជំពូក៤-ម៉ូដែល-GPT`។
