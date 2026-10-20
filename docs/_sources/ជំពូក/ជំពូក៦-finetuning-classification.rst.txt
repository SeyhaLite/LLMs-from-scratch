.. meta::
   :description: ជំពូក ៦ Finetuning សម្រាប់ចំណាត់ថ្នាក់អត្ថបទ — spam classifier, dataset IMDb និង UI។ Chapter 6 Finetuning for Text Classification — spam classifier, IMDb dataset and UI.
   :keywords: ជំពូក ៦, finetuning, text classification, gpt_class_finetune.py, spam classifier, IMDb, pandas, matplotlib

ជំពូក ៦៖ Finetuning សម្រាប់ចំណាត់ថ្នាក់អត្ថបទ
======================================================

.. note::

   ប្រភព៖ ថត `ch06 <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch06>`_
   និងតារាងមាតិកាក្នុង `README.md <https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md>`_ របស់ upstream។

ជំពូកនេះបង្ហាញការកែសម្រួល (finetune) ម៉ូដែល GPT ដែលបាន pretrain ហើយ
ឱ្យក្លាយជាម៉ូដែលចំណាត់ថ្នាក់អត្ថបទ (text classification)។ បណ្ណាល័យដែល
``requirements.txt`` កំណត់សម្រាប់ជំពូកនេះ៖ ``matplotlib`` · ``tensorflow`` · ``pandas``។

មេរៀនពេញលេញ (សិក្សាងាយ)
---------------------------

ការពន្យល់ខាងក្រោមជា **អត្ថបទដើមរបស់សហគមន៍ខ្មែរ** គ្របដណ្តប់គំនិតទាំងអស់នៃជំពូក ៦។
ជំពូកនេះបង្ហាញពីរបៀប **finetune** LLM សម្រាប់កិច្ចការចំណាត់ថ្នាក់អត្ថបទ។

១. ការផ្លាស់ប្តូរគំនិត៖ ពីបង្កើតអត្ថបទ ទៅចំណាត់ថ្នាក់
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ម៉ូដែលពីជំពូក ៥ បញ្ចេញ logits ទំហំ ``vocab_size`` ដើម្បីព្យាករណ៍ token បន្ទាប់។
សម្រាប់ចំណាត់ថ្នាក់ យើងត្រូវការ output តូចជាងនេះច្រើន — ឧទាហរណ៍ **២ ថ្នាក់** (spam / not spam)។

.. code-block:: text

   មុន (LM head):      hidden (emb_dim)  ->  vocab_size (50257)
   ក្រោយ (classification head):  hidden (emb_dim)  ->  num_classes (2)

ដូច្នេះការ finetune សម្រាប់ចំណាត់ថ្នាក់ = **ជំនួសស្រទាប់ចុងក្រោយ** រួចបណ្តុះបណ្តាលឡើងវិញ។

២. ហេតុអ្វី finetune ជាជាងបណ្តុះបណ្តាលពីសូន្យ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - វិធីសាស្ត្រ
     - លក្ខណៈ
   * - បណ្តុះបណ្តាលពីសូន្យ
     - ត្រូវការទិន្នន័យច្រើនមហាសាល + GPU ច្រើន + ពេលវេលាយូរ
   * - **Finetune LLM**
     - ប្រើចំណេះដឹងភាសាដែលរៀនរួចហើយ ត្រូវការទិន្នន័យមានស្លាកតិចតួច និងពេលវេលាខ្លី
   * - ម៉ូដែលបុរាណ (Naive Bayes, logistic regression)
     - លឿន និងត្រូវការធនធានតិច ប៉ុន្តែមិនយល់បរិបទស្មុគស្មាញ

ថត ``03_bonus_imdb-classification`` ប្រៀបធៀប LLM ជាមួយម៉ូដែលផ្សេងៗលើ dataset IMDb
៥០ ពាន់ movie review ដើម្បីបង្ហាញពី trade-off នេះជាលេខពិត។

៣. Dataset និងការបំបែកទិន្នន័យ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ជំពូកនេះប្រើ dataset សារអ៊ីមែល spam។ វិន័យសំខាន់បំផុតក្នុង machine learning៖

.. code-block:: text

   ទិន្នន័យសរុប
      |
      +-- Training set   (~៧០%)  ម៉ូដែលរៀនពីនេះ
      +-- Validation set (~១៥%)  សម្រាប់កំណត់ hyperparameter និងត្រួតពិនិត្យ overfitting
      +-- Test set       (~១៥%)  វាយតម្លៃចុងក្រោយ — ប្រើតែម្តងគត់!

.. important::

   **កុំវាយតម្លៃលើ training set** ហើយរាយការណ៍ថាជាលទ្ធផលពិត។ ម៉ូដែលអាច
   "ចងចាំ" ទិន្នន័យបាន។ ហើយ **កុំប្រើ test set ដើម្បីជ្រើសរើស hyperparameter** —
   បើធ្វើដូច្នោះ test set ក្លាយជា validation set ដោយប្រយោល ហើយលទ្ធផលរបស់អ្នក
   នឹងលម្អិតលើសការពិត។

៤. ការជ្រើសរើស token ដើម្បីចំណាត់ថ្នាក់
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LLM ដំណើរការលើ sequence ទាំងមូល ប៉ុន្តែចំណាត់ថ្នាក់ត្រូវការ **លទ្ធផលតែមួយ** សម្រាប់អត្ថបទទាំងមូល។
តើយើងប្រើ hidden state នៃ token ណា?

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - ជម្រើស
     - លក្ខណៈ
   * - **Token ចុងក្រោយ**
     - ជម្រើសស្តង់ដារសម្រាប់ decoder-only។ ដោយសារ causal attention token ចុងក្រោយបានមើលឃើញព័ត៌មាន **ទាំងអស់** នៃអត្ថបទ ដូច្នេះវាជាសេចក្តីសង្ខេបដ៏ល្អបំផុត
   * - Token ដំបូង
     - មិនសមហេតុផលសម្រាប់ decoder-only ព្រោះវាមើលឃើញតែខ្លួនឯង
   * - មធ្យមភាគនៃ token ទាំងអស់
     - អាចធ្វើបាន ប៉ុន្តែត្រូវការការកែប្រែស្ថាបត្យកម្មបន្ថែម

ថត ``02_bonus_additional-experiments`` ពិសោធន៍លើជម្រើសទាំងនេះដោយផ្ទាល់
(រួមទាំងការបង្រីកប្រវែង input) ដើម្បីបង្ហាញពីផលប៉ះពាល់ជាក់ស្តែង។

៥. Freezing — ការបណ្តុះបណ្តាលតែផ្នែកខ្លះ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

យើងមិនចាំបាច់ធ្វើបច្ចុប្បន្នភាព parameter ទាំងអស់ទេ។ យុទ្ធសាស្ត្រ៖

.. code-block:: text

   for param in model.parameters():
       param.requires_grad = False          # បិទទាំងអស់

   for param in model.out_head.parameters():
       param.requires_grad = True           # បើកតែស្រទាប់ចុងក្រោយ

អត្ថប្រយោជន៍នៃការ freeze៖

- **លឿនជាង** និង **សន្សំអង្គចងចាំ** យ៉ាងច្រើន។
- **កាត់បន្ថយ overfitting** ព្រោះ parameter ដែលរៀនមានចំនួនតិច។
- **រក្សាចំណេះដឹងភាសា** ដែលរៀនពី pretraining។

ជម្រើសដែលអាចធ្វើបាន៖ freeze ទាំងអស់លើកលែង head · freeze ទាំងអស់លើកលែង Transformer block
ចុងក្រោយ · ឬមិន freeze អ្វីសោះ (full finetuning)។

៦. Loss សម្រាប់ចំណាត់ថ្នាក់
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

នៅតែប្រើ cross-entropy ប៉ុន្តែលើ **ថ្នាក់** ជំនួសឱ្យ vocabulary៖

.. code-block:: text

   logits:  (batch_size, num_classes)
   labels:  (batch_size,)   ជាលេខគត់នៃថ្នាក់  ដូចជា 0 ឬ 1

   loss = F.cross_entropy(logits, labels)

ចំណាំ៖ ``F.cross_entropy`` ក្នុង PyTorch រួមបញ្ចូល softmax រួចហើយ — កុំហៅ softmax
ដោយដៃមុនពេលបញ្ជូនចូល ព្រោះវានឹងខុស។

៧. រង្វាស់វាយតម្លៃ
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - រង្វាស់
     - រូបមន្ត / ការបកស្រាយ
   * - Accuracy
     - ``(ទាយត្រូវ) / (សរុប)`` — ងាយយល់ ប៉ុន្តែ **មិនគួរប្រើ** បើថ្នាក់មិនស្មើគ្នា
   * - Precision
     - ក្នុងចំណោមអ្វីដែលយើងថា spam តើប៉ុន្មានជា spam ពិត?
   * - Recall
     - ក្នុងចំណោម spam ពិតទាំងអស់ តើយើងរកឃើញប៉ុន្មាន?
   * - F1 score
     - មធ្យមភាគអាម៉ូនិកនៃ precision និង recall

.. warning::

   បញ្ហា **class imbalance**៖ បើ ៩៥% នៃអ៊ីមែលមិនមែន spam ម៉ូដែលដែលទាយ
   "មិនមែន spam" ជានិច្ច នឹងទទួលបាន accuracy ៩៥% — ប៉ុន្តែវាគ្មានប្រយោជន៍សោះ។
   ក្នុងករណីបែបនេះ សូមប្រើ precision/recall/F1។

៨. User interface
~~~~~~~~~~~~~~~~~~~~~

ថត ``04_user_interface`` អនុវត្តចំណុចប្រទាក់អន្តរកម្ម ដើម្បីឱ្យអ្នកអាចបញ្ចូលអត្ថបទ
ហើយទទួលបានការវិនិច្ឆ័យភ្លាមៗ។ នេះជាជំហានចុងក្រោយដែលបំលែងម៉ូដែល
ទៅជាផលិតផលដែលអ្នកដទៃអាចប្រើបាន។

៩. សំណួរពិនិត្យការយល់ដឹង
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ហេតុអ្វី token ចុងក្រោយជាជម្រើសល្អសម្រាប់ចំណាត់ថ្នាក់ក្នុង decoder-only model?
2. តើអ្វីកើតឡើងបើអ្នកប្រើ test set ដើម្បីជ្រើសរើស learning rate?
3. ហេតុអ្វី freezing ជួយកាត់បន្ថយ overfitting?
4. បើ dataset មានសារ spam តែ ៥% តើ accuracy ជារង្វាស់ល្អទេ? ហេតុអ្វី?
5. តើអ្វីខុសបើអ្នកហៅ softmax មុន ``F.cross_entropy``?

កូដមេ
-------

កូដមេស្ថិតក្នុងថត
`ch06/01_main-chapter-code <https://github.com/SeyhaLite/LLMs-from-scratch/tree/main/កូដ/ch06/01_main-chapter-code>`_៖

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ឯកសារ
     - តួនាទី
   * - `ch06.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-finetuning-classification.ipynb>`_
     - notebook មេរបស់ជំពូក
   * - `gpt_class_finetune.py <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/កូដ/ch06/01_main-chapter-code/gpt_class_finetune.py>`_
     - ស្គ្រីប finetuning សម្រាប់ចំណាត់ថ្នាក់
   * - `exercise-solutions.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-ចម្លើយលំហាត់.ipynb>`_
     - ចម្លើយលំហាត់របស់ជំពូក

Notebooks ខ្មែរសម្រាប់ជំពូកនេះ
-----------------------------

ឃ្លាំងឯកសារនេះមានកំណែ notebook ជាភាសាខ្មែរ (ថត ``ណូតប៊ុក-ខ្មែរ/``)៖

- `ជំពូក៦-finetuning-classification.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-finetuning-classification.ipynb>`_ — notebook មេ ព្រមទាំងការពន្យល់ និង comments ជាខ្មែរ
- `ជំពូក៦-ចម្លើយលំហាត់.ipynb <https://github.com/SeyhaLite/LLMs-from-scratch/blob/main/ណូតប៊ុក-ខ្មែរ/ជំពូក៦-ចម្លើយលំហាត់.ipynb>`_ — ចម្លើយលំហាត់របស់ជំពូក

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
   * - ``02_bonus_additional-experiments``
     - ការពិសោធន៍បន្ថែម៖ finetune ស្រទាប់ផ្សេងៗ និងប្រើម៉ូដែលធំជាង
   * - ``03_bonus_imdb-classification``
     - Finetuning ម៉ូដែលផ្សេងៗលើ dataset រឿងភាគ IMDb 50k (50k IMDb Movie Review Dataset)
   * - ``04_user_interface``
     - ការសាងសង់ User Interface សម្រាប់ប្រើជាមួយ GPT-based Spam Classifier

លំហាត់
--------

ចម្លើយលំហាត់មានក្នុង ``exercise-solutions.ipynb`` ខាងលើ និងក្នុងសង្ខេបនៃភាគបន្ថែម C។

.. tip::

   បន្ទាប់៖ finetune ម៉ូដែលឱ្យធ្វើតាមការណែនាំដូច chatbot ក្នុង
   :doc:`ជំពូក៧-finetuning-instructions`។
