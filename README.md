# This is 10th Solution For [Kaggle LLM Science Competition](https://www.kaggle.com/competitions/kaggle-llm-science-exam)

### Here is [Original Discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446797)


### Dataset

- First of all, we want to express our gratitude to all the amazing Kagglers who have contributed in creating a larger collection of public datasets. 
- Finally, we ended up with around ``360k`` data in total, but because of time constraint, we only utilized approximately ``303k`` data for fine-tuning our models. 
- We generated around ``150k`` data using GPT-3.5 according to lots of  Wikipedia page contents, and combined it with [99k Dataset](https://www.kaggle.com/datasets/cdeotte/99k-data-with-context-v2), [60k Dataset](https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2), [17k MMLU Dataset](https://www.kaggle.com/datasets/cdeotte/40k-data-with-context-v2) and [70k Dataset](https://www.kaggle.com/datasets/sugupoko/llm-70kdataset-with-context). 
- Here: [Code To Generate Dataset](https://www.kaggle.com/code/lizhecheng/generate-new-dataset-using-chatgpt-3-5/notebook "here") and [Whole Dataset](https://www.kaggle.com/datasets/lizhecheng/llm-science-datasets-combination)

### Sentence Transformer Model

- We tried ``all-MiniLM-L6``, ``all-MiniLM-L12``, ``gte-small`` and ``bge-small-en``. 

- We tested different models when we were at ``public leaderboard`` score 0.843. 

  |                    | ``Before`` | ``After`` |
  | :----------------: | :--------: | :-------: |
  | ``all-MiniLM-L6``  | **0.843**  |           |
  | ``all-MiniLM-L12`` | **0.843**  | **0.837** |
  |   ``gte-small``    | **0.843**  | **0.851** |
  |  ``bge-small-en``  | **0.843**  | **0.822** |

  Finally we chose ``gte-small`` as our sentence transformer model.

- Here: [Notebook To Create Your Own Index](https://www.kaggle.com/code/lizhecheng/convert-wikipedia-to-faiss-index). 

### Retrieval 

- We improved our models' performance by extracting new contexts for each dataset, using the parameters ``pages=10`` and ``sentences=30``. 
- To ensure that search documents contain more relevant knowledge when testing on a private dataset, we extract the contents of Wikipedia pages from HTML and re-cluster them to obtain additional articles, thereby preventing any inconsistencies.
- We combined three types of retrievers to obtain the final outcome: 1. Extracting context from the entire Wikipedia page, 2. Regarding the two other retrievers proposed by [MB](https://www.kaggle.com/mbanaei), we have expanded the value ``target_articles`` in the shared notebook by MB. We manually extracted the titles of the dataset, which is considered a useful validation, and added them to the ``target_articles``. Additionally, we have selected more clusters to help us discover more relevant data, increasing it from 270K to approximately 6 million rows. Furthermore, we are utilizing both TF-IDF and sentence transformer techniques for context extraction.

### Training

- Training Dataset: We divided the entire dataset into multiple chunks to enhance the robustness of our ensemble models. For instance, we split the dataset into four parts and trained four models on each large dataset formed by combining three smaller portions. Additionally, we also trained a model using the entire dataset for comparison. Here: [222k-133k-5fold](https://www.kaggle.com/datasets/lizhecheng/llm-science-dataset-222k-133k-5fold), [222k-148k-3fold](https://www.kaggle.com/datasets/lizhecheng/llm-science-dataset-222k-148k-3fold), [303k-202k-3fold](https://www.kaggle.com/datasets/lizhecheng/llm-science-dataset-303k-202k-3fold).


- Validation Dataset: We randomly selected a 2k dataset from the entire dataset we generated using GPT-3.5 and combined it with 200 original training datasets to create our validation dataset. In the end, we selected our final submissions according to its performance on [500 Valid Dataset](https://www.kaggle.com/datasets/cdeotte/99k-data-with-context-v2?select=valid_500_with_context2.csv).

- Parameters: We are incredibly grateful that our teammate has 8 * A100 GPUs. This allowed us to set the max_length to either 512 or 768 for training different models, which could be more suitable. Additionally, we trained some models using fp32, which took approximately four days to train 222k data on a single A100 GPU. Here: [222k-133k-models](https://www.kaggle.com/datasets/lizhecheng/222k-133k-finetuned-deberta-v3-models), [222k-148k-models](https://www.kaggle.com/datasets/lizhecheng/222k-148k-finetuned-deberta-v3-models), [303k-202k-models](https://www.kaggle.com/datasets/lizhecheng/303k-202k-finetuned-deberta-v3-models), [222k-768-fp32-model](https://www.kaggle.com/datasets/lizhecheng/222k-fp32-768-debertav3-model), [303k+50k-finetune-model](https://www.kaggle.com/datasets/lizhecheng/303k-finetune-50k-debertav3-model)

- Adversarial Weight Perturbation: We incorporated adversarial weight perturbation into our training approach to boost the robustness of each individual model, resulting in a further reduction in our training loss. Here: [Training Notebook](https://www.kaggle.com/code/lizhecheng/kaggle-llm-competition-fine-tune-awp).

  | Parameters                      | Value              |
  | :------------------------------ | ------------------ |
  | ``per_device_train_batch_size`` | **2**              |
  | ``per_device_eval_batch_size``  | **1**              |
  | ``learning_rate``               | **4e-6**           |
  | ``lr_end``                      | **8e-7**           |
  | ``num_train_epochs``            | **2** or **3**     |
  | ``gradient_accumulation_steps`` | **16**             |
  | ``weight_decay``                | **1e-4**           |
  | ``MAX_INPUT``                   | **512** or **768** |
  | ``dropout_rate``                | **0**              |
  | ``awp_lr``                      | **0.1**            |
  | ``awp_eps``                     | **1e-4**           |
  | ``awp_start_epoch``             | **0.5**            |

  The above parameters are the best tested on a single A100 GPU.


- Ensemble Model: We used optuna to test the ensemble score on 500 validation dataset. We achieved a score of 0.915 on the GPT-3.5 500k dataset and 0.909 on the GPT-4 500k dataset. Because the test data set was generated using GPT-3.5, we finally chose the combination with a higher score under GPT-3.5. Submissions: [Submission1](https://www.kaggle.com/code/lizhecheng/kaggle-llm-competition-submission-type1), [Submission2](https://www.kaggle.com/code/lizhecheng/kaggle-llm-competition-submission-type2).

  |                         Combination                          |     Ratio      | Public Leaderboard | Private Leaderboard | Choose  |
  | :----------------------------------------------------------: | :------------: | :----------------: | :-----------------: | :-----: |
  | ``222k-full/checkpoint-12300`` + ``133k fold5/checkpoint-7100`` |    **2: 1**    |     **0.929**      |      **0.921**      | **No**  |
  | ``222k-full/checkpoint-12300`` + ``768_fp32_222k/checkpoint-27800`` + ``133k fold5/checkpoint-7100`` | **2: 1.5: 1**  |     **0.930**      |      **0.923**      | **Yes** |
  | ``222k_finetune/checkpoint-800`` + ``222k_relevant/checkpoint-12700`` |    **1: 1**    |     **0.930**      |      **0.924**      | **No**  |
  | ``148k fold1/checkpoint-5300`` + ``202k_fold3_512/checkpoint-12500`` + ``202k_fold2_512/checkpoint-12500`` + ``768-300k-best-para-epoch2/checkpoint-15600`` | **1: 1: 1: 1** |     **0.929**      |      **0.922**      | **No**  |
  | ``222k-full/checkpoint-12300`` + ``768_fp32_222k/checkpoint-27800`` + ``133k fold5/checkpoint-7100`` | **2: 1.5: 1**  |     **0.929**      |      **0.924**      | **No**  |
  | ``133k fold5/checkpoint-7100`` + ``303k-50k-finetune/checkpoint-2800`` + ``singlegpu-200k-run-sft-awp-0-1-ls-dropout-4e6/checkpoint-11300`` + ``768-300k-best-para-epoch2/checkpoint-18900`` | **1: 1: 1: 1** |     **0.930**      |      **0.923**      | **Yes** |
