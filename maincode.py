import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn
from tqdm.notebook import tqdm
from typing import Optional, Union
from datasets import Dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, BitsAndBytesConfig, AutoTokenizer, AutoConfig, AutoModel
# from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
# from sklearn.preprocessing import normalize

train_df = pd.read_csv('21k Dataset.csv')
train_df = train_df.dropna()


def clean(text):
    return text.strip()


for train_column in train_df.columns:
    if train_column != 'id':
        train_df[train_column] = train_df[train_column].astype('string')
        train_df[train_column] = train_df[train_column].apply(clean)


def cal_length(text):
    return len(text.split(' '))


train_df['total_length'] = train_df['prompt'].apply(cal_length) + train_df['A'].apply(cal_length) + \
    train_df['B'].apply(cal_length) + train_df['C'].apply(cal_length) + \
    train_df['D'].apply(cal_length) + train_df['E'].apply(cal_length)

train_df = train_df[train_df['total_length'] <= 256]
train_df = train_df.sample(len(train_df), random_state=42)
train_df.drop(columns=['total_length'], inplace=True)

val_df = train_df[len(train_df) - int(len(train_df) * 0.10):len(train_df)]
train_df = train_df[:len(train_df) - int(len(train_df) * 0.10)]
val_df.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)
val_df['id'] = range(len(val_df))
train_df['id'] = range(len(train_df))

print(train_df.shape)
print(val_df.shape)
print(train_df.dtypes)

model_name = 'microsoft/deberta-v2-xlarge'

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForMultipleChoice.from_pretrained(model_name)
print(model)


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.base_model = AutoModel.from_pretrained(
#             "microsoft/deberta-v3-large")
#         self.pooler = nn.Sequential(
#             nn.Linear(in_features=1024, out_features=1024, bias=True),
#             nn.Dropout(0.4),
#         )
#         self.classifier = nn.Linear(
#             in_features=1024, out_features=1, bias=True)
#         self.dropout = nn.Dropout(0.4)

#     def forward(self, x):
#         x = self.base_model(x)
#         x = self.pooler(x)
#         x = self.classifier(x)
#         x = self.dropout(x)
#         return x
# model = MyModel()
# print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total trainable parameters: ', total_params)


def predictions_to_map_output(predictions):
    sorted_answer_indices = np.argsort(-predictions)
    top_answer_indices = sorted_answer_indices[:, :3]
    top_answers = np.vectorize(index_to_option.get)(top_answer_indices)
    return np.apply_along_axis(lambda row: ' '.join(row), 1, top_answers)


options = 'ABCDE'
indices = list(range(5))

option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}


def preprocess(example):
    try:
        first_sentence = [example['prompt']] * 5
        second_sentence = [example[option] for option in options]

        tokenized_example = tokenizer(
            first_sentence, second_sentence, truncation=True)
        tokenized_example['label'] = option_to_index[example['answer']]
        return tokenized_example
    except:
        print(example)


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1)
                 for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


train_ds = Dataset.from_pandas(train_df)
tokenized_train_ds = train_ds.map(preprocess, batched=False, remove_columns=[
                                  'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
val_ds = Dataset.from_pandas(val_df)
tokenized_val_ds = val_ds.map(preprocess, batched=False, remove_columns=[
                              'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

training_args = TrainingArguments(
    output_dir='./',
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=4500,
    logging_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    learning_rate=3e-6,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    report_to='none',
    # lr_scheduler_type="constant",
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer)
)

trainer.train()

model.save_pretrained('./saved_deberta-v2-xlarge_21k_model')
tokenizer.save_pretrained('./saved_deberta-v2-xlarge_21k_model')

test_df = pd.read_csv('test.csv')
test_df['answer'] = 'A'
test_ds = Dataset.from_pandas(test_df)
tokenized_test_ds = test_ds.map(preprocess, batched=False, remove_columns=[
                                'prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

test_predictions = trainer.predict(tokenized_test_ds)

submission = test_df[['id']]

submission['prediction'] = predictions_to_map_output(
    test_predictions.predictions)

submission.to_csv('submission.csv', index=False)

submission.head(10)
