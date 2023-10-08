import os
import numpy as np
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import (
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    get_polynomial_decay_schedule_with_warmup,
    AdamW
)

VER = 1
NUM_TRAIN_SAMPLES = 200_000
USE_PEFT = False
FREEZE_LAYERS = 0
FREEZE_EMBEDDINGS = False
MAX_INPUT = 384
MODEL = 'microsoft/deberta-v3-large'

"""
90k训练集，有分隔符，2200组测试集，有分隔符。
"""

# df_valid1 = pd.read_json('./final dataset/200 val with #.json')
# df_valid2 = pd.read_json('./final dataset/2k val with #.json')
# df_valid = pd.concat([df_valid1, df_valid2])
# print('Validation data size:', df_valid.shape)

# df_train1 = pd.read_json('./final dataset/60k full with #.json')
# df_train2 = pd.read_json('./final dataset/30k full with #.json')
# df_train = pd.concat([df_train1, df_train2])
# df_train = df_train.fillna('')
# df_train = df_train.drop_duplicates(subset=['prompt'])

"""
90k训练集，无分隔符，2200组测试集，无分隔符。
"""
# df_valid1 = pd.read_json('./final dataset/200 val no #.json')
# df_valid2 = pd.read_json('./final dataset/2k val no #.json')
# df_valid = pd.concat([df_valid1, df_valid2])
# print('Validation data size:', df_valid.shape)

# df_train1 = pd.read_json('./final dataset/60k full no #.json')
# df_train2 = pd.read_json('./final dataset/30k full no #.json')
# df_train = pd.concat([df_train1, df_train2])
# df_train = df_train.fillna('')
# df_train = df_train.drop_duplicates(subset=['prompt'])

"""
30k训练集，有分隔符，2200组测试集，有分隔符
"""
# df_valid1 = pd.read_json('./final dataset/200 val with #.json')
# df_valid2 = pd.read_json('./final dataset/2k val with #.json')
# df_valid = pd.concat([df_valid1, df_valid2])
# print('Validation data size:', df_valid.shape)

# df_train = pd.read_json('./final dataset/30k full with #.json')
# df_train = df_train.fillna('')
# df_train = df_train.drop_duplicates(subset=['prompt'])

"""
30k训练集，无分隔符，2200组测试集，无分隔符
"""
df_valid = pd.read_json('./final dataset/200 val no #.json')
# df_valid2 = pd.read_json('./final dataset/2k val no #.json')
# df_valid = pd.concat([df_valid1, df_valid2])
print('Validation data size:', df_valid.shape)

df_train = pd.read_json('./final dataset/30k full no #.json')
df_train = df_train.fillna('')
df_train = df_train.drop_duplicates(subset=['prompt'])

"""
打乱
"""

df_valid.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)

df_train = df_train.sample(min(NUM_TRAIN_SAMPLES, len(df_train)))
print('Train data size:', df_train.shape)

option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k, v in option_to_index.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMultipleChoice.from_pretrained(MODEL)


def preprocess(example):
    first_sentence = ["[CLS] " + example['context']] * 5
    second_sentences = [" #### " + example['prompt'] +
                        " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first',
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]

    return tokenized_example


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


dataset_valid = Dataset.from_pandas(df_valid)
dataset = Dataset.from_pandas(df_train)
dataset = dataset.remove_columns(["__index_level_0__"])
print(dataset)

tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=[
                                            'prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
tokenized_dataset = dataset.map(preprocess, remove_columns=[
                                'prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
print(tokenized_dataset)

if FREEZE_EMBEDDINGS:
    print('Freezing embeddings.')
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False


def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1 * np.array(predictions), axis=1)[:, :3]
    for x, y in zip(pred, labels):
        z = [1 / i if y == j else 0 for i, j in zip([1, 2, 3], x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)


def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


training_args = TrainingArguments(
    # warmup_ratio=0.005,
    learning_rate=4e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    report_to='none',
    output_dir=f'./checkpoints_{VER}',
    overwrite_output_dir=True,
    fp16=True,
    gradient_accumulation_steps=32,
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=False,
    metric_for_best_model='map@3',
    lr_scheduler_type='linear',
    weight_decay=0.01,
    save_total_limit=3
)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=training_args.num_train_epochs *
    int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
        training_args.gradient_accumulation_steps),
    # num_training_steps=6750,
    power=1.0,
    lr_end=2e-6
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

trainer.train()
trainer.save_model(f'model_v{VER}')

del model, trainer
