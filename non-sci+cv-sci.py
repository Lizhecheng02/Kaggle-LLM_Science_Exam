import os
import numpy as np
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
import regex as re
from datasets import Dataset
from sklearn.model_selection import KFold
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

total_df = pd.read_parquet("./clean_dataset/fillna_clean_60k.parquet")

"""
In this python file, we use 52k dataset to train a pretrained model to learn MCQ,
then we use cross validation and 8k science data to finetune model on Science MCQ.
"""

science_df = total_df[(total_df["source"] == 6) | (total_df["source"] == 7)]
science_df.drop("source", axis=1, inplace=True)
print("The shape of science dataset is:", science_df.shape)

non_science_df = total_df[(total_df["source"] != 6)
                          & (total_df["source"] != 7)]
non_science_df.drop("source", axis=1, inplace=True)
non_science_df.reset_index(drop=True, inplace=True)
print("The shape of non science dataset is:", non_science_df.shape)

"""
Add science dataset from this directory
"""

add_science_df = pd.read_csv("./final_test.csv")
print("The shape of additional science dataset is:", add_science_df.shape)

science_df = pd.concat([science_df, add_science_df], axis=0)
science_df.reset_index(drop=True, inplace=True)
print("The shape of total science dataset is:", science_df.shape)

"""
Now we need to start pretraining step on non-science dataset
"""

non_science_df.drop_duplicates(
    subset=["prompt", "A", "B", "C", "D", "E"], inplace=True)
science_df.drop_duplicates(
    subset=["prompt", "A", "B", "C", "D", "E"], inplace=True)
print("The shape of non-science dataset after dropping duplicates is:",
      non_science_df.shape)
print("The shape of science dataset after dropping duplicates is:", science_df.shape)


def clean_context(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\(\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


non_science_df["context"] = non_science_df["context"].apply(clean_context)
science_df["context"] = science_df["context"].apply(clean_context)
non_science_df.to_csv("./non_science.csv", index=False)
science_df.to_csv("./science.csv", index=False)

non_science_df = non_science_df.sample(len(non_science_df))
science_df = science_df.sample(len(science_df))
non_science_df.reset_index(drop=True, inplace=True)
science_df.reset_index(drop=True, inplace=True)

"""
Now we start to train the model on non-science data
"""

MODEL = "microsoft/deberta-v3-large"
MAX_INPUT = 384

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMultipleChoice.from_pretrained(MODEL)

option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
index_to_option = {v: k for k, v in option_to_index.items()}


def preprocess(example):
    first_sentence = ["[CLS] " + example["context"]] * 5
    second_sentences = [" #### " + example["prompt"] +
                        " [SEP] " + example[option] + " [SEP]" for option in "ABCDE"]
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation="only_first",
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example["label"] = option_to_index[example["answer"]]
    return tokenized_example


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1)
                 for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


"""
Prepare dataset for training
"""
non_science_df_train = non_science_df[:len(non_science_df) - 2000]
non_science_df_valid = non_science_df[-2000:]

non_science_valid = Dataset.from_pandas(non_science_df_valid)
non_science_train = Dataset.from_pandas(non_science_df_train)

non_science_train = non_science_train.remove_columns(["__index_level_0__"])

tokenized_valid = non_science_valid.map(
    preprocess, remove_columns=[
        "prompt", "context", "A", "B", "C", "D", "E", "answer"
    ]
)
tokenized_train = non_science_train.map(
    preprocess, remove_columns=[
        "prompt", "context", "A", "B", "C", "D", "E", "answer"
    ]
)


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
    learning_rate=4e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    # report_to="none",
    output_dir=f"./checkpoints_non_science",
    overwrite_output_dir=True,
    fp16=True,
    gradient_accumulation_steps=16,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=False,
    metric_for_best_model="map@3",
    lr_scheduler_type="linear",
    weight_decay=0.01,
    save_total_limit=3
)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=training_args.num_train_epochs *
    int(len(tokenized_train) * 1.0 / training_args.per_device_train_batch_size /
        training_args.gradient_accumulation_steps),
    power=1.0,
    lr_end=2e-6
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

trainer.train()
trainer.save_model(f"./non_science")

del model
del trainer

model_dir = "./non_science"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMultipleChoice.from_pretrained(model_dir)

kf = KFold(n_splits=5, shuffle=True)

for fold, (train_idx, test_idx) in enumerate(kf.split(science_df)):
    print(10 * "=", f"Fold = {fold + 1}", 10 * "=")

    science_train = science_df.iloc[train_idx]
    science_valid = science_df.iloc[test_idx]

    science_train = science_train.reset_index(drop=True)
    science_valid = science_valid.reset_index(drop=True)

    science_train = Dataset.from_pandas(science_train)
    science_valid = Dataset.from_pandas(science_valid)

    science_train = science_train.remove_columns(["__index_level_0__"])

    tokenized_science_valid = science_valid.map(
        preprocess, remove_columns=[
            "prompt", "context", "A", "B", "C", "D", "E", "answer"
        ]
    )
    tokenized_science_train = science_train.map(
        preprocess, remove_columns=[
            "prompt", "context", "A", "B", "C", "D", "E", "answer"
        ]
    )

    training_args = TrainingArguments(
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        # report_to="none",
        output_dir=f"./checkpoints_science",
        overwrite_output_dir=True,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="map@3",
        lr_scheduler_type="linear",
        weight_decay=0.01,
        save_total_limit=2
    )

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=training_args.num_train_epochs *
        int(len(tokenized_science_train) * 1.0 / training_args.per_device_train_batch_size /
            training_args.gradient_accumulation_steps),
        power=1.0,
        lr_end=2e-6
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_science_train,
        eval_dataset=tokenized_science_valid,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()
    # trainer.save_model(f"./non_science_{fold + 1}")

    if (fold + 1) == 5:
        trainer.save_model("./final_model")

del model

model_dir = "./final_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMultipleChoice.from_pretrained(model_dir)


test_df = pd.read_csv("./clean_dataset/clean_200.parquet")
tokenized_test_dataset = Dataset.from_pandas(test_df).map(
    preprocess, remove_columns=["prompt", "context", "A", "B", "C", "D", "E"]
)

test_predictions = trainer.predict(tokenized_test_dataset).predictions
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list("ABCDE"))[predictions_as_ids]
predictions_as_string = test_df["prediction"] = [
    " ".join(row) for row in predictions_as_answer_letters[:, :3]
]


def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k


def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k + 1) * user_results[k]
    return map_at_3 / U


m = MAP_at_3(test_df.prediction.values, test_df.answer.values)
print("CV MAP@3 =", m)
