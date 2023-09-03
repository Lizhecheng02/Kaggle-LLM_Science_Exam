import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers

from datasets import load_dataset, Dataset
from typing import Optional, Union
from dataclasses import dataclass
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, Trainer
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from langchain.prompts import PromptTemplate
from IPython.display import Markdown, display
from tqdm import tqdm
# from sklearn.preprocessing import normalize

data = pd.read_csv("21k Dataset.csv")
data = data.dropna()


def clean(text):
    return text.strip()


for column in data.columns:
    if column != "id":
        data[column] = data[column].astype("string")
        data[column] = data[column].apply(clean)

data = data.sample(len(data), random_state=42)
data["id"] = range(len(data))
data.reset_index(drop=True, inplace=True)
# data = data[:1000]
print(data.head())
print(data['answer'].value_counts())

data.to_csv("Shuffled Data.csv", index=False)
data = load_dataset("csv", data_files="Shuffled Data.csv", split="train")
# data = load_dataset("csv", data_files="/kaggle/input/kaggle-llm-science-exam/train.csv", split="train")
print(data)

template = """Suppose you are an expert on all subjects related to science. Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]\n
###Question: {prompt}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n
E) {e}\n
\n###Output the correct answer: {answer}"""

prompt = PromptTemplate(template=template, input_variables=[
                        "prompt", "a", "b", "c", "d", "e", "answer"])

sample = data[0]
display(Markdown(prompt.format(prompt=sample["prompt"], a=sample["A"],
                               b=sample["B"], c=sample["C"], d=sample["D"],
                               e=sample["E"], answer=sample["answer"])))


def format_text(example):
    text = prompt.format(prompt=example["prompt"], a=example["A"],
                         b=example["B"], c=example["C"], d=example["D"],
                         e=example["E"], answer=example["answer"])
    return {"text": text}


data = data.map(format_text)
print(data)


def plot_sequence_lengths(data, split="train", max_length=2048):
    sequence_lengths = []
    selected_indices = []

    for idx, example in tqdm(enumerate(data), total=len(data)):
        sequence_lengths.append(len(example["text"]))
        if sequence_lengths[idx] < max_length:
            selected_indices.append(idx)

    plt.hist(sequence_lengths, bins=30)
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Distribution of Text Sequence Lengths")
    plt.show()

    print("Max Sequence Length:", max(sequence_lengths))
    print("Min Sequence Length:", min(sequence_lengths))

    return selected_indices


keep_indices_train = plot_sequence_lengths(data)
data = data.select(keep_indices_train)
print("The length of selected data:", len(data))


def format_func(example):
    output_texts = []
    for i in range(len(example)):
        text = f"Suppose you are an expert on all subjects related to science. Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]\n ###Question: {example['prompt']}\n A) {example['a']}\n B) {example['b']}\n C) {example['c']}\n D) {example['d']}\n E) {example['e']}\n ###Output the correct answer: {example['answer']}"
        output_texts.append(text)
    return output_texts


model_id = "baichuan-inc/Baichuan-13B-Chat"
# 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
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    # use_auth_token=access_token,
    trust_remote_code=True,
    #     pad_token="<|endoftext|>"
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    # use_auth_token=access_token
)

model.config.use_cache = False
model.config.pretraining_tp = 1

print(model)

qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # target_modules=["query_key_value", "dense",
    #                 "dense_h_to_4h", "dense_4h_to_h"],  # chatglm2-6b
    # target_modules=["k_proj", "v_proj", "q_proj", "out_proj",
    #                 "c_fc", "c_proj"],  # EleutherAI/gpt-neo-2.7B
    target_modules=["W_pack", "o_proj", "gate_proj",
                    "down_proj", "up_proj"],  # baichuan-13b-base
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    #                 "gate_proj", "down_proj", "up_proj"],  # internlm/internlm-chat-7b
    # target_modules=["fc_in", "fc_out", "k_proj", "q_proj",
    #                 "v_proj", "out_proj"],  # togethercomputer/GPT-JT-6B-v1
    task_type="CAUSAL_LM"
)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )


# model_ = get_peft_model(model, qlora_config)
# print_trainable_parameters(model_)
# print(model_)

training_args = TrainingArguments(
    output_dir="./SFT-Train",
    per_device_train_batch_size=2,
    #     per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=50,
    logging_strategy="steps",
    save_steps=1000,
    save_strategy="steps",
    # eval_steps=100,
    # evaluation_strategy="steps",
    save_total_limit=1,
    # load_best_model_at_end=True,
    # max_steps=2,
    num_train_epochs=2,
    optim="paged_adamw_32bit",
    fp16=True,
    run_name="baseline-sft",
    report_to="none",
    lr_scheduler_type="constant",
    warmup_ratio=0.02,
)

response_template_with_context = "\n###Output the correct answer:"
response_template_ids = tokenizer.encode(
    response_template_with_context, add_special_tokens=False)[2:]
response_template_text = tokenizer.decode(response_template_ids)
print(response_template_ids)
print(response_template_text)
data_collator = DataCollatorForCompletionOnlyLM(
    response_template_ids, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=data,
    args=training_args,
    tokenizer=tokenizer,
    peft_config=qlora_config,
    # dataset_text_field="text",
    formatting_func=format_func,
    # max_seq_length=2048,
    packing=False,
    data_collator=data_collator
)

trainer.train()

print("Saving The Final Model...")
trainer.save_model("./dir1")
trainer.model.save_pretrained("./dir2")

print(model)


# preds = []
# for _, row in tqdm(df.iterrows(), total=len(df)):
#     inputs = tokenizer(prompt.replace(
#         'query', row['instruction']), return_tensors="pt").to(f"cuda:{model.device.index}")
#     with torch.no_grad():
#         output = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=1024,
#                                 return_dict_in_generate=True, output_scores=True)

#     first_token_probs = output.scores[0][0]
#     option_scores = first_token_probs[[
#         319, 350, 315, 360, 382]].float().cpu().numpy()  # ABCDE
#     pred = np.array(["A", "B", "C", "D", "E"])[
#         np.argsort(option_scores)[::-1][:3]]
#     pred = ' '.join(pred)
#     preds.append(pred)

# sub['prediction'] = preds
