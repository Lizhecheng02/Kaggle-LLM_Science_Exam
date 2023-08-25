import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers

from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from langchain.prompts import PromptTemplate
from IPython.display import Markdown, display
from tqdm import tqdm

data = pd.read_csv("High Quality Dataset.csv")
data = data.dropna()


def clean(text):
    return text.strip()


for column in data.columns:
    if column != "id":
        data[column] = data[column].astype("string")
        data[column] = data[column].apply(clean)

data = data.drop_duplicates(
    subset=['prompt', 'A', 'B', 'C', 'D', 'E'])

data = data.sample(len(data), random_state=2023)
data["id"] = range(len(data))
data.reset_index(drop=True, inplace=True)
print(data.head())

data.to_csv("Shuffled Data.csv", index=False)
data = load_dataset("csv", data_files="Shuffled Data.csv", split="train")
# data = load_dataset("csv", data_files="train.csv", split="train")
print(data)

template = """
Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]\n
Question: {prompt}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n
E) {e}\n
Answer: {answer}"""

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

model_id = "meta-llama/Llama-2-7b-hf"
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_auth_token=access_token)
tokenizer.pad_token = tokenizer.eos_token

qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

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
    use_auth_token=access_token
)

model.config.use_cache = False
# model.config.pretraining_tp=1

print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters: ", total_params)

training_args = TrainingArguments(
    output_dir="./SFT-Llama-7b",
    per_device_train_batch_size=2,
    #     per_device_eval_batch_size=2,
    #     gradient_accumulation_steps=2,
    learning_rate=5e-6,
    logging_steps=100,
    logging_strategy="steps",
    #     max_steps=2,
    num_train_epochs=2,
    optim="paged_adamw_8bit",
    fp16=True,
    run_name="baseline-llama-sft",
    report_to="none"
)

trainer = SFTTrainer(
    model,
    train_dataset=data,
    args=training_args,
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()

print("Saving The Final Model...")
trainer.save_model("./finetuned_llama_7b")
# os.makedirs("./model", exist_ok=True)
# trainer.model.save_pretrained("./model")

finetuned_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=access_token
)
finetuned_model = PeftModel.from_pretrained(
    finetuned_model, "./finetuned_llama_7b")
# finetuned_model = finetuned_model.merge_and_unload()
print(finetuned_model)

print(model)

params1 = model.state_dict()
params2 = finetuned_model.base_model.model.state_dict()


def are_models_equal(params1, params2):
    for key in params2.keys():
        if key in params1.keys():
            if not torch.allclose(params1[key].half(), params2[key].half()):
                return False
#             else:
#                 print(params1[key].half())
#                 print(params2[key].half())
#                 print("Same!!!")
        else:
            print("Additional Keys:", key)
    return True


if are_models_equal(params1, params2):
    print("They are the same model")
else:
    print("They are not the same model")


if True:
    test = pd.read_csv("test.csv", index_col="id")
    test["answer"] = "A"
else:
    test = pd.read_csv("train.csv", index_col="id")
print(test.head())


class Perplexity(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity


perp = Perplexity()


def precision_at_k(r, k):
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k


def MAP_at_3(predictions, true_items):
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u]
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k + 1) * user_results[k]
    return map_at_3 / U


maps = []
preds = []
for idx, row in tqdm(test.iterrows(), total=len(test)):
    with torch.no_grad():
        cols = ["A", "B", "C", "D", "E"]
        perps = []
        samples = []
        for col in cols:
            samples.append("<|question|>" + row["prompt"] +
                           "</s><|answer|>" + "answer: " + row[col])
        inputs = tokenizer(samples, return_tensors="pt",
                           add_special_tokens=False, padding=True,
                           truncation=True).to("cuda")
        output = model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"])
        output = output.logits
        labels = inputs["input_ids"]
        labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
        for j in range(len(cols)):
            p = perp(output[j].unsqueeze(0), labels[j].unsqueeze(0))
            perps.append(p.detach().cpu())

        del inputs
        del labels
        del output
        del p

    perps = np.array(perps)

    predictions = [np.array(cols)[np.argsort(perps)]]
    preds.append(predictions)
    tp = [row.answer]
    map = MAP_at_3(predictions, tp)
    maps.append(map)
    print(np.mean(maps))

submission = pd.read_csv("sample_submission.csv")
submission["prediction"] = [" ".join(p[0][:3]) for p in preds]

print(submission.head())

submission.to_csv("submission.csv", index=False)
