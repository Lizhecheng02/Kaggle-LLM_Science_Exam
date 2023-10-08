import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import datasets
from transformers import get_polynomial_decay_schedule_with_warmup
import argparse
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, AutoConfig
from dataclasses import dataclass
from typing import Optional, Union
import os
from transformers import Trainer


class AWP:
    def __init__(self, model, adv_param="weight", adv_lr=0.1, adv_eps=0.0001):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

#     def attack_backward(self, inputs):
#         if self.adv_lr == 0:
#             return
#         self._save()
#         self._attack_step()

#         y_preds = self.model(inputs)

#         adv_loss = self.criterion(y_preds, labels)
#         self.optimizer.zero_grad()
#         return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class CustomTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None,
                 awp_lr=0,
                 awp_eps=0,
                 awp_start_epoch=0):

        super().__init__(model=model,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)

        self.awp_lr = awp_lr
        self.awp_eps = awp_eps
        self.awp_start_epoch = awp_start_epoch

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        ########################
        # AWP
        if self.awp_lr != 0 and self.state.epoch >= self.awp_start_epoch:
            self.awp = AWP(model, adv_lr=self.awp_lr, adv_eps=self.awp_eps)
            self.awp._save()
            self.awp._attack_step()
            with self.compute_loss_context_manager():
                awp_loss = self.compute_loss(self.awp.model, inputs)

            if self.args.n_gpu > 1:
                awp_loss = awp_loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(awp_loss).backward()
            elif self.use_apex:
                with amp.scale_loss(awp_loss, self.optimizer) as awp_scaled_loss:
                    awp_scaled_loss.backward()
            else:
                self.accelerator.backward(awp_loss)
            self.awp._restore()
        ########################

        return loss.detach() / self.args.gradient_accumulation_steps


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


def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions), axis=1)[:, :3]
    for x, y in zip(pred, labels):
        z = [1/i if y == j else 0 for i, j in zip([1, 2, 3], x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)


def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


def train(args):

    VER = args.VER
    # HUGGING FACE MODEL
    MODEL = args.MODEL

    # load data
    df_train = pd.read_json(args.train_data).reset_index(drop=True)
    df_valid = pd.read_json(args.valid_data).reset_index(drop=True)
    # process dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
    index_to_option = {v: k for k, v in option_to_index.items()}

    def preprocess(example):
        first_sentence = ["[CLS] " + example['context']] * 5
        second_sentences = [" #### " + example['prompt'] +
                            " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first',
                                      max_length=args.MAX_INPUT, add_special_tokens=False)
        tokenized_example['label'] = option_to_index[example['answer']]
        return tokenized_example

    dataset_valid = datasets.Dataset.from_pandas(df_valid)
    dataset = datasets.Dataset.from_pandas(df_train)
    tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=[
                                                'prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    tokenized_dataset = dataset.map(preprocess, remove_columns=[
                                    'prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    config = AutoConfig.from_pretrained(MODEL)
    config.hidden_dropout_prob = args.dropout_rate
    config.attention_probs_dropout_prob = args.dropout_rate
    model = AutoModelForMultipleChoice.from_pretrained(MODEL, config=config)

    training_args = TrainingArguments(
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        report_to='none',
        output_dir=f'./checkpoints_{VER}',
        overwrite_output_dir=True,
        fp16=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        metric_for_best_model='map@3',
        lr_scheduler_type='cosine',
        weight_decay=args.weight_decay,
        save_total_limit=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_args.learning_rate)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=training_args.num_train_epochs *
        int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
            training_args.gradient_accumulation_steps),
        power=1.0,
        lr_end=args.lr_end
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        awp_lr=args.awp_lr,
        awp_eps=args.awp_eps,
        awp_start_epoch=args.awp_start_epoch
    )

    trainer.train()
    trainer.save_model(f'model_v{VER}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--per_device_train_batch_size', default=1, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=4e-6, type=float)
    parser.add_argument('--lr_end', default=2e-6, type=float)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--num_train_epochs', default=5, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--logging_steps', default=5, type=int)
    parser.add_argument('--eval_steps', default=8, type=int)
    parser.add_argument('--save_steps', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--MAX_INPUT', default=8, type=int)
    parser.add_argument(
        '--MODEL', default='microsoft/deberta-v3-large', type=str)
    parser.add_argument('--VER', default=5, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--awp_lr', default=0.1, type=float)
    parser.add_argument('--awp_eps', default=1e-4, type=float)
    parser.add_argument('--awp_start_epoch', default=0.5, type=float)
    parser.add_argument('--label_smoothing_factor', default=0, type=float)
    args = parser.parse_args()
    train(args)
