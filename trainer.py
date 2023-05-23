import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
from models import RankGenModel
from rank_datasets import DataCollatorForPairRank, RankGenCollator
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, Trainer, TrainingArguments
from transformers.training_args import OptimizerNames
from utils import argument_parsing, freeze_top_n_layers, get_datasets, get_tokenizer

os.environ["WANDB_PROJECT"] = "reward-model"

accuracy = evaluate.load("accuracy")
parser = ArgumentParser()
# Note, these override the config yaml, and get merged in argument_parsing() in utils.py
# Do not set defaults here, but set them in utils.py so that the config yaml can override them.
parser.add_argument("config", type=str)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--deepspeed", action="store_true")
parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
parser.add_argument("--wandb-entity", type=str)
parser.add_argument("--per-digit-tokens", action="store_true")
parser.add_argument("--output_dir", type=str)


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=[0] * predictions.shape[0])


def extended_log(x, eps=1e-1):
    return torch.where(x > 0, torch.log(x+eps)-np.log(eps), 10*x)

def extended_inv(x, eps=1e-1):
    return torch.where(x > 0, 1/(x+eps)-1/eps, -100*x)

def extended_x2(x, eps=1e-1):
    return torch.where(x > 0, (x+eps)**2-eps**2, x)

def extended_inv2(x, eps=1e-1):
    return torch.where(x > 0, (1/(x+eps))**2-1/eps**2, -x)

class RankLoss(nn.Module):
    def __init__(self, eps=1e-8, loss_type="log") -> None:
        super().__init__()
        self.eps = eps
        if loss_type == "log":
            self.loss = lambda x: -extended_log(x)
        elif loss_type == "x":
            self.loss = lambda x: -x
        elif loss_type == "x2":
            self.loss = lambda x: -extended_x2(x)
        elif loss_type == "x_inv2":
            self.loss = lambda x: extended_inv2(x)
        elif loss_type == "x_inv":
            self.loss = lambda x: extended_inv(x)
        elif loss_type == "adaptive":
            self.loss = [
                lambda x: -x,
                lambda x: extended_inv(x)
            ]
        elif loss_type == "adaptive2":
            self.loss = [
                lambda x: -extended_x2(x),
                lambda x: extended_inv2(x)
            ]
        elif loss_type == "logsigmoid":
            self.loss = lambda x: -torch.log(torch.sigmoid(4*x))

    def forward(self, pos, neg, uniform=None):
        if uniform is not None and isinstance(self.loss, list):
            loss = self.loss[uniform](torch.sigmoid(pos) - torch.sigmoid(neg))
        else:
            loss = self.loss(torch.sigmoid(pos) - torch.sigmoid(neg))
        return loss


class RankTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        model_name: str = None,
        args: Optional[TrainingArguments] = None,
        loss_function: str = "rank",
        use_scores: bool = False,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.loss_fct = RankLoss(loss_type=loss_function) if loss_function != "xent" else nn.CrossEntropyLoss()
        self.loss_function = loss_function
        self.model_name = model_name
        self.use_scores = use_scores

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass

        # import pdb; pdb.set_trace()
        if "rankgen" in self.model_name:
            positive_outputs = model(inputs["prefix"], inputs["positive"])
            negative_outputs = model(inputs["prefix"], inputs["negative"])
            if self.loss_function != "xent":
                loss = self.loss_fct(positive_outputs, negative_outputs)
            else:
                raise NotImplementedError("Only ranking loss has been implemented for rankgen model")
            outputs = torch.hstack((positive_outputs[:, None], negative_outputs[:, None]))
        else:
            # import pdb; pdb.set_trace()
            inputs.pop("token_type_ids", None)
            scores = inputs.pop("scores", None)
            uniform = inputs.pop("uniform", None)
            outputs = model(**inputs)
            logits = outputs.get("logits").view(-1, 2)
            if self.loss_function != "xent":
                loss = self.loss_fct(logits[:, 0], logits[:, 1], uniform=uniform)
                if self.use_scores:
                    loss = loss * scores
                loss = loss.mean()
            else:
                loss = self.loss_fct(logits, torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long))

        return (loss, outputs) if return_outputs else loss

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        scores = inputs.pop("scores", None)
        uniform = inputs.pop("uniform", None)
        outputs = model(**inputs)
        logits = outputs.get("logits").view(-1, 2)
        if self.loss_function != "xent":
            loss = self.loss_fct(logits[:, 0], logits[:, 1], uniform=uniform)
            # if scores is not None:
            #     self.log_metrics("eval", {"correlation": torch.corrcoef(torch.vstack((scores.view(-1), loss.view(-1))))[0, 1]})
            loss = loss.mean()
        else:
            loss = self.loss_fct(logits, torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long))

        return loss, logits

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.inference_mode():
            if "rankgen" in self.model_name:
                inputs = self._prepare_inputs(inputs)
                positive_outputs = model(inputs["prefix"], inputs["positive"])
                negative_outputs = model(inputs["prefix"], inputs["negative"])
                if self.loss_function != "xent":
                    loss = self.loss_fct(positive_outputs, negative_outputs)
                else:
                    raise NotImplementedError("Only ranking loss has been implemented for rankgen model")
                logits = torch.hstack((positive_outputs[:, None], negative_outputs[:, None]))
                # Create labels which are not None so HF will call compute_metrics:
                labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
                return loss, logits, labels
            else:
                loss, logits = self._compute_loss(model, inputs)

                loss = loss.mean().detach()
                labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
                if self.args.prediction_loss_only:
                    return loss, None, None

                return loss, logits, labels


if __name__ == "__main__":
    training_conf = argument_parsing(parser)

    model_name = training_conf["model_name"]
    if "rankgen-t5" in model_name:
        model = RankGenModel(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")
    if "freeze_layer" in training_conf:
        num_layer = training_conf["freeze_layer"]
        model = freeze_top_n_layers(model, num_layer)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of trainable : {}M".format(int(params / 1e6)))

    # model = torch.compile(model)

    optimizer = OptimizerNames.ADAMW_HF
    args = TrainingArguments(
        output_dir=training_conf["output_dir"],
        # num_train_epochs=training_conf["num_train_epochs"],
        max_steps=training_conf["max_steps"],
        warmup_ratio=training_conf["warmup_ratio"],
        optim=optimizer,
        # lr_scheduler_type=training_conf["scheduler"],
        learning_rate=training_conf["learning_rate"],
        half_precision_backend="auto",
        deepspeed="configs/zero_config.json" if training_conf["deepspeed"] else None,
        fp16=training_conf["fp16"],
        local_rank=training_conf["local_rank"],
        gradient_checkpointing=training_conf["gradient_checkpointing"],
        gradient_accumulation_steps=training_conf["gradient_accumulation_steps"],
        per_device_train_batch_size=training_conf["per_device_train_batch_size"],
        per_device_eval_batch_size=training_conf["per_device_eval_batch_size"],
        weight_decay=training_conf["weight_decay"],
        max_grad_norm=training_conf["max_grad_norm"],
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=training_conf["eval_steps"],
        save_steps=training_conf["save_steps"],
        report_to="wandb",
    )

    tokenizer = get_tokenizer(training_conf["tokenizer_name"], training_conf["per_digit_tokens"])
    train, evals = get_datasets(training_conf["datasets"], tokenizer)
    if "rankgen" in model_name:
        collate_fn = RankGenCollator(tokenizer, max_length=training_conf["max_length"])
    else:
        collate_fn = DataCollatorForPairRank(
            tokenizer,
            max_length=training_conf["max_length"],
            drop_token_type=training_conf.get("drop_token_type", False),
        )
    assert len(evals) > 0

    if not training_conf["deepspeed"] or training_conf["local_rank"] == 0:
        import wandb

        wandb.init(
            project=os.environ["WANDB_PROJECT"], name=f"{model_name}-finetuned"
        )

    trainer = RankTrainer(
        model=model,
        model_name=model_name,
        args=args,
        loss_function=training_conf["loss"],
        train_dataset=train,
        eval_dataset=torch.utils.data.ConcatDataset(evals.values()),
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        use_scores=training_conf["use_scores"],
        # optimizers=(optimizer, scheduler),
    )
    trainer.train()
    trainer.evaluate()
