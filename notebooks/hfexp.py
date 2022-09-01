import argparse, os

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--name', type=str, default='script', help='experiment_name')
    argparser.add_argument('--lr', type=float, default=1e-5, help='lr')
    argparser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    argparser.add_argument('--seed', type=int, default=42, help='random seed')
    argparser.add_argument('--randaug', type=float, default=0.05, help='randaug proba')
    return argparser.parse_args()

args = parse_args()
exp_name = args.name
dropout = args.dropout
seed = args.seed
randaug = args.randaug
lr = args.lr

print('Expname, dropout, seed, randaug, lr')
print(exp_name, dropout, seed, randaug, lr)

extra_tags = []

DEBUG = False
if DEBUG: extra_tags += ['debug']
k_folds = 2 if DEBUG else 5
n_epochs = 1 if DEBUG else 2.2

cfg = {
    "num_proc": 2,
    "aug_prob": randaug,
    "k_folds": k_folds,
    "max_length": 2048,
    "padding": False,
    "stride": 0,
    "data_dir": "../input/feedback-prize-effectiveness",
    "load_from_disk": None,
    "pad_multiple": 8,
    "model_name_or_path": "../output/HF-pret-3-fold0/checkpoint-23660/",
    "dropout": dropout,
    "trainingargs": {
        "output_dir": f"../output/{exp_name}",
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 4,
        "learning_rate": lr,
        "weight_decay": 0.01,
        "num_train_epochs": n_epochs,
        "warmup_ratio": 0.1,
        "optim": 'adamw_torch',
        "logging_steps": 25,
        "save_strategy": "steps",
        "save_steps": 25,
        "evaluation_strategy": "steps",
        "eval_steps": 25,
        "eval_delay": 600,
        "report_to": "wandb",
        "group_by_length": True,
        "save_total_limit": 1,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "seed": seed,
        "fp16": True,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 1,
    }
}

import re
import pickle
import codecs
import warnings
import logging
from functools import partial
from pathlib import Path
from itertools import chain
from text_unidecode import unidecode
from typing import Any, Optional, Tuple

import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, set_seed

from datasets import Dataset, load_from_disk

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end

def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def read_text_files(example, data_dir):
    
    id_ = example["essay_id"]
    
    with open(data_dir / "train" / f"{id_}.txt", "r") as fp:
        example["text"] = resolve_encodings_and_normalize(fp.read())
    
    return example

set_seed(cfg["trainingargs"]["seed"])

warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

essay_folds = pd.read_csv('../input/feedback-folds/df_folds.csv')
essay_folds.head()
essay_folds_dict = {x:y for x,y in zip(essay_folds.essay_id.values.tolist(), essay_folds.fold_k_5_seed_42.values.tolist())}

data_dir = Path(cfg["data_dir"])

if cfg["load_from_disk"]:
    if not cfg["load_from_disk"].endswith(".dataset"):
        cfg["load_from_disk"] += ".dataset"
    ds = load_from_disk(cfg["load_from_disk"])
    
    pkl_file = f"{cfg['load_from_disk'][:-len('.dataset')]}_pkl"
    with open(pkl_file, "rb") as fp: 
        grouped = pickle.load(fp)
        
    print("loading from saved files")
else:
    train_df = pd.read_csv(data_dir / "train.csv")
    
    if DEBUG: train_df = train_df.sample(n=100).reset_index(drop=True)
    
    text_ds = Dataset.from_dict({"essay_id": train_df.essay_id.unique()})
    
    text_ds = text_ds.map(
        partial(read_text_files, data_dir=data_dir),
        num_proc=cfg["num_proc"],
        batched=False,
        desc="Loading text files",
    )
    
    text_df = text_ds.to_pandas()
    
    train_df["discourse_text"] = [
        resolve_encodings_and_normalize(x) for x in train_df["discourse_text"]
    ]
    
    train_df = train_df.merge(text_df, on="essay_id", how="left")
    
disc_types = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
]

cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}

label2id = {
    "Adequate": 0,
    "Effective": 1,
    "Ineffective": 2,
}

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
tokenizer.add_special_tokens(
    {"additional_special_tokens": list(cls_tokens_map.values())+list(end_tokens_map.values())}
)

cls_id_map = {
    label: tokenizer.encode(tkn)[1]
    for label, tkn in cls_tokens_map.items()
}


def find_positions(example):

    text = example["text"][0]
    
    # keeps track of what has already
    # been located
    min_idx = 0
    
    # stores start and end indexes of discourse_texts
    idxs = []
    
    for dt in example["discourse_text"]:
        # calling strip is essential
        matches = list(re.finditer(re.escape(dt.strip()), text))
        
        # If there are multiple matches, take the first one
        # that is past the previous discourse texts.
        if len(matches) > 1:
            for m in matches:
                if m.start() >= min_idx:
                    break
        # If no matches are found
        elif len(matches) == 0:
            idxs.append([-1]) # will filter out later
            continue  
        # If one match is found
        else:
            m = matches[0]
            
        idxs.append([m.start(), m.end()])

        min_idx = m.start()

    return idxs

def tokenize(example):
    example["idxs"] = find_positions(example)

    text = example["text"][0]
    chunks = []
    labels = []
    prev = 0

    zipped = zip(
        example["idxs"],
        example["discourse_type"],
        example["discourse_effectiveness"],
    )
    for idxs, disc_type, disc_effect in zipped:
        # when the discourse_text wasn't found
        if idxs == [-1]:
            continue

        s, e = idxs

        # if the start of the current discourse_text is not 
        # at the end of the previous one.
        # (text in between discourse_texts)
        if s != prev:
            chunks.append(text[prev:s])
            prev = s

        # if the start of the current discourse_text is 
        # the same as the end of the previous discourse_text
        if s == prev:
            chunks.append(cls_tokens_map[disc_type])
            chunks.append(text[s:e])
            chunks.append(end_tokens_map[disc_type])
        
        prev = e

        labels.append(label2id[disc_effect])

    tokenized = tokenizer(
        " ".join(chunks),
        padding=False,
        truncation=True,
        max_length=cfg["max_length"],
        add_special_tokens=True,
    )
    
    # at this point, labels is not the same shape as input_ids.
    # The following loop will add -100 so that the loss function
    # ignores all tokens except CLS tokens

    # idx for labels list
    idx = 0
    final_labels = []
    for id_ in tokenized["input_ids"]:
        # if this id belongs to a CLS token
        if id_ in cls_id_map.values():
            final_labels.append(labels[idx])
            idx += 1
        else:
            # -100 will be ignored by loss function
            final_labels.append(-100)
    
    tokenized["labels"] = final_labels

    return tokenized

def add_fold(example):
    example["fold"] = essay_folds_dict[example["essay_id"]]
    return example

# I frequently restart my notebook, so to reduce time
# you can set this to just load the tokenized dataset from disk.
# It gets loaded in the 3rd code cell, but a check is done here
# to skip tokenizing
if cfg["load_from_disk"] is None:

    # make lists of discourse_text, discourse_effectiveness
    # for each essay
    grouped = train_df.groupby(["essay_id"]).agg(list)

    ds = Dataset.from_pandas(grouped)

    ds = ds.map(
        tokenize,
        batched=False,
        num_proc=cfg["num_proc"],
        desc="Tokenizing",
    )

    save_dir = f"{cfg['trainingargs']['output_dir']}"
    ds.save_to_disk(f"{save_dir}.dataset")
    with open(f"{save_dir}_pkl", "wb") as fp:
        pickle.dump(grouped, fp)
    print("Saving dataset to disk:", cfg['trainingargs']['output_dir'])


ds = ds.map(add_fold)

bad_matches = []
cls_ids = set(list(cls_id_map.values()))
for id_, l, ids, dt in zip(ds["essay_id"], ds["labels"], ds["input_ids"], grouped.discourse_text):
    
    # count number of labels (ignoring -100)
    num_cls_label = sum([x!=-100 for x in l])
    # count number of cls ids
    num_cls_id = sum([x in cls_ids for x in ids])
    # true number of discourse_texts
    num_dt = len(dt)
    
    if num_cls_label != num_dt or num_cls_id != num_dt:
        bad_matches.append((id_, l, ids, dt))
        
print("Num bad matches", len(bad_matches))

import math
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import softmax_backward_data
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model

special_tokens = [tokenizer.encode(tkn)[1] for tkn in list(cls_tokens_map.values())+list(end_tokens_map.values())] + [0,1,2]

from typing import List, Dict, Any

def random_mask_data_collator(features: List[Dict[str, Any]], mlm_probability=cfg["aug_prob"]) -> Dict[str, Any]:
    
    label_pad_token_id = -100
    label_name = "label" if "label" in features[0].keys() else "labels"
    labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
    batch = tokenizer.pad(
        features,
        padding=True,
        max_length=cfg["max_length"],
        pad_to_multiple_of=cfg["pad_multiple"],
        # Conversion to tensors will fail if we have labels as they are not of the same length yet.
        return_tensors="pt" if labels is None else None,
    )
    
    sequence_length = torch.tensor(batch["input_ids"]).shape[1]
    padding_side = tokenizer.padding_side
    if padding_side == "right":
        batch[label_name] = [
            list(label) + [label_pad_token_id] * (sequence_length - len(label)) for label in labels
        ]
    else:
        batch[label_name] = [
            [label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
        ]

    batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
    
    probability_matrix = torch.full(batch['input_ids'].shape, mlm_probability)
    special_tokens_mask = [[
        1 if x in special_tokens else 0 for x in row.tolist() 
    ] for row in batch['input_ids']]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    batch['input_ids'][masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return batch

import gc
import torch
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.checkpoint import checkpoint
import wandb

default_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, pad_to_multiple_of=cfg["pad_multiple"], padding=True
)

from transformers import AdamW
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
from torch import nn

from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
import datasets
from transformers.file_utils import is_datasets_available

class MyTrainer(Trainer): 
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=default_collator,   #KEY CHANGE = default data collator for eval!
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=default_collator,   #KEY CHANGE = default data collator for eval!
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters)\
                               and ('deberta' in n)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters)\
                               and ('deberta' not in n)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 5,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters)\
                              and ('deberta' in n)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters)\
                              and ('deberta' not in n)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 5,
                },
            ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

import gc
import torch
from torch.utils.checkpoint import checkpoint
import wandb

args = TrainingArguments(**cfg["trainingargs"])

# if using longformer pad to multiple of 512
# for others pad to multiple of 8

collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, pad_to_multiple_of=cfg["pad_multiple"], padding=True
)

output = args.output_dir
for fold in range(cfg["k_folds"]):
    
    args.output_dir = f"{output}-fold{fold}"
    
    model_config = AutoConfig.from_pretrained(
        cfg["model_name_or_path"],
    )
    model_config.update(
        {
            "num_labels": 3,
            "cls_tokens": list(cls_id_map.values()),
            "label2id": label2id,
            "id2label": {v:k for k, v in label2id.items()},
            "hidden_dropout_prob": dropout,
            "attention_probs_dropout_prob": dropout,
        }
    )
    
    model = AutoModelForTokenClassification.from_pretrained(cfg["model_name_or_path"], config=model_config)

    # need to resize embeddings because of added tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # split dataset to train and eval
    keep_cols = {"input_ids", "attention_mask", "labels"}
    train_dataset = ds.filter(lambda example: example["fold"] != fold).remove_columns([c for c in ds.column_names if c not in keep_cols])
    eval_dataset = ds.filter(lambda example: example["fold"] == fold).remove_columns([c for c in ds.column_names if c not in keep_cols])
        
    wandb.init(project="fbck", 
           name=f"{exp_name}_fold_{fold}",
           tags=["HF", f"fold_{fold}"]+extra_tags,
           group=f"{exp_name}")
    
    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=random_mask_data_collator,
    )
    
    trainer.train()
    wandb.finish()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


import json
from pathlib import Path
best_metrics = []
best_checkpoints = []

for fold in range(5):
    folder = Path(f"../output/{exp_name}-fold{fold}")
    checkpoint = sorted(list(folder.glob("checkpoint*")))[-1]
    with open(checkpoint/"trainer_state.json", "r") as fp:
        data = json.load(fp)
        best_metrics.append(data["best_metric"])
        best_checkpoints.append(data["best_model_checkpoint"])

print('*******************')
print(exp_name)
print(best_metrics)
average = sum(best_metrics)/len(best_metrics)
print(average)
print('*******************')


