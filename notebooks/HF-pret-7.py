exp_name = "HF-pret-7"
extra_tags = ['pretraining']

DEBUG = False
if DEBUG: extra_tags += ['debug']
n_epochs = 1 if DEBUG else 10

cfg = {
    "num_proc": 2,
    "k_folds": k_folds,
    "max_length": 2048,
    "padding": False,
    "stride": 0,
    "data_dir": "../input/fbck2021",
    "load_from_disk": None,
    "pad_multiple": 8,
    "model_name_or_path": "microsoft/deberta-v3-large",
    "dropout": 0.1,
    "trainingargs": {
        "output_dir": f"../output/{exp_name}",
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "num_train_epochs": n_epochs,
        "warmup_ratio": 0.1,
        "optim": 'adamw_torch',
        "logging_steps": 50,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "report_to": "wandb",
        "group_by_length": True,
        "save_total_limit": 1,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "seed": 42,
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
    train_df = pd.read_csv("../input/2021_data_for_pseudo_mlm.csv")
    
    if DEBUG: train_df = train_df[:200]
    
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

end_id_map = {
    label: tokenizer.encode(tkn)[1]
    for label, tkn in end_tokens_map.items()
}



special_tokens = list(set(cls_id_map.values())) + list(set(end_id_map.values()))

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
    
    # tokenized["labels"] = final_labels

    return tokenized

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
    

# basic kfold 
def get_folds(df, k_folds=k_folds):

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    return [
        val_idx
        for _, val_idx in kf.split(df)
    ]

fold_idxs = get_folds(ds["discourse_id"], cfg["k_folds"])

# add "special_tokens_mask" to dataset .... and remove labels from it...

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.data.data_collator import DataCollatorForLanguageModeling

class MyMLMCollator(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        for tok in special_tokens: 
            probability_matrix = torch.where(labels == tok, 1., probability_matrix)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

import gc
import torch
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.checkpoint import checkpoint
import wandb
from transformers import AutoModelForMaskedLM


args = TrainingArguments(**cfg["trainingargs"])

# if using longformer pad to multiple of 512
# for others pad to multiple of 8

collator = MyMLMCollator(
    tokenizer=tokenizer, pad_to_multiple_of=cfg["pad_multiple"]
)

output = args.output_dir
for fold in range(1):
    
    args.output_dir = f"{output}-fold{fold}"
    
    model_config = AutoConfig.from_pretrained(
        cfg["model_name_or_path"],
    )
    model_config.update(
        {
            "cls_tokens": list(cls_id_map.values()),
        }
    )
    
    model = AutoModelForMaskedLM.from_pretrained(cfg["model_name_or_path"], config=model_config)
    
    # need to resize embeddings because of added tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # split dataset to train and eval
    keep_cols = {"input_ids", "attention_mask"}
    train_idxs = list(chain(*[i for f,i in enumerate(fold_idxs) if f!= fold]))
    train_dataset = ds.select(train_idxs).remove_columns([c for c in ds.column_names if c not in keep_cols])
    eval_dataset = ds.select(fold_idxs[fold]).remove_columns([c for c in ds.column_names if c not in keep_cols])
        
    wandb.init(project="fbck", 
           name=f"{exp_name}_fold_{fold}",
           tags=["HF", f"fold_{fold}"]+extra_tags,
           group=f"{exp_name}")
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    trainer.train()
    wandb.finish()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()



