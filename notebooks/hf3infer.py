#hf3.py infer

cfg = {
    "num_proc": 2,
    "k_folds": 5,
    "max_length": 2048,
    "padding": False,
    "stride": 0,
    "data_dir": "../input/feedback-prize-effectiveness",
    "load_from_disk": None,
    "pad_multiple": 8,
    "model_name_or_path": "../input/deberta-v3-large/deberta-v3-large",
    "dropout": 0.1,
    "trainingargs": {
        "output_dir": f"../output",
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 1,
        "learning_rate": 9e-6,
        "weight_decay": 0.01,
        "num_train_epochs": 3,
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
    
    with open(data_dir / "test" / f"{id_}.txt", "r") as fp:
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
    train_df = pd.read_csv(data_dir / "test.csv")
            
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
#         example["discourse_effectiveness"],
    )
    for idxs, disc_type in zipped:
        
        disc_effect = 'Effective'
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

disc_ids = [x for z in ds['discourse_id'] for x in z]

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
# temp = train_df[train_df["essay_id"]==bad_matches[0][0]]
# temp_txt = temp.text.values[0]
# print(temp_txt)
# print("*"*100)
# print([x for x in temp.discourse_text if x.strip() not in temp_txt])

assert len(bad_matches) == 0

import gc
import torch
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.checkpoint import checkpoint
import wandb

args = TrainingArguments(**cfg["trainingargs"])

# if using longformer pad to multiple of 512
# for others pad to multiple of 8

collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, pad_to_multiple_of=cfg["pad_multiple"], padding=True
)

output = args.output_dir

fold_preds = []

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
        }
    )
    
    model = AutoModelForTokenClassification.from_pretrained(cfg["model_name_or_path"], config=model_config)
    
    # need to resize embeddings because of added tokens
    model.resize_token_embeddings(len(tokenizer))
    
    PATH = f'../input/hf-3-download/pytorch_model_{fold}.bin'
    
    model.load_state_dict(torch.load(PATH))
    
    # split dataset to train and eval
    keep_cols = {"input_ids", "attention_mask", "labels"}
    test_dataset = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    preds = trainer.predict(test_dataset)
    fold_preds.append(preds.predictions)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

import numpy as np

len(fold_preds), fold_preds[0].shape

preds = np.stack(fold_preds).mean(axis=0)

preds_torch = torch.tensor(preds, dtype=torch.float32)
preds_torch.shape

# tokenizer = AutoTokenizer.from_pretrained('../input/hffbcktokenizer')

all_preds = []

for i in range(len(test_dataset)):
    indices = np.array(test_dataset[i]['labels']) == 1
    mypreds = preds_torch[i][:len(indices),:][indices]
    mypreds = torch.nn.functional.softmax(mypreds, dim=-1)
    all_preds.append(mypreds)
    
all_preds = torch.cat(all_preds, dim=0).numpy()
all_preds.shape

sub = pd.read_csv('../input/feedback-prize-effectiveness/sample_submission.csv')

sub[sub.columns[-3:]] = all_preds[:,[2,0,1]]

sub['discourse_id'] = disc_ids

sub.to_csv('submission2.csv', index=False)

