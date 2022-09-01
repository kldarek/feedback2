DEBUG = False

cfg = {
    "num_proc": 2,
    "model_name_or_path": "../input/deberta-v3-large/deberta-v3-large",
    "data_dir": "../input/feedback-prize-effectiveness",
    "trainingargs": {
        "seed": 42,
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

train_df = pd.read_csv(data_dir / "test.csv")
train_df['discourse_effectiveness'] = 'Adequate'

if DEBUG: train_df = train_df[:100]

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

type2id = {'Lead': 0,
 'Position': 1,
 'Claim': 2,
 'Evidence': 3,
 'Counterclaim': 4,
 'Rebuttal': 5,
 'Concluding Statement': 6,
 'Other': 7}

label2id = {
    "Ineffective": 0,
    "Adequate": 1,
    "Effective": 2,
}

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])

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
            print('!!!! MISS !!!')
            print(dt.strip())
            print('!!here!!')
            print(text)
            print()
            continue  
        # If one match is found
        else:
            m = matches[0]
            
        idxs.append([m.start(), m.end()])

        min_idx = m.start()

    return idxs

def tokenize(example):
    example["idxs"] = find_positions(example)
    
    # print("New example")
    # print(example["idxs"])
    # print()

    text = example["text"][0]
    text = text.replace('\n', '|')
    chunks = []
    chunk_example = []
    chunk_idxs = []
    examples_classes = [type2id[disc_type] for disc_type in example["discourse_type"]]
    examples_scores = [label2id[disc_effect] for disc_effect in example["discourse_effectiveness"]]
    
    prev = 0

    zipped = zip(
        example["idxs"],
        example["discourse_type"],
        example["discourse_effectiveness"],
    )
    
    for idxs, disc_type, disc_effect in zipped:
        # when the discourse_text wasn't found
        if idxs == [-1]:
            chunk_idxs.append([-1])
            chunk_example.append(-1)
            chunks.append('')
            continue

        s, e = idxs

        # if the start of the current discourse_text is not 
        # at the end of the previous one.
        # (text in between discourse_texts)
        if s != prev:
            chunk_idxs.append([prev,s])
            chunk_example.append(-1)
            chunks.append(text[prev:s])
            prev = s

        # if the start of the current discourse_text is 
        # the same as the end of the previous discourse_text
        if s == prev:
            chunk_idxs.append([s,e])
            chunks.append(text[s:e])
            chunk_example.append(1)
        
        prev = e
        
    input_ids = [tokenizer.cls_token_id]
    token_class_labels = [-100]
    token_scores_labels = [-100]
    token_examples_mapping = [-100]
    
    assert len(examples_classes) == len(examples_scores) 
    assert len(chunks) == len(chunk_idxs) 
    assert len(examples_classes) == len(example["discourse_effectiveness"])

    i = 0
    
    for j, chunk in enumerate(chunks):
        chunk_ids = tokenizer(chunk, padding=False, truncation=False, add_special_tokens=False)
        chunk_ids = chunk_ids['input_ids']
        if len(chunk_ids) == 0: 
            assert chunk_example[j] == -1
            continue
            
        if chunk_example[j] == -1:
            input_ids.extend(chunk_ids)
            token_class_labels += [-100] * len(chunk_ids)
            token_scores_labels += [-100] * len(chunk_ids)
            token_examples_mapping += [-100] * len(chunk_ids)
        if chunk_example[j] == 1: 
            input_ids.extend(chunk_ids)
            token_class_labels += [examples_classes[i]] * len(chunk_ids)
            token_scores_labels += [examples_scores[i]] * len(chunk_ids)
            token_examples_mapping += [i] * len(chunk_ids)

            # DEBUG
            # print(i)
            # print('class', examples_classes[i])
            # print('score', examples_scores[i])
            # ss,ee = example["idxs"][i]
            # print(text[ss:ee])
            # print('***********************')
            # print(tokenizer.decode(chunk_ids))
            # print('***********************')
            # print()            
            # DEBUG
            
            i += 1
            
              
    # print(example["idxs"])
        
    # if (i+1 < len(example["idxs"])):
    #     print('ouch!!!!')
    #     for sss,eee in example["idxs"]:
    #           print(text[sss:eee])
        
    input_ids += [tokenizer.sep_token_id]
    token_class_labels += [-100]
    token_scores_labels += [-100]
    token_examples_mapping += [-100]
    attention_mask = [1] * len(input_ids)

    example['input_ids'] = input_ids
    example['attention_mask'] = attention_mask
    example['token_class_labels'] = token_class_labels
    example['token_scores_labels'] = token_scores_labels
    example['token_examples_mapping'] = token_examples_mapping
    example['examples_scores'] = examples_scores
    example['examples_classes'] = examples_classes
    
    return example


# make lists of discourse_text, discourse_effectiveness
# for each essay
grouped = train_df.groupby(["essay_id"]).agg(list)

ds = Dataset.from_pandas(grouped)

ds = ds.map(
    tokenize,
    batched=False,
)   

bad_matches = []
for id_, l, ids, dt, tem in zip(ds["essay_id"], ds["examples_scores"], ds["input_ids"], grouped.discourse_text,
                               ds["token_examples_mapping"]):
    
    # count number of labels (ignoring -100)
    num_cls_label = len(set(tem)) - 1
    # count number of cls ids
    num_cls_id = max(tem) + 1
    # true number of discourse_texts
    num_dt = len(dt)
    # print(num_cls_label, num_cls_id, num_dt)
    
    if num_cls_label != num_dt or num_cls_id != num_dt:
        bad_matches.append((id_, l, ids, dt))
        
print("Num bad matches", len(bad_matches))
# temp = train_df[train_df["essay_id"]==bad_matches[0][0]]
# temp_txt = temp.text.values[0]
# print(temp_txt)
# print("*"*100)
# print([x for x in temp.discourse_text if x.strip() not in temp_txt])

assert len(bad_matches) == 0

pdf = ds.to_pandas()
list_cols = ['input_ids',
       'attention_mask', 'token_class_labels', 'token_scores_labels',
       'token_examples_mapping', 'examples_scores', 'examples_classes']
for c in list_cols:
    pdf[c] = [x.tolist() for x in pdf[c].values]



######################
## TOKENIZATION #####
#####################

EXP = 'PL20'

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pandas as pd
import numpy as np
import random
import re
import itertools
import argparse

from torch.utils.data import Dataset
import spacy
import ast
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import pickle
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings
from torch.optim import Adam, SGD, AdamW
from pytorch_lightning.loggers import WandbLogger

pl.seed_everything(42, workers=True)

class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=1024, stage='train', rand_prob=0.1, lowup_proba=0.0, swap_proba=0.0):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        self.stage = stage
        self.rand_prob = rand_prob
        self.lowup_proba = lowup_proba
        self.swap_proba = swap_proba

        self.essay_id = df['essay_id'].values
        self.input_ids = df['input_ids'].values
        self.attention_mask = df['attention_mask'].values
        self.token_class_labels = df['token_class_labels'].values
        self.token_scores_labels = df['token_scores_labels'].values
        self.token_examples_mapping = df['token_examples_mapping'].values
        self.examples_scores = df['examples_scores'].values
        self.examples_classes = df['examples_classes'].values    
        
    def __getitem__(self, idx):
        essay_id = self.essay_id[idx]

        token_examples_mapping = self.token_examples_mapping[idx]
        examples_scores = self.examples_scores[idx]
        examples_classes = self.examples_classes[idx]

#         token_examples_mapping = torch.tensor(token_examples_mapping, dtype=torch.long)
#         examples_scores = torch.tensor(examples_scores + [-1] * (40 - len(examples_scores)), dtype=torch.long)
#         examples_classes = torch.tensor(examples_classes + [-1] * (40 - len(examples_classes)), dtype=torch.long)
        
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        token_class_labels = self.token_class_labels[idx]
        token_scores_labels = self.token_scores_labels[idx]

#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         attention_mask = torch.tensor(attention_mask, dtype=torch.long)
#         token_class_labels = torch.tensor(token_class_labels, dtype=torch.long)
#         token_scores_labels = torch.tensor(token_scores_labels, dtype=torch.long)

#         if self.stage == 'train':
#             ix = torch.rand(size=(len(input_ids),)) < self.rand_prob
#             input_ids[ix] = self.mask_token
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_class_labels": token_class_labels,
            "token_scores_labels": token_scores_labels,
            "token_examples_mapping": token_examples_mapping,
            "examples_scores": examples_scores,
            "examples_classes": examples_classes
        }

    def __len__(self):
        return len(self.df)

emb_dim = 64

class MyModule(pl.LightningModule):
    def __init__(self, lr, model_checkpoint, num_classes, num_classes_class, emb_dim):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.num_classes_class = num_classes_class
        self.emb_dim = emb_dim
        self.name = model_checkpoint
        self.pad_idx = 1 if "roberta" in self.name else 0
        config = AutoConfig.from_pretrained(model_checkpoint, output_hidden_states=True)
        self.longformer = AutoModel.from_pretrained(model_checkpoint, config=config)
        self.nb_features = config.hidden_size
        self.logits = nn.Linear(self.nb_features, num_classes)
        self.example_logits = nn.Linear(self.nb_features + self.emb_dim, num_classes)
        self.class_logits = nn.Linear(self.nb_features, num_classes_class)  
        transformers.logging.set_verbosity_error()
        self.embedding = nn.Embedding(num_classes_class, emb_dim, max_norm=True)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        print('Model Loaded!')
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.lr, 'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.lr, 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
                
    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, token_scores_labels, token_examples_mapping, \
        examples_scores, examples_classes, token_class_labels = \
            train_batch["input_ids"], train_batch["attention_mask"], train_batch["token_scores_labels"], \
            train_batch['token_examples_mapping'], train_batch['examples_scores'], \
            train_batch['examples_classes'], train_batch['token_class_labels']
        
        hidden_states = self.longformer(
            input_ids,
            attention_mask=attention_mask,
        )[-1]
        features = hidden_states[-1]
        logits = self.logits(features)
        class_logits = self.class_logits(features)
        loss = F.cross_entropy(logits.view(-1, self.num_classes), token_scores_labels.view(-1))
        class_loss = F.cross_entropy(class_logits.view(-1, self.num_classes_class), token_class_labels.view(-1))

        # Convert to examples loss
        bs, ml, nc1 = logits.shape
        
        batch_preds = []
        batch_targs = []
        
        for i in range(bs):
            example_preds = []
            example_targs = []
            num_examples = token_examples_mapping[i].max()
            assert examples_scores[i,num_examples] >= 0 # and examples_scores[i,num_examples+1] < 0 # truncation breaks this
            for j in range(num_examples + 1):
                indices = token_examples_mapping[i] == j
                fts = features[i][indices].mean(dim=0)
                class_idx = examples_classes[i,j]
                emb = self.embedding(class_idx)
                preds = self.example_logits(torch.cat([emb,fts]))
                example_preds.append(preds)
                example_targs.append(examples_scores[i,j].view(1))
                
            example_preds = torch.cat(example_preds, dim=0).view(-1, nc1)
            example_targs = torch.cat(example_targs, dim=0)
            batch_preds.append(example_preds)
            batch_targs.append(example_targs)
        
        batch_preds = torch.cat(batch_preds, dim=0).view(-1, nc1)
        batch_targs = torch.cat(batch_targs, dim=0)
        
        example_loss = F.cross_entropy(batch_preds, batch_targs)
        
#         if self.current_epoch == 0:
    
        total_loss = loss + class_loss + example_loss

        self.log('train_scores_loss', loss)
        self.log('train_classes_loss', class_loss)
        self.log('train_examples_loss', example_loss)
        self.log('train_total_loss', total_loss)

        return total_loss
        
    def validation_step(self, val_batch, batch_idx):
        input_ids, attention_mask, token_scores_labels, token_examples_mapping, \
        examples_scores, examples_classes, token_class_labels = \
            val_batch["input_ids"], val_batch["attention_mask"], val_batch["token_scores_labels"], \
            val_batch['token_examples_mapping'], val_batch['examples_scores'], val_batch['examples_classes'], \
            val_batch['token_class_labels']
        hidden_states = self.longformer(
            input_ids,
            attention_mask=attention_mask,
        )[-1]
        features = hidden_states[-1]
        logits = self.logits(features)
        class_logits = self.class_logits(features)
        y_pred = F.log_softmax(logits, dim=-1)                                                
        loss = F.cross_entropy(logits.view(-1, self.num_classes), token_scores_labels.view(-1))
        class_loss = F.cross_entropy(class_logits.view(-1, self.num_classes_class), token_class_labels.view(-1))
        self.log('val_loss', loss)
        self.log('val_class_loss', class_loss)
        return {"preds": y_pred,
                "logits": logits,
                "features": features,
                "val_losses": loss,
                "token_examples_mapping": token_examples_mapping,
                "examples_scores": examples_scores,
                "examples_classes": examples_classes}   
    
    def validation_epoch_end(self, validation_step_outputs):

        bs, ml, nc1 = validation_step_outputs[0]["preds"].shape
        ml2 = validation_step_outputs[0]["examples_scores"].shape[-1]
        all_preds = [x["logits"][0] for x in validation_step_outputs]
        all_features = [x["features"][0] for x in validation_step_outputs]
        all_mappings = [x["token_examples_mapping"][0] for x in validation_step_outputs]
        all_scores = [x["examples_scores"][0] for x in validation_step_outputs]
        all_classes = [x["examples_classes"][0] for x in validation_step_outputs]

        num_texts = len(all_scores)
        
        example_preds = []
        example_targs = []
        
        for i in range(num_texts):
            num_examples = all_mappings[i].max()
            for j in range(num_examples + 1):
                indices = all_mappings[i] == j
                fts = all_features[i][indices].mean(dim=0)
                class_idx = all_classes[i][j]
                emb = self.embedding(class_idx)
                preds = self.example_logits(torch.cat([emb,fts]))
                example_preds.append(preds)
                example_targs.append(all_scores[i][j].view(1))              
                
        example_preds = torch.cat(example_preds, dim=0).view(-1, nc1)
        example_targs = torch.cat(example_targs, dim=0)
        
        example_loss = F.cross_entropy(example_preds, example_targs)
        self.log('example_loss', example_loss)
        print(example_loss)
        
    def predict_step(self, val_batch, batch_idx):
        input_ids, attention_mask, token_scores_labels, token_examples_mapping, examples_scores, examples_classes = \
            val_batch["input_ids"], val_batch["attention_mask"], val_batch["token_scores_labels"], \
            val_batch['token_examples_mapping'], val_batch['examples_scores'], val_batch['examples_classes']
        hidden_states = self.longformer(
            input_ids,
            attention_mask=attention_mask,
        )[-1]
        features = hidden_states[-1]
        logits = self.logits(features)
        y_pred = F.softmax(logits, dim=-1)  
        
        bs, ml, nc1 = logits.shape
        ml2 = 40
        
        batch_preds = []
        batch_targs = []
        
        for i in range(bs):
            example_preds = []
            example_targs = []
            num_examples = token_examples_mapping[i].max()
            assert examples_scores[i,num_examples] >= 0 # and examples_scores[i,num_examples+1] < 0 # truncation breaks this
            for j in range(num_examples + 1):               
                indices = token_examples_mapping[i] == j
                fts = features[i][indices].mean(dim=0)
                class_idx = examples_classes[i,j]
                emb = self.embedding(class_idx)
                preds = self.example_logits(torch.cat([emb,fts]))
                example_preds.append(preds)
                example_targs.append(examples_scores[i,j].view(1))   
                
            example_preds = torch.cat(example_preds, dim=0).view(-1, nc1)
            example_targs = torch.cat(example_targs, dim=0)
            batch_preds.append(example_preds)
            batch_targs.append(example_targs)
        
        return batch_preds, batch_targs
        

class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["token_class_labels"] = [sample["token_class_labels"] for sample in batch]
        output["token_scores_labels"] = [sample["token_scores_labels"] for sample in batch]
        output["token_examples_mapping"] = [sample["token_examples_mapping"] for sample in batch]
        output["examples_scores"] = [sample["examples_scores"] for sample in batch]
        output["examples_classes"] = [sample["examples_classes"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        batch_max_ex = max([len(sco) for sco in output["examples_scores"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            output["token_class_labels"] = [s + (batch_max - len(s)) * [-100] for s in output["token_class_labels"]]
            output["token_scores_labels"] = [s + (batch_max - len(s)) * [-100] for s in output["token_scores_labels"]]
            output["token_examples_mapping"] = [s + (batch_max - len(s)) * [-1] for s in output["token_examples_mapping"]]
            output["examples_scores"] = [s + (batch_max_ex - len(s)) * [-1] for s in output["examples_scores"]]
            output["examples_classes"] = [s + (batch_max_ex - len(s)) * [-1] for s in output["examples_classes"]]

        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]
            output["token_class_labels"] = [(batch_max - len(s)) * [-100] + s for s in output["token_class_labels"]]
            output["token_scores_labels"] = [(batch_max - len(s)) * [-100] + s for s in output["token_scores_labels"]]
            output["token_examples_mapping"] = [(batch_max - len(s)) * [-1] + s for s in output["token_examples_mapping"]]
            output["examples_scores"] = [(batch_max_ex - len(s)) * [-1] + s for s in output["examples_scores"]]
            output["examples_classes"] = [(batch_max_ex - len(s)) * [-1] + s for s in output["examples_classes"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["token_class_labels"] = torch.tensor(output["token_class_labels"], dtype=torch.long)
        output["token_scores_labels"] = torch.tensor(output["token_scores_labels"], dtype=torch.long)
        output["token_examples_mapping"] = torch.tensor(output["token_examples_mapping"], dtype=torch.long)
        output["examples_scores"] = torch.tensor(output["examples_scores"], dtype=torch.long)
        output["examples_classes"] = torch.tensor(output["examples_classes"], dtype=torch.long)

        return output

from pytorch_lightning.callbacks import ModelCheckpoint

model_checkpoint = cfg["model_name_or_path"]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
collate_fn = Collate(tokenizer)
bs = 2
project = 'fbck'
seed = 42
randmask_proba = 0.0
lr = 2e-5
epochs = 1 if DEBUG else 2
num_classes = 3
num_classes_class = 8

valid_dataset = MyDataset(
    pdf,
    tokenizer,
    stage='valid',
    rand_prob=randmask_proba
)

val_loader = DataLoader(valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=4, pin_memory=True, drop_last=False)

import gc
gc.collect()

PATHS = [
            '../input/pl20f012-download/pytorch_model_0.ckpt',
            '../input/pl20fold1/feedback-epoch01-example_loss0.65.ckpt',
            '../input/pl20fold2/feedback-epoch01-example_loss0.65.ckpt',
            '../input/pl20fold3/feedback-epoch01-example_loss0.66.ckpt',
            '../input/pl20fold4/feedback-epoch01-example_loss0.64.ckpt'
]

preds_all = []

for PATH in PATHS:
    model = MyModule.load_from_checkpoint(PATH, lr=lr,
                     model_checkpoint=model_checkpoint, 
                     num_classes=num_classes,
                     num_classes_class=num_classes_class,
                     emb_dim=emb_dim)
    trainer = pl.Trainer(accelerator="gpu")
    predictions = trainer.predict(model, dataloaders=val_loader)
    preds = torch.cat([p for b in predictions for p in b[0]])
    preds_all.append(preds)
    del model
    del trainer
    del predictions
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()
    
preds = torch.stack(preds_all, dim=0).mean(dim=0)
# preds

disc_ids = [x for z in pdf['discourse_id'].values for x in z]

# torch.set_printoptions(precision=3, sci_mode=False)
preds_softmax = torch.softmax(preds, dim=1).numpy()

sub = pd.read_csv('../input/feedback-prize-effectiveness/sample_submission.csv')
sub[sub.columns[-3:]] = preds_softmax
sub['discourse_id'] = disc_ids
sub.to_csv('submission3.csv', index=False)