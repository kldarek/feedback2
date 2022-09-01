# deb619.py

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
import shutil

from torch.utils.data import DataLoader, Dataset
import datasets, transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset


import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CFG:
    num_workers=1
    path="../input/feedback-deberta-large-051/"
    config_path=path+'config.pth'
    model="microsoft/deberta-large"
    batch_size=16
    fc_dropout=0.2
    target_size=3
    max_len=512
    seed=42
    n_fold=4
    trn_fold=[i for i in range(n_fold)]
    gradient_checkpoint=False

# ====================================================
# Utils
# ====================================================

def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def get_score(y_true, y_pred):
    y_pred = softmax(y_pred)
    score = log_loss(y_true, y_pred)
    return round(score, 5)

INPUT_DIR = "../input/feedback-prize-effectiveness/"
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
test['essay_text']  = test['essay_id'].apply(lambda x: get_essay(x, is_train=False))


# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.path + 'tokenizer')
CFG.tokenizer = tokenizer

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

test['discourse_text'] = test['discourse_text'].apply(lambda x : resolve_encodings_and_normalize(x))
test['essay_text'] = test['essay_text'].apply(lambda x : resolve_encodings_and_normalize(x))

SEP = tokenizer.sep_token
test['text'] = test['discourse_type'] + ' ' + test['discourse_text'] + SEP + test['essay_text']
test['label'] = np.nan

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df['text'].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = self.cfg.tokenizer.encode_plus(
                        self.text[item],
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.cfg.max_len
                    )
        samples = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
        }

        if 'token_type_ids' in inputs:
            samples['token_type_ids'] = inputs['token_type_ids']
        
        return samples

class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

collate_fn = Collate(CFG.tokenizer, isTrain=False)

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    
class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return mean_max_embeddings

    
class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out
    
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

# ====================================================
# Model
# ====================================================
from torch.cuda.amp import autocast
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        
        # gradient checkpointing
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()
            print(f"Gradient Checkpointing: {self.model.is_gradient_checkpointing}")
            
        
        # self.pooler = MeanPooling()
        
        self.bilstm = nn.LSTM(self.config.hidden_size, (self.config.hidden_size) // 2, num_layers=2, 
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.cfg.target_size)
            # nn.Linear(256, self.cfg.target_size)
        )
        
        

    def loss(self, outputs, targets):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, targets)
        return loss
    
    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        # print(outputs)
        # print(targets)
        mll = log_loss(
            targets.cpu().detach().numpy(),
            softmax(outputs.cpu().detach().numpy()),
            labels=[0, 1, 2],
        )
        return mll
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)
        
        # LSTM/GRU header
#         all_hidden_states = torch.stack(transformer_out[1])
#         sequence_output = self.pooler(all_hidden_states)
        
        # simple CLS
        sequence_output = transformer_out[0][:, 0, :]

        
        # Main task
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        if targets is not None:
            metric = self.monitor_metrics(logits, targets)
            return logits, metric
        
        return logits, 0.

# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for data in tk0:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        with torch.no_grad():
            y_preds, _ = model(ids, mask)
        y_preds = softmax(y_preds.to('cpu').numpy())
        preds.append(y_preds)
    predictions = np.concatenate(preds)
    return predictions

deberta_predictions = []
test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         collate_fn=collate_fn,
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

deberta_predictions = []
for fold in CFG.trn_fold:
    print("Fold {}".format(fold))

    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    deberta_predictions.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()

predictions = np.mean(deberta_predictions, axis=0)

submission['Ineffective'] = predictions[:, 0]
submission['Adequate'] = predictions[:, 1]
submission['Effective'] = predictions[:, 2]

submission.to_csv('submission1.csv', index=False)