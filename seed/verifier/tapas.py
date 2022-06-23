import os
import sys

import jsonlines
import orjson as json
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from accelerate import Accelerator
from datasets import load_metric
from transformers import (
    HfArgumentParser,
    TapasConfig,
    TapasForSequenceClassification,
    TapasTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        table = pd.DataFrame(json.loads(item["table"])).astype(str)
        if len(table.columns) > 200:
            table = table.iloc[:, :199]

        encoding = self.tokenizer(
            table=table,
            queries=str(item["sentence"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        token_type_ids = encoding["token_type_ids"]
        token_type_ids[token_type_ids > 255] = 255

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = int(item["label"])
        return encoding

    def __len__(self):
        return len(self.data)



class TapasVerifier:
    def __init__(self) -> None:
        model_config = TapasConfig.from_pretrained("google/tapas-base", num_labels=2)
        self.model = TapasForSequenceClassification.from_pretrained(
            "google/tapas-base", config=model_config
        )
        # self.model.classifier = nn.Linear(model_config.hidden_size, 3)
        self.tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
        self.accelerator = Accelerator()
        

    def load(self, config):
        self.model, self.tokenizer = self.accelerator.prepare(
            self.model, self.tokenizer
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(config.model_path))

    