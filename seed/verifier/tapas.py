import os
import sys

import jsonlines
import orjson as json
import pandas as pd
import torch
import tqdm
from accelerate import Accelerator
from datasets import load_metric
from transformers import (
    HfArgumentParser,
    TapasConfig,
    TapasForSequenceClassification,
    TapasTokenizer,
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

    def train(self, train_config):


def process_table(items):
    for item in items:
        encoding = tokenizer(
            table=item["table"],
            queries=str(item["sentence"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        token_type_ids = encoding["token_type_ids"]
        token_type_ids[token_type_ids > 127] = 127
        token_type_ids[token_type_ids > -128] = -128


        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = (item["label"])
    return encoding


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    metric = load_metric("accuracy")

    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
    config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
    model = TapasForSequenceClassification.from_pretrained(
        "google/tapas-base-finetuned-wtq", config=config
    )
    model.num_labels = 2

    train_df = pd.DataFrame(
        list(
            jsonlines.open(
                f"data/totto_data/processed_train_data_tabularized_induced.jsonl"
            )
        )
    )

    dev_df = pd.DataFrame(
        list(
            jsonlines.open(
                f"data/totto_data/processed_dev_data_tabularized_induced.jsonl"
            )
        )
    )

    train_dataset = TableDataset(train_df, tokenizer)
    dev_dataset = TableDataset(dev_df, tokenizer)

    for param in model.tapas.parameters():
        param.requires_grad = False

    start_epoch = 0
    num_epochs = args.num_train_epochs

    train_dl = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    eval_dl = torch.utils.data.DataLoader(dev_dataset, shuffle=False)

    accelerator = Accelerator()

    optimizer = optim.AdamW(params=model.parameters(), lr=1e-5)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dl, eval_dl
    )

    num_train_steps = len(train_dl) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_train_steps,
    )

    gradient_accumulation_steps = 1
    progress_bar = tqdm(range(num_train_steps))

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            running_loss += loss.item()
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
        progress_bar.set_description(
            f"Loss value: {running_loss / len(train_dataloader)}"
        )

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        accelerator.print(f"Eval epoch {epoch}:", eval_metric)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(
            unwrapped_model.state_dict(), f"models/tapas_base/tapas_{epoch}.model"
        )
