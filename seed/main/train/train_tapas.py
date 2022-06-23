import json
import os
import sys
from dataclasses import dataclass, field

import datasets
import jsonlines
import pandas as pd
import torch
import torch.optim as optim
import wandb
from accelerate import Accelerator
from datasets import Dataset, load_metric
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    TapasConfig,
    TapasForSequenceClassification,
    TapasTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

metric = load_metric("accuracy")
pd.set_option("mode.chained_assignment", "raise")


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


@dataclass
class DataArguments:
    train_file: str = field(
        default="data/train.json",
        metadata={"help": "The path to train data file"},
    )
    dev_file: str = field(
        default="data/dev.json",
        metadata={"help": "The path to dev data file"},
    )
    test_file: str = field(
        default="data/test.json",
        metadata={"help": "The path to test data file"},
    )


if __name__ == "__main__":
    wandb.init()
    parser = HfArgumentParser((TrainingArguments, DataArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, data_args = parser.parse_args_into_dataclasses()
    metric = load_metric("accuracy")

    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
    config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
    model = TapasForSequenceClassification.from_pretrained(
        "google/tapas-base-finetuned-wtq", config=config
    )
    model.num_labels = 2

    train_df = pd.DataFrame(list(jsonlines.open(data_args.train_file)))[:100].fillna(1.0)

    dev_df = pd.DataFrame(list(jsonlines.open(data_args.dev_file)))[:100].fillna(1.0)

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
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_train_steps,
    )

    gradient_accumulation_steps = 1
    progress_bar = tqdm(range(int(num_train_steps)))

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
        all_predictions = []
        all_labels = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
            all_predictions.extend(accelerator.gather(predictions))
            all_labels.extend(accelerator.gather(batch["labels"]))

        dev_acc = metric.compute()
        accelerator.print(f"Eval epoch {epoch}:", dev_acc)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        model.save_pretrained(os.path.join(args.output_dir, str(epoch + 1)))

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss": running_loss,
                "dev_accuracy": dev_acc,
                "dev_gold": all_labels,
                "dev_pred": all_predictions,
                "test_accuracy": dev_acc,
                "test_gold": all_labels,
                "test_pred": all_predictions,
            }
        )
