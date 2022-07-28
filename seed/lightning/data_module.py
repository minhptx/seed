from typing import Optional

import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import multiprocessing
import pandas as pd
import json
from pathlib import Path
from json.decoder import JSONDecodeError


class PLDataModule(LightningDataModule):

    dataset_text_field_map = {
        "clapika2010/totto": ["sentence", "table"],
        "clapika2010/totto2": ["sentence", "table"],
        "clapika2010/infotab": ["sentence", "table"],
        "clapika2010/infotab2": ["sentence", "table"],
        "clapika2010/totto_triplet": ["anchor", "positive", "negative"],
    }

    dataset_remove_columns = {
        "clapika2010/totto": ["label"],
        "clapika2010/totto2": ["label"],
        "clapika2010/infotab": ["label"],
        "clapika2010/infotab2": ["label"],
        "clapika2010/totto_triplet": [],
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        dataset: str = "clapika2010/totto",
        processed: bool = True,
        num_labels: int = 2,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        test_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset + "_processed" if processed else dataset
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.dataset_text_field_map[dataset]
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )
        self.eval_splits = eval_splits
        self.test_splits = test_splits

    def setup(self, stage: str):
        print("Rank Setup", self.trainer.local_rank)
        self.dataset = datasets.load_dataset(self.dataset_name)
        self.dataset = self.dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=self.dataset_remove_columns[self.dataset_name],
            num_proc=20,
            cache_file_names={
                x: f".cache/huggingface/{self.dataset_name}/cache_{x}.arrow"
                for x in self.dataset
            },
        )
        for split in self.dataset.keys():
            self.columns = [
                c
                for c in self.dataset[split].column_names
                if c in self.loader_columns
                or c.replace("positive_", "") in self.loader_columns
                or c.replace("negative_", "") in self.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    def prepare_data(self):
        dataset = datasets.load_dataset(self.dataset_name)
        Path(f".cache/huggingface/{self.dataset_name}").mkdir(
            parents=True, exist_ok=True
        )
        dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=self.dataset_remove_columns[self.dataset_name],
            num_proc=20,
            cache_file_names={
                x: f".cache/huggingface/{self.dataset_name}/cache_{x}.arrow"
                for x in dataset
            },
        )
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset[self.eval_splits[0]],
                batch_size=self.eval_batch_size,
                num_workers=4,
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=4,
                )
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.test_splits) == 1:
            return DataLoader(
                self.dataset[self.test_splits[0]],
                batch_size=self.eval_batch_size,
                num_workers=4,
            )
        elif len(self.test_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=4,
                )
                for x in self.test_splits
            ]

    def convert_to_features(self, example_batch, indices=None):
        for key in self.text_fields:
            if not example_batch[key]:
                continue
            try:
                result = []
                for obj in example_batch[key]:
                    table = json.loads(obj)
                    table = pd.DataFrame(table)
                    for column in table.columns:
                        table[column] = table[column].apply(
                            lambda x: f"{column} : {' , '.join(x) if isinstance(x, list) else x}"
                        )

                    result.append(
                        self.tokenizer.sep_token.join(
                            table.apply(lambda x: " ; ".join(x), axis=1).values.tolist()
                        )
                    )
                example_batch[key] = result
            except JSONDecodeError:
                continue

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) == 3:
            texts_or_text_pairs = (
                list(
                    zip(
                        example_batch[self.text_fields[0]],
                        example_batch[self.text_fields[1]],
                    )
                ),
                list(
                    zip(
                        example_batch[self.text_fields[0]],
                        example_batch[self.text_fields[2]],
                    )
                ),
            )
        elif len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        if isinstance(texts_or_text_pairs, tuple):
            positives = self.tokenizer(
                texts_or_text_pairs[0],
                max_length=self.max_seq_length,
                padding=True,
                truncation=True,
            )
            negatives = self.tokenizer(
                texts_or_text_pairs[1],
                max_length=self.max_seq_length,
                padding=True,
                truncation=True,
            )
            result = {}
            for key in positives.data.keys():
                result[f"positive_{key}"] = positives[key]
                result[f"negative_{key}"] = negatives[key]
            return result
        else:
            features = self.tokenizer(
                texts_or_text_pairs,
                max_length=self.max_seq_length,
                padding=True,
                truncation=True,
            )

            # Rename label to labels to make it easier to pass to model forward

            features["labels"] = example_batch["label"]

            return features
