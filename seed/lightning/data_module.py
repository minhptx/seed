from typing import Optional

import datasets
from pytorch_lightning import (
    LightningDataModule
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import multiprocessing


class PLDataModule(LightningDataModule):

    dataset_text_field_map = {
        "clapika2010/totto": ["sentence", "table"],
        "clapika2010/infotab": ["sentence", "table"],
        "clapika2010/totto_triplet": ["anchor", "positive", "negative"],
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
            remove_columns=["label"],
            num_proc=20,
            cache_file_names = {"train": ".cache/huggingface/cache_train.arrow", "dev": ".cache/huggingface/cache_dev.arrow"}
        )
        for split in self.dataset.keys():
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)


    def prepare_data(self):
        dataset = datasets.load_dataset(self.dataset_name)
        dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=["label"],
            num_proc=20,
            cache_file_names = {"train": ".cache/huggingface/cache_train.arrow", "dev": ".cache/huggingface/cache_dev.arrow"}
        )
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset[self.eval_splits[0]],
                batch_size=self.eval_batch_size,
                num_workers=multiprocessing.cpu_count()
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=multiprocessing.cpu_count()
                )
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.test_splits) == 1:
            return DataLoader(
                self.dataset[self.test_splits[0]],
                batch_size=self.eval_batch_size,
                num_workers=multiprocessing.cpu_count()
            )
        elif len(self.test_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=multiprocessing.cpu_count()
                )
                for x in self.test_splits
            ]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        # print(features)
        return features
