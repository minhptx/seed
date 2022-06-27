from dataclasses import dataclass, field
import functools
from pathlib import Path
from datasets import (
    Array2D,
    ClassLabel,
    Features,
    Sequence,
    Value,
    load_dataset,
    load_from_disk,
)

import orjson as json
import pandas as pd
import multiprocessing as mp

@dataclass
class InfotabExample:
    table_id: str = None
    hypothesis: str = None
    table: pd.DataFrame = None
    label: str = None
    title: str = None
    annotator_id: str = None

    def to_dict(self):
        return self.__dict__


class TableNLIData:
    def __init__(self, dataset):
        self.dataset = dataset

    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, **kwargs)
        return self

    @staticmethod
    def from_jsonlines(file_path):
        # features = Features({"table": Value(dtype='string'), "sentence": Value(dtype="string"), "label": ClassLabel(num_classes=2, names=[True, False]), "highlighted_cells": Array2D(shape=(None, 2), dtype='int32')})
        return TableNLIData(load_dataset("json", data_files=file_path).filter(
                lambda x: len(x["highlighted_cells"]) > 0
            ))

    @staticmethod
    def load_from_disk(file_path):
        return TableNLIData(load_from_disk(file_path))

    def save_to_disk(self, file_path):
        self.dataset.save_to_disk(file_path)

    def filter_main_row(self):
        def filter(obj):
            rows = [x[0] for x in obj["highlighted_cells"]]
            most_dominant_row = max(set(rows), key=rows.count)
            cols = [x[1] for x in obj["highlighted_cells"] if x[0] == most_dominant_row]

            try:
                obj["table"] = (
                    obj["table"].iloc[[most_dominant_row], cols].reset_index()
                )
                return obj
            except:
                return obj

        return self.map(filter)

    def tabularize(self):
        def tabularize(item):
            item["df"] = pd.DataFrame(json.loads(item["table"]))
            return item
        return self.map(tabularize)

    def to_infotab(self):
        i = 0

        def counter():
            i += 1
            return i

        return self.dataset.map(
            lambda example: {
                "table_id": counter(),
                "hypothesis": example.sentence,
                "table": example.table,
                "label": example.label,
                "title": example["table_page_tiltle"],
            }
        )

    def preprocess_with_func(self, func):
        return self.dataset.map(func, num_proc=mp.cpu_count())

    def __getitem__(self, i):
        return self.data[i]
