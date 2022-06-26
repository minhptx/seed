from dataclasses import dataclass, field
import functools
from pathlib import Path
from datasets import Array2D, ClassLabel, Features, Sequence, Value, load_dataset

import orjson as json
import pandas as pd


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


@dataclass
class TableNLIExample:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    label: bool = False
    sentence: str = ""
    metadata: dict = field(default_factory=dict)

    def get_row_values(self):
        return self.table.iloc[0, :].values.tolist()


class TableNLIData:
    def __init__(self, dataset):
        self.dataset = dataset

    def from_jsonlines(file_path):
        # features = Features({"table": Value(dtype='string'), "sentence": Value(dtype="string"), "label": ClassLabel(num_classes=2, names=[True, False]), "highlighted_cells": Array2D(shape=(None, 2), dtype='int32')})
        return TableNLIData(load_dataset("json", data_files=file_path)).map(
            lambda x: x.update({"table": pd.DataFrame(json.loads(x["table"]))})
        )

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

        return self.dataset.map(filter)

    def to_infotab(self):
        return self.dataset.map(
            lambda example: InfotabExample(
                table_id=id,
                hypothesis=example.sentence,
                table=example.table,
                label=example.label,
                title=example["table_page_tiltle"],
            )
        )

    def preprocess_with_func (self, func):
        return self.dataset.map(func)

    def __getitem__(self, i):
        return self.data[i]
