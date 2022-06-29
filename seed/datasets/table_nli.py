from dataclasses import dataclass, field
import functools
from pathlib import Path
from datasets import (
    Array2D,
    ClassLabel,
    Features,
    Sequence,
    Value,
    Dataset,
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


class TableNLIUltis():
    def __init__(self, dataset):
        self.dataset = dataset
    

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.dataset, item)

    @staticmethod
    def from_jsonlines(file_path, cache_dir=None, *args, **kwargs):
        return load_dataset("json", data_files=file_path, cache_dir=cache_dir, *args, **kwargs).filter(
                lambda x: len(x["highlighted_cells"]) > 0 and x["table"] != "[]"
            )
    
    @staticmethod
    def filter_main_row(dataset, *args, **kwargs):
        def filter(obj):
            table = pd.DataFrame(json.loads(obj["table"]))
            rows = [x[0] for x in obj["highlighted_cells"]]
            most_dominant_row = max(set(rows), key=rows.count)
            cols = [x[1] for x in obj["highlighted_cells"] if x[0] == most_dominant_row]

            try:
                obj["table"] = (
                    table.iloc[[most_dominant_row], cols].reset_index()
                ).to_json(orient="records")
                return obj
            except:
                return obj

        return dataset.map(filter, *args, **kwargs)

    @staticmethod
    def tabularize(dataset, *args, **kwargs):
        def tabularize(item):
            item["df"] = pd.DataFrame(json.loads(item["table"]))
            return item
        return dataset.map(tabularize, *args, **kwargs)

    @staticmethod
    def to_infotab(dataset):
        return dataset.dataset.map(
            lambda example, idx: {
                "table_id": idx,
                'annotator_id': idx,
                "hypothesis": example["sentence"],
                "table": example["table"],
                "label": example["label"],
                "title": example["table_page_title"],
            },
            with_indices=True
        )
