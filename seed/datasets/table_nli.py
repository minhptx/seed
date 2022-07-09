import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import jsonlines
import orjson as json
import pandas as pd

from datasets import Dataset, load_dataset


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


class TableNLIUltis:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.dataset, item)

    @staticmethod
    def from_jsonlines(
        file_path,
        cache_dir=None,
        filter_row=True,
        columns=["sentence", "table", "label", "table_page_title", "highlighted_cells"],
        output_format="tapex",
        *args,
        **kwargs
    ):
        df = pd.DataFrame(
            list(jsonlines.open(file_path)),
            columns=columns,
        )
        dataset = Dataset.from_pandas(df)
        dataset = load_dataset(
            "json", data_files=file_path, cache_dir=cache_dir, *args, **kwargs
        ).filter(lambda x: len(x["highlighted_cells"]) > 0 and x["table"] != "[]")
        if filter_row:
            dataset = TableNLIUltis.filter_main_row(dataset)
        if output_format == "infotab":
            dataset = TableNLIUltis.to_infotab(dataset)
        return dataset

    @staticmethod
    def filter_main_row(dataset, *args, **kwargs):
        def filter(obj):
            table = pd.DataFrame(json.loads(obj["table"]))
            rows = [x[0] for x in obj["highlighted_cells"]]
            most_dominant_row = max(set(rows), key=rows.count)
            cols = [x[1] for x in obj["highlighted_cells"] if x[0] == most_dominant_row]

            try:
                obj["table"] = table.iloc[[most_dominant_row], cols].reset_index().to_json(orient="records")
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
        return dataset.map(
            lambda example, idx: {
                "table_id": idx,
                "annotator_id": idx,
                "hypothesis": example["sentence"],
                "table": example["table"],
                "label": example["label"],
                "title": example["table_page_title"],
            },
            with_indices=True,
        )
