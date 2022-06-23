from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
import jsonlines
import orjson as json

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


def get_offset(table):
    offset = table.columns[0].count("'") // 2
    if offset == 1:
        offset = table.columns[0].count('"') // 2
    if offset == 0:
        offset = 1
    return offset


class TableNLIDataset:
    def __init__(self, data=None):
        if data is None:
            self.data = []
        else:
            self.data = data

    @staticmethod
    def from_csv(file_path, table_csv_path, filter_cell=False):
        dataset = TableNLIDataset()
        tables_df = pd.read_csv(file_path)
        for idx, obj in tables_df.iterrows():
            table = pd.read_csv(Path(table_csv_path) / f"{obj['index']}.csv")

            if filter_cell:
                rows = [x[0] for x in obj["highlighted_cells"]]
                most_dominant_row = max(set(rows), key=rows.count)
                cols = [x[1] for x in obj["highlighted_cells"] if x[0] == most_dominant_row]

                try:
                    table = table.iloc[[most_dominant_row - get_offset(table)], cols].reset_index()
                except:
                    continue

            dataset.data.append(
                TableNLIExample(
                    table=table,
                    title=obj["title"],
                    label=obj["label"] if "label" in obj else 1.0,
                    sentence=obj["sentence"],
                )
            )
        return dataset


    def from_jsonlines(file_path, filter_cell=False):
        dataset = TableNLIDataset()
        with jsonlines.open(file_path, "r") as reader:
            for obj in reader:
                if obj is None:
                    continue
                table = pd.DataFrame(json.loads(obj["table"])).fillna("").astype(str)

                if filter_cell:

                    rows = [x[0] for x in obj["highlighted_cells"]]
                    most_dominant_row = max(set(rows), key=rows.count)
                    cols = [x[1] for x in obj["highlighted_cells"] if x[0] == most_dominant_row]

                    try:
                        table = table.iloc[[most_dominant_row - get_offset(table)], cols].reset_index()
                    except:
                        continue

                dataset.data.append(
                    TableNLIExample(
                        table=table,
                        label=obj["label"] if "label" in obj else 1.0,
                        sentence=obj["sentence"] if "sentence" in obj else "",
                        metadata={"page": obj["table_page_title"], "section": obj["table_section_title"]},
                    )
                )
        return dataset

    def to_infotab(self):
        infotab_data = []
        id = 0
        for example in self.data:
            infotab_data.append(InfotabExample(
                table_id=id,
                hypothesis=example.sentence,
                table=example.table,
                label=example.label,
                title=example.metadata["page"],
            ))
        return infotab_data

    def __getitem__(self, i):
        return self.data[i]
