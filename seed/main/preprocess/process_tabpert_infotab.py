from dataclasses import dataclass, field
import json
from transformers import HfArgumentParser
import pandas as pd
from pathlib import Path
import collections
import jsonlines


@dataclass
class DataArguments:
    table_folder_path: str = field(default="data/tabpert/tables/")
    output_folder_path: str = field(default="data/tabpert/processed/")




if __name__ == "__main__":
    parser = HfArgumentParser(DataArguments)
    args = parser.parse_args_into_dataclasses()[0]

    name2table = {}

    for table_file in Path(args.table_folder_path).glob("*.json"):
        table_obj = json.load(table_file.open("r"))
        table = {x: [tuple(d)[0] for d in y] for x, y in table_obj.items()}
        table["title"] = table_obj["title"]
        name2table[table_file.stem] = table

    examples = []
    for name, table in name2table.items():
        example = {
            "sentence": "",
            "table": json.dumps([table]),
            "label": 0,
            "highlighted_cells": [[0, x] for x in range(len(table))],
            "table_page_title": table["title"][0],
        }
        examples.append(example)

    with jsonlines.open(args.output_folder_path + f"output.jsonl", "w") as writer:
        writer.write_all(examples)
