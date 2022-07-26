import collections
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
import copy
import jsonlines


import pandas as pd
from transformers import HfArgumentParser


@dataclass
class Argument:
    input_path: str = field(
        default="data/tabpert/initial_dataset/all_data/table_categories.tsv",
        metadata={"help": "Path to the input file"},
    )
    tables_path: str = field(
        default="data/tabpert/initial_dataset/all_data/tables",
        metadata={"help": "Path to the tables file"},
    )
    category_path: str = field(
        default="data/tabpert/initial_dataset/all_data/key_categories.json",
        metadata={"help": "Path to key category file"},
    )

    output_path: str = field(
        default="data/tabpert/processed/test.jsonl",
        metadata={"help": "Path to the output file"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser(Argument)
    args = parser.parse_args()

    table2category = {
        row["table_id"]: row["category"]
        for _, row in pd.read_csv(args.input_path, sep="\t").iterrows()
    }
    category2tables = collections.defaultdict(list)

    for table, category in table2category.items():
        category2tables[category].append(table)
    type2keys = json.load(open(args.category_path, "r"))
    key2type = {key: category for category, keys in type2keys.items() for key in keys}

    category_type2values = collections.defaultdict(list)
    table2valid_keys = collections.defaultdict(list)


    tables = []

    for table_path in Path(args.tables_path).glob("*.json"):
        table_id = table_path.stem
        table = json.load(open(table_path, "r"))
        tables.append(table)
        if len(table) == 0:
            continue
        if table_id not in table2category:
            continue
        category = table2category[table_id]

        for k, v in table.items():
            if k in key2type:
                category_type2values[f"{category}_{key2type[k]}"].extend(v)
                table2valid_keys[table_id].append(k)

    examples = []

    for table_path in Path(args.tables_path).glob("*.json"):
        table_id = table_path.stem
        table = json.load(open(table_path, "r"))
        new_table = copy.deepcopy(table)
        if table2valid_keys[table_id] == []:
            continue
        random_key = random.choice(table2valid_keys[table_id])
        print(random_key)
        random_value = random.choice(category_type2values[f"{table2category[table_id]}_{key2type[random_key]}"])

        new_table[random_key] = [random_value]
        examples.append({"table": table, "label": True, "counter_fact": "", "title": table["title"][0]})
        examples.append({"table": new_table, "label": False, "counter_fact": (None, None, table[random_key][0], random_value[0]), "title": table["title"][0]})

    with jsonlines.open(args.output_path, mode="w") as writer:
        for example in examples:
            writer.write(example)
