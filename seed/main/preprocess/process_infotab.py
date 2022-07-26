from dataclasses import dataclass, field
import json
from transformers import HfArgumentParser
import pandas as pd
from pathlib import Path
import collections
import jsonlines


@dataclass
class DataArguments:
    csv_folder_path: str = field(default="data/infotab/maindata/")
    table_folder_path: str = field(default="data/infotab/tables/json/")
    output_folder_path: str = field(default="data/infotab/processed/")




if __name__ == "__main__":
    parser = HfArgumentParser(DataArguments)
    args = parser.parse_args_into_dataclasses()[0]

    csv_folder_path = Path(args.csv_folder_path)
    name2table = {}

    for table_file in Path(args.table_folder_path).glob("*.json"):
        table_obj = json.load(table_file.open("r"))
        name2table[table_file.stem] = [{x: y for x, y in table_obj.items()}]

    split2examples = collections.defaultdict(list)
    for csv_file in csv_folder_path.iterdir():
        df = pd.read_csv(csv_file, sep="\t")

        for idx, row in df.iterrows():
            example = {
                "sentence": row["hypothesis"],
                "table": json.dumps(name2table[row["table_id"]]),
                "label": 0
                if row["label"] == "C"
                else 1
                if row["label"] == "N"
                else 2,
                "title": name2table[row["table_id"]][0]["title"][0],
            }
            split2examples[csv_file.stem.replace("infotabs_", "")].append(example)

    for split, examples in split2examples.items():
        with jsonlines.open(args.output_folder_path + f"{split}.jsonl", "w") as writer:
            writer.write_all(examples)
