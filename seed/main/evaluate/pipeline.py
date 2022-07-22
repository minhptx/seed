from re import I
import sys
from seed.pipeline.pipeline import Pipeline
from seed.pipeline.retriever import Retriever
from seed.pipeline.verifier import Verifier
from seed.pipeline.extractor import Extractor
import pandas as pd
import jsonlines
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import os


sys.path.append("/nas/ckgfs/users/minhpham/workspace/seed")
sys.path.append("/nas/ckgfs/users/minhpham/workspace/BLINK")


@dataclass
class PipelineArguments:
    retriever: str = field(default="wiki")
    verifier: str = field(default="bart")
    extractor: str = field(default="unifiedqa")
    output_file: str = field(default="output.csv")


if __name__ == "__main__":
    parser = HfArgumentParser((PipelineArguments,))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    extractor = Extractor.get(args.extractor)
    verifier = Verifier.get(args.verifier)
    retriever = Retriever.get(args.retriever)
    pipeline = Pipeline(retriever, verifier, extractor)


    df = pd.DataFrame(list(jsonlines.open("data/totto/test.jsonl")))
    df = df.iloc[:50]
    df["pred"] = None
    df["is_correct_cell"] = False
    df["is_correct_case"] = False

    for idx, obj in df.iterrows():
        table = pd.DataFrame([obj["table"]])
        print(table)

        print("Label: ", obj["label"])
        print("Counter factual", obj["counter_fact"])
        if table.empty or len(df.columns) == 0:
            continue
        label = obj["label"]
        column = pipeline.run(table)



        if not label:
            _, _, _, wrong_value = obj["counter_fact"]
        else:
            wrong_value = None
        if column is None:
            value = None
        elif column == "Test":
            value = "Test"
        else:
            value = table[column][0]
        
        if value is not None and wrong_value is not None:
            df.loc[idx, "is_correct_case"] = True
        elif value is None and wrong_value is None:
            df.loc[idx, "is_correct_case"] = True

        df.loc[idx, "is_correct_cell"] = value == wrong_value

        if column is not None:
            df.loc[idx, "preds"] = value
        else:
            df.loc[idx, "preds"] = None

    df.to_csv(args.output_file)

