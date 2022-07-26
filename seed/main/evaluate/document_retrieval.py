from dataclasses import dataclass, field
from transformers import HfArgumentParser
from seed.pipeline.retriever import Retriever
import sys
import os
import jsonlines
import pandas as pd
import editdistance

@dataclass
class EvaluationArgument:
    model: str = field(
        default="wiki",
        metadata={"help": "Retriever model to user"},
    )
    input_path: str = field(
        default="data/tabpert/processed/test.jsonl",
        metadata={"help": "Dataset to use"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((EvaluationArgument,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        exp_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )[0]
    else:
        exp_args = parser.parse_args_into_dataclasses()[0]

    retriever = Retriever.get(exp_args.model)

    mrr = 0
    recall_at_10 = 0

    df = pd.DataFrame(list(jsonlines.open(exp_args.input_path)))
    for idx, obj in df.iterrows():
        table = pd.DataFrame([obj["table"]])
        table = table.applymap(lambda x: " ; ".join(x) if isinstance(x, list) else x)
        label = obj["label"]
        documents = retriever.search(table)
        
        for idx, document in enumerate(documents[:10]):
            if editdistance.eval(document["title"], obj["title"]) <= 4:
                print(document["title"], "----", obj["title"])

                recall_at_10 += 1
                mrr += 1 / (idx + 1)
                break

    print(f"MRR: {mrr / len(df)}")
    print(f"Recall@10: {recall_at_10 / len(df)}")
    
