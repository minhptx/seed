from collections import defaultdict
from dataclasses import dataclass, field
import itertools
import json
import random
import sys
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import spacy
from blingfire import *
from elasticsearch_dsl import MultiSearch, Search, connections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from transformers import HfArgumentParser
import wikitextparser as wtp

connections.create_connection(hosts=["http://ckg05:9200/"], timeout=100)
nlp = spacy.load("en_core_web_sm")


def replace_table_values(df, highlighted_cells):
    x, y = random.choice(highlighted_cells)

    x_replaced = random.choice([i for i in range(len(df)) if i != x and (i, y) not in highlighted_cells])

    original_value = df.iloc[x, y]
    df.iloc[x, y] = df.iloc[x_replaced, y]

    return df, (x, y, original_value, df.iloc[x, y])

def preprocess(sample):
    table = pd.DataFrame(json.loads(sample["table"]), index=None).fillna("")
    if len(table) == 0 or len(sample["highlighted_cells"]) == 0:
        return None, None

    try:
        new_sample, counter_fact = replace_table_values(table, sample["highlighted_cells"])
    except:
        return None, None
    example = defaultdict(list)
    new_example = defaultdict(list)
    for i, j in sample["highlighted_cells"]:
        example[table.columns[j]].append(table.iloc[i, j])
        new_example[table.columns[j]].append(new_sample.iloc[i, j])
    return {"table": example, "label": True, "counter_fact": "", "title": sample["table_page_title"]}, {"table": new_example, "label": False, "counter_fact": counter_fact, "title": sample["table_page_title"]}


@dataclass
class Argument:
    input_path: str = field(
        default="data/totto/dev_processed.jsonl",
        metadata={"help": "Path to the input file"},
    )
    output_path: str = field(
        default="data/totto/test.jsonl",
        metadata={"help": "Path to the output file"},
    )
    cache_file: str = field(
        default=".cache/dev_title2text.json",
        metadata={"help": "Path to cache file"},
    )
    filter: bool = field(
        default=False, metadata={"help": "Filter out samples with no highlighted cells"}
    )
    test_set: bool = field(
        default=False, metadata={"help": "Is this a test set?"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(Argument)
    args = parser.parse_args()

    title2text = {}

    if Path(args.cache_file).exists():
        with open(args.cache_file, "r") as f:
            title2text = json.load(f)
    else:
        with jsonlines.open(args.input_path) as reader:
            samples = [x for x in reader]

            doc = []

            print("Processing sentences")

            for i in range(0, len(samples), 10000):
                ms = MultiSearch(index="test_en")

                for sample in samples[i: i + 10000]:
                    ms = ms.add(Search().query("match", title__keyword=sample["table_page_title"]))

                responses = ms.execute()

                for response in responses:
                    if not response.hits:
                        continue
                    if not response.hits[0]["title"] in title2text:
                        try:
                            title2text[response.hits[0]["title"]] = wtp.remove_markup(response.hits[0]["text"])
                        except:
                            title2text[response.hits[0]["title"]] = response.hits[0]["text"]

        json.dump(title2text, open(args.cache_file, "w"))

    pool = Pool(processes=50)

    with jsonlines.open(args.input_path, "r") as reader:
        pool = Pool(processes=50)
        results = pool.imap_unordered(
            preprocess, reader.iter(type=dict, skip_invalid=True)
        )
        pool.close()

        with jsonlines.open(args.output_path, mode="w") as writer:
            for result in results:
                if result[0] is not None:
                    writer.write(result[0])
                if result[1] is not None:
                    writer.write(result[1])
