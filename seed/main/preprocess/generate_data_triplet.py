from copy import deepcopy
from dataclasses import dataclass, field
import json
import random
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import spacy
from blingfire import *
from elasticsearch_dsl import connections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from transformers import HfArgumentParser
from collections import defaultdict

connections.create_connection(hosts=["http://ckg05:9200/"], timeout=100)
nlp = spacy.load("en_core_web_sm")


@dataclass
class Argument:
    input_path: str = field(
        default="data/totto/train_processed.jsonl",
        metadata={"help": "Path to the input file"},
    )
    output_path: str = field(
        default="data/totto/train_triplet.jsonl",
        metadata={"help": "Path to the output file"},
    )
    cache_file: str = field(
        default=".cache/title2text.json",
        metadata={"help": "Path to cache file"},
    )
    filter: bool = field(
        default=False, metadata={"help": "Filter out samples with no highlighted cells"}
    )
    test_set: bool = field(default=False, metadata={"help": "Is this a test set?"})


def replace_table(obj, replacements, main_column):
    obj = deepcopy(obj)
    if not replacements[main_column] or len(replacements) <= 1:
        return None

    another_column = random.choice([x for x in replacements if x != main_column and obj[x]])

    main_replacement = random.choice(replacements[main_column])
    other_replacement = random.choice(replacements[another_column])

    obj[main_column][0] = main_replacement
    obj[another_column][0] = other_replacement

    return obj


def replace_sentence(sentence, text):
    sentences = [x for x in text_to_sentences(text).split("\n") if x]

    vectorizer = TfidfVectorizer()
    if not sentences:
        return None
    X_corpus = vectorizer.fit_transform(sentences)

    X_query = vectorizer.transform([sentence])

    scores = cosine_similarity(X_query, X_corpus)
    indices = np.argsort(scores[0])[::-1]

    key_indices = []

    for idx in indices[1:]:
        key_indices.append(idx)

    if not key_indices:
        return None
    idx = random.choice(indices[1:])

    return sentences[idx]


def linearize_table(obj):
    table = pd.DataFrame([obj])
    for column in table.columns:
        table[column] = table[column].apply(
            lambda x: f"{column} : {' , '.join(x) if isinstance(x, list) else x}"
        )

    linearized_table = " ; ".join(
        table.apply(lambda x: " ; ".join(x), axis=1).values.tolist()
    )
    return linearized_table


def generate_negative_examples(obj, replacements, main_column, sentence, title):
    false_sentence = None
    if title in title2text:
        text = title2text[title]
        false_sentence = replace_sentence(sentence, text)
    false_table = replace_table(obj, replacements, main_column)
    examples = []
    if false_sentence is not None:
        examples.append(
            {
                "anchor": linearize_table(obj),
                "positive": sentence,
                "negative": false_sentence,
            }
        )
    if false_table is not None:
        examples.append(
            {
                "anchor": sentence,
                "positive": linearize_table(obj),
                "negative": linearize_table(false_table),
            }
        )

    return examples


def filter_table(table, sample):
    if (
        len(table.columns) == 0
        or len(table) == 0
        or len(sample["highlighted_cells"]) == 0
    ):
        return False
    rows = set([x[0] for x in sample["highlighted_cells"]])
    if len(rows) >= 3:
        return False
    main_column = table.columns[0]
    try:
        if len(table.loc[0, main_column]) == 0 or table.loc[0, main_column][0].isdigit():
            return False
    except:
        print("Error", table, main_column)
    count = 0
    for column in table.columns:
        if (
            len([x for x in table[column][0] if x.isdigit()])
            >= len(table[column][0]) * 0.5
        ):
            count += 1
    if count >= len(table.columns):
        return False
    return True


def preprocess(sample):
    table = pd.DataFrame(json.loads(sample["table"]), index=None).fillna("")
    if not filter_table(table, sample):
        return []

    main_column = table.columns[0]

    obj = defaultdict(list)
    replacements = defaultdict(list)
    for row, col in sample["highlighted_cells"]:
        if len(table.iloc[row, col]) != 0:
            obj[table.columns[col]].append(table.iloc[row, col])

    for idx, row in table.iterrows():
        for col in table.columns:
            if col in obj and row[col] not in obj[col] and len(row[col]) != 0:
                replacements[col].append(row[col])
    sentence = sample["sentence_annotations"][0]["original_sentence"]

    return generate_negative_examples(
        obj, replacements, main_column, sentence, sample["table_page_title"]
    )


if __name__ == "__main__":
    parser = HfArgumentParser(Argument)
    args = parser.parse_args()

    title2text = {}

    if Path(args.cache_file).exists():
        with open(args.cache_file, "r") as f:
            title2text = json.load(f)

    with jsonlines.open(args.input_path, "r") as reader:
        pool = Pool(processes=50)
        results = pool.imap_unordered(
            preprocess, reader.iter(type=dict, skip_invalid=True)
        )
        pool.close()

        with jsonlines.open(args.output_path, mode="w") as writer:
            for result in results:
                if result:
                    for res in result:
                        if res:
                            writer.write(res)
