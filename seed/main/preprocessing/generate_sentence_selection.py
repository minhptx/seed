import json
from dataclasses import dataclass, field
from pathlib import Path
from turtle import forward

from transformers import HfArgumentParser
from blingfire import text_to_sentences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import torch
import jsonlines
from multiprocessing import Pool
import pandas as pd


@dataclass
class Argument:
    input_path: str = field(
        default="data/totto/totto_train_data_processed.jsonl",
        metadata={"help": "Path to the input file"},
    )
    output_path: str = field(
        default="data/totto/totto_triplet.jsonl",
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


def replace_sentence(df, sentence, text):
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

    return {"table": df, "true_sentence": sentence, "false_sentence": sentences[idx]}


def preprocess(sample):
    table = pd.DataFrame(json.loads(sample["table"]), index=None).fillna("")
    if len(table) == 0:
        return None
    
    obj = {}
    for row, col in sample["highlighted_cells"]:
        obj[table.columns[col]] = table.iloc[row, col]

    sentence = sample["sentence_annotations"][0]["original_sentence"]
    try:
        new_sample = replace_sentence(obj, sentence, title2text[sample["table_page_title"]])
    except:
        return None
    return new_sample

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

        with jsonlines.open(args.output_path, mode="w") as writer:
            for result in results:
                if result is not None:
                    writer.write(result)