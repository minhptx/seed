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
    return sentences[idx]


def replace_sentence_words(df, sentence, text):
    replaced_ents = nlp(sentence).ents
    replacing_ents = nlp(text).ents
    if replaced_ents:
        replaced_ent = random.choice(replaced_ents)
        same_type_ents = list(
            filter(
                lambda x: x.label_ == replaced_ent.label_
                and x.text != replaced_ent.text,
                replacing_ents,
            )
        )
        if same_type_ents:
            replacing_ent = random.choice(same_type_ents)
            sent1 = sentence.replace(replaced_ent.text, replacing_ent.text)
            return sent1
    return None


def replace_random_words(df, sentence, text):
    sentence_doc = nlp(sentence)
    valid_words = list(
        filter(lambda x: x.pos_ in ["VERB", "NOUN", "PROPN", "NUM"], sentence_doc)
    )
    try:
        chosen_word = random.choice(valid_words)
    except:
        return None

    valid_replacements = list(
        filter(
            lambda x: x.pos_ == chosen_word.pos_ and x.text != chosen_word.text,
            sentence_doc,
        )
    )

    if valid_replacements:
        replacing_word = random.choice(valid_replacements)
        return sentence.replace(chosen_word.text, replacing_word.text)

    return None


def replace_table_named_entities(df, highlighted_cells):
    x, y = random.choice(highlighted_cells)

    column_indices = set([cell[1] for cell in highlighted_cells])

    ne_columns = []
    try:
        for idx in column_indices:
            entity_count = min(df.iloc[:, idx].apply(lambda x: len(nlp(x).ents)))

            if entity_count > 0:
                ne_columns.append(idx)

        ne_columns = [x for x in highlighted_cells if x[1] in ne_columns]
        if not ne_columns:
            return None
        x, y = random.choice(ne_columns)

        other_values = df.iloc[:, y].values.tolist()
        replacing_value = random.choice(other_values)
        choices = [v for v in other_values if v != df.iloc[x, y]]
        if not choices:
            return None
        replacing_value = random.choice(choices)
        return (x, y, replacing_value)
    except Exception as e:
        print(e)
        return None


def replace_table_numbers(df, highlighted_cells):
    x, y = random.choice(highlighted_cells)

    column_indices = set([cell[1] for cell in highlighted_cells])

    numeric_columns = []

    for idx in column_indices:
        if idx >= len(df.columns):
            continue
        if df[df.columns[idx]].dtype in [np.float64, np.int64]:
            numeric_columns.append(idx)

    numeric_cells = [x for x in highlighted_cells if x[1] in numeric_columns]
    if not numeric_cells:
        return None
    x, y = random.choice(numeric_cells)

    try:
        other_values = df.iloc[:, y].values.tolist()
        replacing_value = random.choice(other_values)
        choices = [v for v in other_values if v != df.iloc[x, y]]
        if not choices:
            return None
        replacing_value = random.choice(choices)
        return (x, y, replacing_value)
    except Exception as e:
        print(e)
        return None


def generate_negative_examples(df, sample, sentence):
    highlighted_cells = sample["highlighted_cells"]

    t2func = {
        0: replace_sentence,
        1: replace_sentence_words,
        2: replace_random_words,
        3: replace_table_named_entities,
        4: replace_table_numbers,
    }


    if sample["table_page_title"] in title2text:
        doc = title2text[sample["table_page_title"]]
        valid_choices = [0, 1, 2, 3, 4]
    else:
        doc = ""
        valid_choices = [3, 4]

    if not sample["highlighted_cells"]:
        valid_choices = [x for x in valid_choices if x not in[3, 4]]

    while valid_choices:
        i = random.choice(valid_choices)
        if sample["highlighted_cells"]:
            rows = [x[0] for x in sample["highlighted_cells"]]
            most_dominant_row = max(set(rows), key=rows.count)
        else:
            most_dominant_row = 0

        if i not in [3, 4]:
            new_sentence = t2func[i](df, sentence, doc)
                
            if new_sentence is not None:
                new_sample = sample.copy()
                new_sample["sentence"] = new_sentence
                new_sample["label"] = 0
                try:
                    new_sample["table"] = (
                        df.iloc[[most_dominant_row], :]
                        .reset_index()
                        .drop("index", axis=1)
                        .to_json(orient="records")
                    )
                except:
                    print("Error", most_dominant_row, df)

                new_sample["note"] = f"Replaced '{sentence}' with {new_sentence} type {i}"
                return new_sample, most_dominant_row
        else:
            # replace sentence with sentence that have the same entities
            value = t2func[i](df, highlighted_cells)
            new_sample = sample.copy()

            print("Value", i, value)

            if value is not None:
                x, y, value = value
                new_df = df.copy()
                try:
                    new_sample[
                        "note"
                    ] = f"{df.iloc[x, y]} at {x},{y} is replaced with {value}"
                except Exception as e:
                    print(e)
                    return None
                new_df.iloc[x, y] = value
                new_sample["table"] = (
                    new_df.iloc[[most_dominant_row], :]
                    .reset_index()
                    .drop("index", axis=1)
                    .to_json(orient="records")
                )
                return new_sample, most_dominant_row
        valid_choices.remove(i)

    return None, -1


def preprocess(sample):

    table = pd.DataFrame(json.loads(sample["table"]), index=None).fillna("")
    if len(table) == 0:
        return sample
    sentences = [
        sample["sentence_annotations"][0]["original_sentence"],
        sample["sentence_annotations"][0]["sentence_after_deletion"],
        sample["sentence_annotations"][0]["sentence_after_ambiguity"],
        sample["sentence_annotations"][0]["final_sentence"],
    ]

    sentence = np.random.choice(sentences, 1, p=[0.4, 0.2, 0.2, 0.2]).tolist()[0]

    new_sample, _ = generate_negative_examples(table, sample, sentence)

    return new_sample


@dataclass
class Argument:
    input_path: str = field(
        default="data/totto_data/processed_totto_train_data.jsonl",
        metadata={"help": "Path to the input file"},
    )
    output_path: str = field(
        default="data/totto_data/processed_totto_train_data_augmented.jsonl",
        metadata={"help": "Path to the output file"},
    )
    cache_file: str = field(
        default=".cache/title2text.json",
        metadata={"help": "Path to cache file"},
    )
    filter: bool = field(
        default=False, metadata={"help": "Filter out samples with no highlighted cells"}
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

        with jsonlines.open(args.output_path, mode="w") as writer:
            for result in results:
                writer.write(result)
