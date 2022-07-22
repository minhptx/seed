#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The Microsoft and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning the library models for tapex on table-based fact verification tasks.
Adapted from script: https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
"""

import logging
from seed.datasets.table_nli import TableNLIUltis
from dataclasses import dataclass, field
from typing import Optional, List
import json
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    AutoModelForSequenceClassification
)
import os
import sys
import torch
import pandas as pd
import wandb
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset: Optional[str] = field(
        default="clapika2010/totto",
        metadata={
            "help": "Dataset to use for training and eval."
        }
    )

    test_splits: Optional[List[str]] = field(
        default_factory=lambda: ["test"],
        metadata={
            "help": "Test splits to use for evaluation."
        }
    )

    train_file: Optional[str] = field(
        default="data/train.json",
        metadata={
            "help": "The path to the train dataset to use (via the datasets library)."
        },
    )
    dev_file: Optional[str] = field(
        default="data/dev.json",
        metadata={
            "help": "The path to the dev dataset to use (via the datasets library)."
        },
    )
    test_file: Optional[str] = field(
        default="data/test.json",
        metadata={
            "help": "The path to the test dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/bart-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def process_table(item, tokenizer):
    table = pd.DataFrame(json.loads(item["table"])).drop("index", axis=1)
    for column in table.columns:
        table[column] = table[column].apply(lambda x: f"{column} : {' , '.join(x) if isinstance(x, list) else x}")

    linearized_table = tokenizer.sep_token.join(
        table.apply(lambda x: " ; ".join(x), axis=1).values.tolist()
    )
    encoding = tokenizer(
        linearized_table,
        item["sentence"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    encoding = {x: y.squeeze() for x, y in encoding.items()}
    encoding["labels"] = torch.tensor([item["label"]]).long()
    encoding["label"] = encoding["labels"]
    return encoding


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    with training_args.main_process_first():
        wandb.init(
            project="seed",
            entity="clapika",
            name=datetime.now().strftime("bart " + "_%Y%m%d-%H%M%S"),
            group="bart",
            config={"dataset": data_args.dataset}
        )

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=3,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        problem_type="single_label_classification",
    )
    # load tapex tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        add_prefix_space=True,
    )

    if not isinstance(data_args.test_file, list):
        data_args.test_file = [data_args.test_file]

    with training_args.main_process_first():
        datasets = TableNLIUltis.from_jsonlines(
            data_args.dataset
        )

        # datasets.push_to_hub(data_args.dataset + "_processed") 
        train_dataset, val_dataset, predict_datasets = datasets["train"], datasets["dev"], [datasets[split] for split in data_args.test_splits]
    

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Set seed before initializing model.

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1).squeeze()
        return {"accuracy": (preds == p.label_ids.squeeze()).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Training

    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train(ignore_keys_for_eval=["encoder_last_hidden_state"])
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=val_dataset, ignore_keys=["encoder_last_hidden_state"])
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(val_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(val_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        output_predict_file = os.path.join(training_args.output_dir, f"infotab.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:

                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                for idx, predict_dataset in enumerate(predict_datasets):
                    outputs = trainer.predict(predict_dataset, ignore_keys=["encoder_last_hidden_state"])
                    all_predictions = outputs.predictions
                    metrics = outputs.metrics
                    trainer.log_metrics(f"test_{idx}", metrics)
                    trainer.save_metrics(f"test_{idx}", metrics)
                    wandb.log({f"test_{idx}/accuracy": metrics["test_accuracy"]})
                    predictions = np.argmax(all_predictions, axis=-1)

                    logger.info(f"***** Predict Results *****")
                    for index, item in enumerate(predictions):
                        writer.write(f"Index: {index}\t Preds: {item}\n")
                        writer.write(str(all_predictions[index]) + "\n")
                        writer.write(str(predict_dataset[index]["label"]) + "\n")
                        writer.write(predict_dataset[index]["sentence"] + "\n")
                        writer.write(predict_dataset[index]["table"] + "\n")
                        result = predictions[index] == predict_dataset[index]["label"]
                        writer.write("Result: " + str(result) + "\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
