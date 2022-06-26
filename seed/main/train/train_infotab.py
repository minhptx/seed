import json
import os
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import sys
import inflect
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
from seed.datasets.table_nli import TableNLIDataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, HfArgumentParser

inflect = inflect.engine()


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, labels):
        """Constructor
        Input: in_dim	- Dimension of input vector
                   out_dim	- Dimension of output vector
                   vocab	- Vocabulary of the embedding
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.drop = torch.nn.Dropout(0.2)
        self.fc2 = nn.Linear(out_dim, labels)
        # self.soft_max = torch.nn.Softmax(dim=1)

    def forward(self, inp):
        """Function for forward pass
        Input:	inp 	- Input to the network of dimension in_dim
        Output: output 	- Output of the network with dimension vocab
        """
        out_intermediate = F.relu(self.fc1(inp))
        output = self.fc2(out_intermediate)
        return output

def is_date(string):
    match = re.search("\d{4}-\d{2}-\d{2}", string)
    if match:
        return True
    else:
        return False


def load_sentences(file, skip_first=True, single_sentence=False):
    """Loads sentences into process-friendly format for a given file path.
    Inputs
    -------------------
    file    - str or pathlib.Path. The file path which needs to be processed
    skip_first      - bool. If True, skips the first line.
    single_sentence - bool. If True, Only the hypothesis statement is chosen.
                            Else, both the premise and hypothesis statements are
                            considered. This is useful for hypothesis bias experiments.

    Outputs
    --------------------
    rows    - List[dict]. Consists of all data samples. Each data sample is a
                    dictionary containing- uid, hypothesis, premise (except hypothesis
                    bias experiment), and the NLI label for the pair
    """
    rows = []
    print(file)
    df = pd.read_csv(file, sep="\t")
    for idx, row in df.iterrows():
        # Takes the relevant elements of the row necessary. Putting them in a dict,
        print(row)
        if single_sentence:
            sample = {
                "uid": row["annotator_id"],
                "hypothesis": row["hypothesis"],
                "label": int(row["label"]),
            }
        else:
            sample = {
                "uid": row["index"],
                "hypothesis": row["hypothesis"],
                "premise": row["premise"],
                "label": int(row["label"]),
            }

        rows.append(sample)  # Append the loaded sample
    return rows


def json_to_para(data, args):
    result = []

    if args.rand_perm == 2:
        table_ids = []
        for index, row in enumerate(data):
            row = row.to_dict()
            table_ids += [row["table_id"]]
        random.shuffle(table_ids)
        for index, row in enumerate(data):
            row = row.to_dict()
            row["table_id"] = table_ids[index]

    if args.rand_perm == 1:
        table_ids = []
        for index, row in enumerate(data):
            row = row.to_dict()
            table_ids += [row["table_id"]]

        set_of_orignal = list(set(table_ids))
        set_of_random = set_of_orignal
        random.shuffle(set_of_random)
        set_of_orignal = list(set(table_ids))
        random_mapping_tableids = {}
        jindex = 0

        for key in set_of_orignal:
            random_mapping_tableids[key] = set_of_random[jindex]
            jindex += 1

        for index, row in data.iterrows():
            table_id = row["table_id"]
            row["table_id"] = random_mapping_tableids[table_id]

    for index, row in enumerate(data):
        row = row.to_dict()
        obj = row["table"].to_dict(orient="records")

        if not obj:
            continue
        obj = obj[0]
        obj["title"] = row['title']
        try:
            title = obj["title"][0]
        except KeyError as e:
            print(e)
            exit()

        del obj["title"]

        para = ""

        print(obj)
        for key in obj:
            line = ""
            values = [obj[key]]
            if isinstance(key, tuple):
                key = " ".join(tuple)

            try:
                res = inflect.plural_noun(key)
            except:
                res = False

            print(values)

            if (len(values) > 1) and res:
                verb_use = "are"
                if is_date("".join(values)):
                    para += title + " was " + str(key) + " on "
                    line += title + " was " + str(key) + " on "
                else:
                    try:
                        para += (
                            "The " + str(key) + " of " + title + " " + verb_use + " "
                        )
                        line += (
                            "The " + str(key) + " of " + title + " " + verb_use + " "
                        )
                    except TypeError as e:
                        print(e)
                        print(row)
                        print(key)
                        print(title)
                        continue
                for value in values[:-1]:
                    para += value + ", "
                    line += value + ", "
                if len(values) > 1:
                    para += "and " + values[-1] + ". "
                    line += "and " + values[-1] + ". "
                else:
                    para += values[-1] + ". "
                    line += values[-1] + ". "
            else:
                verb_use = "is"
                if is_date(values[0]):
                    para += title + " was " + str(key) + " on " + values[0] + ". "
                    line += title + " was " + str(key) + " on " + values[0] + ". "
                else:
                    para += (
                        "The "
                        + str(key)
                        + " of "
                        + title
                        + " "
                        + verb_use
                        + " "
                        + values[0]
                        + ". "
                    )
                    line += (
                        "The "
                        + str(key)
                        + " of "
                        + title
                        + " "
                        + verb_use
                        + " "
                        + values[0]
                        + ". "
                    )

        label = row["label"]
        if row["label"] == "E":
            label = 0
        if row["label"] == "N":
            label = 1
        if row["label"] == "C":
            label = 2

        obj = {
            "index": index,
            "table_id": row["table_id"],
            "annotator_id": row["annotator_id"],
            "premise": para,
            "hypothesis": row["hypothesis"],
            "label": label,
        }
        result.append(obj)
    return result


def preprocess_roberta(data, args):
    new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type)
    # Process for every split
    # Initialize dictionary to store processed information
    keys = ["uid", "encodings", "attention_mask", "segments", "labels"]
    data_dict = {key: [] for key in keys}
    result = []
    samples_processed = 0
    # Iterate over all data points
    for pt_dict in data:

        samples_processed += 1
        # Encode data. The premise and hypothesis are encoded as two different segments. The
        # maximum length is chosen as 504, i.e, 500 sub-word tokens and 4 special characters
        # If there are more than 504 sub-word tokens, sub-word tokens will be dropped from
        # the end of the longest sequence in the two (most likely the premise)
        if args.single_sentence:
            pt_dict["hypothesis"] = str(pt_dict["hypothesis"])
            encoded_inps = new_tokenizer(
                pt_dict["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=504,
            )
        else:
            pt_dict["hypothesis"] = str(pt_dict["hypothesis"])
            pt_dict["premise"] = str(pt_dict["premise"])
            encoded_inps = new_tokenizer(
                pt_dict["premise"],
                pt_dict["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=504,
            )

        # Some models do not return token_type_ids and hence
        # we just return a list of zeros for them. This is just
        # required for completeness.
        if "token_type_ids" not in encoded_inps.keys():
            encoded_inps["token_type_ids"] = [0] * len(encoded_inps["input_ids"])

        data_dict["uid"].append(int(pt_dict["index"]))
        data_dict["encodings"].append(encoded_inps["input_ids"])
        data_dict["attention_mask"].append(encoded_inps["attention_mask"])
        data_dict["segments"].append(encoded_inps["token_type_ids"])
        data_dict["labels"].append(pt_dict["label"])

        if (samples_processed % 100) == 0:
            print("{} examples processed".format(samples_processed))

    print("Preprocessing Finished")
    return data_dict





def test(model, classifier, data):
    """Evaluate the model on a given dataset.
    Inputs
    ---------------
    model - transformers.AutoModel. The transformer model being used.
    classifier - torch.nn.Module. The classifier which sits on top of
                    the transformer model
    data - dict. Consists the processed input data

    Outputs
    ---------------
    accuracy - float. Accuracy of the model on that evaluation split
    gold_inds - List[int]. Gold labels
    predictions_ind - List[int]. Parallel list to gold_inds. Contains
                        label predictions
    """
    # Separate the data fields in the evaluation data
    enc = torch.tensor(data["encodings"]).cuda()
    attention_mask = torch.tensor(data["attention_mask"]).cuda()
    segs = torch.tensor(data["segments"]).cuda()
    labs = torch.tensor(data["labels"]).cuda()
    ids = torch.tensor(data["uid"]).cuda()

    # Create Data Loader for the split
    dataset = TensorDataset(enc, attention_mask, segs, labs, ids)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model.eval()
    correct = 0
    total = 0
    gold_inds = []
    predictions_inds = []

    for batch_ndx, (enc, mask, seg, gold, ids) in enumerate(loader):
        # Forward-pass w/o calculating gradients
        with torch.no_grad():
            outputs = model(enc, attention_mask=mask, token_type_ids=seg)
            predictions = classifier(outputs[1])

        # Calculate metrics
        _, inds = torch.max(predictions, 1)
        gold_inds.extend(gold.tolist())
        predictions_inds.extend(inds.tolist())
        correct += inds.eq(gold.view_as(inds)).cpu().sum().item()
        total += len(enc)

    accuracy = correct / total

    return accuracy, gold_inds, predictions_inds


def train(train_data, dev_data, test_data, args):
    """Train the transformer model on given data
    Inputs
    -------------
    args - dict. Arguments passed via CLI
    """

    # Creating required save directories
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    print("{} train data loaded".format(len(train_data["encodings"])))
    print("{} dev data loaded".format(len(dev_data["encodings"])))
    print("{} test data loaded".format(len(test_data["encodings"])))

    # Separating the data fields
    train_enc = torch.tensor(train_data["encodings"]).cuda()
    train_attention_mask = torch.tensor(train_data["attention_mask"]).cuda()
    train_segs = torch.tensor(train_data["segments"]).cuda()
    train_labs = torch.tensor(train_data["labels"]).cuda()
    train_ids = torch.tensor(train_data["uid"]).cuda()

    # Intialize Models
    model = AutoModel.from_pretrained(args.model_type).cuda()
    args.embed_size = model.config.hidden_size
    classifier = FeedForward(
        args.embed_size, int(args.embed_size / 2), 2
    ).cuda()

    # Creating the training dataloaders
    dataset = TensorDataset(
        train_enc, train_attention_mask, train_segs, train_labs, train_ids
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # Intialize the optimizer and loss functions
    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.Adagrad(params, lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        epoch_loss = 0
        start = time.time()
        model.train()
        # Iterate over batches

        for batch_ndx, (enc, mask, seg, gold, ids) in enumerate(tqdm(loader)):
            batch_loss = 0

            optimizer.zero_grad()
            # Forward-pass
            outputs = model(enc, attention_mask=mask, token_type_ids=seg)
            predictions = classifier(outputs[1])
            # Loss calculation and weight update
            out_loss = loss_fn(predictions, gold)
            out_loss.backward()
            optimizer.step()

        batch_loss += out_loss.item()
        epoch_loss += batch_loss

        normalized_epoch_loss = epoch_loss / (len(loader))
        print("Epoch {}".format(ep + 1))
        print("Epoch loss: {} ".format(normalized_epoch_loss))

        # Evaluate on the dev and test sets
        dev_acc, dev_gold, dev_pred = test(model, classifier, dev_data)
        test_acc, test_gold, test_pred = test(model, classifier, test_data)
        end = time.time()
        print("Dev Accuracy: {}".format(dev_acc))
        print("Time taken: {} seconds\n".format(end - start))
        # Save model
        torch.save(
            {
                "epoch": ep + 1,
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "loss": normalized_epoch_loss,
                "dev_accuracy": dev_acc,
            },
            os.path.join(args.output_dir, "model_"
            + str(ep + 1)
            + "_"
            + str(dev_acc))
        )

        wandb.log({"epoch": ep + 1,"loss": normalized_epoch_loss, "dev_accuracy": dev_acc, "dev_gold": dev_gold, "dev_pred": dev_pred, 
        "test_accuracy": test_acc, "test_gold": test_gold, "test_pred": test_pred})


def test_data(data, args):
    """Test pre-trained model on evaluation splits
    Inputs
    ----------
    args - dict. Arguments passed via CLI
    """
    # Intialize model
    model = AutoModel.from_pretrained(args.model_type).cuda()
    embed_size = model.config.hidden_size
    classifier = FeedForward(embed_size, int(embed_size / 2), 2).cuda()

    # Load pre-trained models
    checkpoint = torch.load(os.path.join(args.output_dir, args.model_name))
    model.load_state_dict(checkpoint["model_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    # Evaluate over splits

    # Compute Accuracy
    acc, gold, pred = test(model, classifier, data)

    results = {"accuracy": acc, "gold": gold, "pred": pred}

    return results

@dataclass
class DataArguments:
    train_file: str = field(
        default="data/train.json",
        metadata={"help": "The path to train data file"},
    )
    dev_file: str = field(
        default="data/dev.json",
        metadata={"help": "The path to dev data file"},
    )
    test_file: str = field(
        default="data/test.json",
        metadata={"help": "The path to test data file"},
    )
    rand_perm: int = field(
        default=0,
        metadata={"help": "If 1, random permutation of table ids is performed"},
    )
    single_sentence: int = field(
        default=0,
        metadata={"help": "If 1, the hypothesis is treated as a single sentence"},
    )
    max_len: int = field(
        default=512,
        metadata={"help": "The maximum length of the premise and hypothesis"},
    )
    tokenizer_type: str = field(
        default="roberta-base",
        metadata={"help": "The type of tokenizer to use"},
    )
    model_name: str = field(
        default="model_4_0.301", metadata={"help": "The directory containing the model"}
    )
    output_dir: str = field(
        default="temp/models/",
        metadata={"help": "The directory to save the data files"},
    )

if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()

    wandb.init(project="seed", entity="clapika", config={"model_name": "infotab"})
    print("Reading datasets")
    train_dataset = TableNLIDataset.from_jsonlines(args.train_file).to_infotab()
    dev_dataset = TableNLIDataset.from_jsonlines(args.dev_file).to_infotab()
    test_dataset = TableNLIDataset.from_jsonlines(args.test_file).to_infotab()
    datasets = [train_dataset, dev_dataset, test_dataset]
    for idx in range(3):
        print("Processing dataset ...")
        datasets[idx] = json_to_para(datasets[idx], args)
        datasets[idx] = preprocess_roberta(datasets[idx], args)

    print("Training ...")
    train_dataset, dev_dataset, test_dataset = datasets
    train(train_dataset, dev_dataset, test_dataset, args)
    