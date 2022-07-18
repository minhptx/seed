from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from dataclasses import field, dataclass
import jsonlines
from transformers import HfArgumentParser
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

@dataclass
class Argument:
    output_path: str = field(
        default="data/totto/totto_triplet.jsonl",
        metadata={"help": "Path to the output file"},
    )
    output_dir: str = field(
        default="models/bart-triplet-2",
        metadata={"help": "Path to the output directory"},
    )


if __name__ == "__main__":
    args = HfArgumentParser((Argument, )).parse_args_into_dataclasses()[0]

    examples = []

    for item in jsonlines.open(args.output_path):
        table = pd.DataFrame([item["table"]])
        for column in table.columns:
            table[column] = table[column].apply(lambda x: f"{column} : {' , '.join(x) if isinstance(x, list) else x}")

        linearized_table = " ; ".join(
            table.apply(lambda x: " ; ".join(x), axis=1).values.tolist()
        )
        examples.append(InputExample(texts=[linearized_table, item["true_sentence"], item["false_sentence"]]))


    word_embedding_model = models.Transformer('facebook/bart-base', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
    train_loss = losses.TripletLoss(model=model, triplet_margin=0.5)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=4, show_progress_bar=True, output_path=args.output_dir)
