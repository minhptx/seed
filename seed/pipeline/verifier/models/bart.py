import torch
from transformers import (
    BartTokenizer,
    BartForSequenceClassification,
)
import os
from nltk import wordpunct_tokenize

from seed.pipeline.verifier.sent_selection import SentenceSelector


class BartVerifier:
    def __init__(self) -> None:
        self.model = BartForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        )
        # self.model.classifier = nn.Linear(model_config.hidden_size, 3)
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.sentence_selector = SentenceSelector()
        
    def text_similarity(self, doc, table):
        list1 = wordpunct_tokenize(doc)
        list2 = wordpunct_tokenize(" ".join(table.astype(str).values.tolist()[0]))
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def verify(self, sentence, df):
        if self.text_similarity(sentence, df) < 0.01:
            return None

        table = df.copy()

        for column in table.columns:
            table[column] = table[column].apply(lambda x: f"{column} : {' , '.join(x) if isinstance(x, list) else x}")

        column2res = {}
        title = table.columns[0]
        for column in table.columns:
            if column == title:
                continue
            linearized_table = f" {self.tokenizer.sep_token} ".join(
                table[[title, column]].apply(lambda x: " ; ".join(x), axis=1).values.tolist()
            )

            if not self.sentence_selector.check_select(linearized_table, sentence):
                continue

            encoding = self.tokenizer(
                linearized_table,
                sentence,
                truncation=True,
                padding=True,
                return_tensors="pt")

            outputs = self.model(**encoding)
            predicted_class_idx = outputs.logits[0].argmax(dim=0).item()
            probs = torch.softmax(outputs.logits[0], dim=0)
            if predicted_class_idx != 1:
                column2res[column] = (predicted_class_idx, torch.max(probs).item())
        return column2res

    