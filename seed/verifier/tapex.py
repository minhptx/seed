from transformers import AutoModel, AutoTokenizer
import torch
from nltk import wordpunct_tokenize

class TapexVerifier:
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained("../../models/tapex/checkpoint-10000")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-base")

    def text_similarity(self, doc, table):
        list1 = wordpunct_tokenize(doc)
        list2 = wordpunct_tokenize(" ".join(table.astype(str).values.tolist()[0]))
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def verify(self, sentence, df):
        if self.text_similarity(sentence, df) < 0.01:
            return 0, 0
        encoding = self.tokenizer(
            table=df,
            query=sentence,
            truncation=True,
            padding=True,
            return_tensors="pt")

        outputs = self.model(**encoding)

        predicted_class_idx = outputs.logits[0].argmax(dim=0).item()
        probs = torch.softmax(outputs.logits[0], dim=0)

        return predicted_class_idx, probs