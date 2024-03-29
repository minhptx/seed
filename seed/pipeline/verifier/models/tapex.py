from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from nltk import wordpunct_tokenize

class TapexVerifier:
    def __init__(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained("../saved_models/tapex/")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-base")

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