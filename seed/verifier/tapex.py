from transformers import AutoModel, AutoTokenizer
import torch

class TapexVerifier:
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained("models/tapex/checkpoint-10000")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-base")

    def verify(self, sentence, df):
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