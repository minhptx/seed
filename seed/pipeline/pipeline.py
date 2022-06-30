from seed.retriever import Retriever
from seed.verifier import Verifier
from seed.extractor import Extractor
from blingfire import text_to_sentences


class Pipeline:
    def __init__(self, config) -> None:
        self.retriever = Retriever.get(config.retriever_model)
        self.verifier = Verifier.get(config.verifier if config.verifier else "tapex")
        self.extractor = Extractor.get(config.extractor if config.extractor else "unifiedqa")

    def run(self, query):
        documents = self.retriever.retrieve(query)

        for document in documents:
            title = document["title"]
            text = document["text"]
            sentences = text_to_sentences(text).split("\n")
            is_verified = self.verifier.verify(document, query)
            if is_verified:
                yield self.extractor.extract(document)
        predicted_class_idx, probs = self.verifier.verify(query, df)
        if predicted_class_idx is None:
            return None
        return self.extractor.extract(df, predicted_class_idx, probs)