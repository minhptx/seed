from seed.retriever import Retriever
from seed.verifier import Verifier
from seed.extractor import Extractor
from blingfire import text_to_sentences


class Pipeline:
    def __init__(self, retriever=None, verifier=None, extractor=None) -> None:
        self.retriever = Retriever.get(retriever if retriever else "dpr")
        self.verifier = Verifier.get(verifier if verifier else "tapex")
        self.extractor = Extractor.get(extractor if extractor else "unifiedqa")

    def run(self, df):
        query = " and ".join([f"{x} is {y}" for x, y in df.to_dict(orient="records")[0].items()])
        documents = self.retriever.retrieve(query)

        relevant_sentences = []

        for document in documents:
            title = document["title"]
            text = document["text"]
            sentences = text_to_sentences(text).split("\n")
            for sentence in sentences:

                is_verified, prob = self.verifier.verify(document, query)
                relevant_sentences.append({"title": title, "text": sentence, "is_verified": is_verified, "prob": prob})

        erroneous_cells = []
        for relevant_sentence in relevant_sentences:
            for column in df:
                compare = self.extractor.compare(relevant_sentence["text"], df[column].values.tolist(), df, column)
                if not compare: 
                    erroneous_cells.append({"title": relevant_sentence["title"], "text": relevant_sentence["text"], "column": column})
                
        return erroneous_cells