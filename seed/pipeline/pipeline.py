from blingfire import text_to_sentences
from seed.helpers.entity_linker import EntityLinker
import re

class Pipeline:
    def __init__(self, retriever, verifier, extractor) -> None:
        self.retriever = retriever
        self.verifier = verifier
        self.extractor = extractor
        self.entity_linker = EntityLinker()

    def run(self, df):
        documents = self.retriever.search(df)

        relevant_sentences = []
        table_entities = list(self.entity_linker.link_entities(" and ".join(df.values.flatten().tolist())))

        for document in documents:
            title = document["title"]
            text = document["text"]
            for sentences in re.split(r"\n|\*|(===)|(==)", text):
                if sentences is None:
                    continue
                for sentence in text_to_sentences(sentences).split("\n"):
                    if sentence is None or len(sentence) < 10:
                        continue
                    
                    sent_entities = list(self.entity_linker.link_entities(sentence))
                    if len(set(sent_entities).intersection(set(table_entities))) == 0:
                        continue
                    relevant_sentences.append(sentence)
                    column2res = self.verifier.verify(sentence, df)
                    print(f"Title: {title} --- Sentence: '{sentence}' --- Label: {column2res}")
                
        return []