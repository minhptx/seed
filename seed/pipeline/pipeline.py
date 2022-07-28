from blingfire import text_to_sentences
import re

class Pipeline:
    def __init__(self, retriever, verifier, extractor) -> None:
        self.retriever = retriever
        self.verifier = verifier
        self.extractor = extractor
        # self.entity_linker = EntityLinker()

    def run(self, df):
        documents = self.retriever.search(df)

        # relevant_sentences = []
        # table_entities = list(self.entity_linker.link_entities(" and ".join(df.values.flatten().tolist())))

        column2error = {}

        for document in documents:
            title = document["title"]
            text = document["text"]
            for sentences in re.split(r"\n|\*|(===)|(==)", text):
                if sentences is None:
                    continue
                for sentence in text_to_sentences(sentences).split("\n"):
                    if sentence is None or len(sentence) < 10:
                        continue
                    
                    column2res = self.verifier.verify(sentence, df)
                    print(f"Title: {title} --- Sentence: '{sentence}' --- Label: {column2res}")
                    if column2res is None:
                        continue
                    for column, res in column2res.items():
                        if res:
                            column2error[column] = res[1]
                        else:
                            column2error[column] = 0
        
        if not column2error:
            return None
        column = sorted(column2error.items(), key=lambda x: x[1], reverse=True)[0][0]
        if column2error[column] == 0:
            return None
        return column