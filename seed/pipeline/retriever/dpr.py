from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
from pyserini.search.hybrid import HybridSearcher
import orjson as json


class DPRRetriever:
    def __init__(self, mode="hybrid") -> None:
        self.ssearcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr")
        self.encoder = DprQueryEncoder("facebook/dpr-question_encoder-multiset-base")
        self.dsearcher = FaissSearcher.from_prebuilt_index(
            "wikipedia-dpr-multi-bf", self.encoder
        )
        self.searcher = HybridSearcher(self.dsearcher, self.ssearcher)
        self.mode = mode

        if mode == "sparse":
            self.searcher = self.ssearcher
        elif mode == "dense":
            self.searcher = self.dsearcher

    def process_content(self, raw_content: str) -> str:
        title, content = json.loads(raw_content)["contents"].split("\n", 1)
        return {"title": title[1:-1], "text": content}

    def search(self, df, k: int = 10) -> list:
        query = " and ".join([f"{x} is {y}" for x, y in df.to_dict(orient="records")[0].items() if all([not c.isdigit() for c in y])])
        docs = self.searcher.search(query, k)
        result = []
        if self.mode == "sprase":
            for doc in docs:
                result.append(self.process_content(doc.raw))
        elif self.mode == "hybrid":
            for doc in docs:
<<<<<<< HEAD:seed/pipeline/retriever/dpr.py
                if doc.score > 80:
                    result.append(self.process_content(self.ssearcher.doc(doc.docid).raw()))
=======
                print(self.ssearcher.doc(doc.docid).raw())
                result.append(self.process_content(self.ssearcher.doc(doc.docid).raw()))
>>>>>>> a46888adae3f322224e6354ce8244ac5ab226da5:seed/retriever/dpr.py
        return result
