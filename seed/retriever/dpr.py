from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
from pyserini.search.hybrid import HybridSearcher
import orjson as json


class DPRRetriever:
    def __init__(self, mode="hybrid") -> None:
        self.ssearcher = LuceneSearcher.from_prebuilt_index("enwiki-paragraphs")
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

    def search(self, query: str, k: int = 10) -> list:
        docs = self.searcher.search(query, k)
        if self.mode == "sprase":
            for doc in docs:
                yield self.process_content(doc.raw)
        elif self.mode == "hybrid":
            for doc in docs:
                yield self.process_content(self.ssearcher.doc(doc.docid).raw())
