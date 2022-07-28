from .wikisearch import WikiRetriever
from .dpr import DPRRetriever

class Retriever:
    name2class = {
        "dpr": DPRRetriever,
        "wiki": WikiRetriever
    }

    def __init__(self) -> None:
        pass

    def search(self, df, k: int = 10) -> list:
        raise NotImplementedError()

    @staticmethod
    def get(retriever_model: str, *args, **kwargs):
        return Retriever.name2class[retriever_model](*args, **kwargs)

    