from .wikisearch import WikiRetriever
from .dpr import DPRRetriever

class Retriever:
    name2class = {
        "dpr": DPRRetriever,
        "wiki": WikiRetriever
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def get(retriever_model: str, *args, **kwargs):
        return Retriever.name2class[retriever_model](*args, **kwargs)

    