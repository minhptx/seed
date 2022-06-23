from .dpr import DPRRetriever

class RetrieverFactory:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get(retriever_model: str, *args, **kwargs):
        if retriever_model == "dpr":
            class_instance = DPRRetriever

        return class_instance(*args, **kwargs)

    