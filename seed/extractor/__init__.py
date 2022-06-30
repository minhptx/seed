from .unifiedqa import UQAExtractor

class Extractor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get(retriever_model: str, *args, **kwargs):
        if retriever_model == "unifiedqa":
            class_instance = UQAExtractor

        return class_instance(*args, **kwargs)
