from seed.retriever.factory import RetrieverFactory


class Pipeline:
    def __init__(self, config) -> None:
        self.retriever = RetrieverFactory.get(config.retriever_model)
        self.verifier = VerifierFactory.get(config.verifier)