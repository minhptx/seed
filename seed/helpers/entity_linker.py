import itertools
import blink.main_dense as main_dense
import blink.ner as NER
import argparse
import os

dirname = os.path.dirname(__file__)
models_path = os.path.join(dirname, "../../models/el/") # the path where you stored the BLINK models


class EntityLinker:

    def __init__(self):
        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 10,
            "biencoder_model": models_path+"biencoder_wiki_large.bin",
            "biencoder_config": models_path+"biencoder_wiki_large.json",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
            "crossencoder_config": models_path+"crossencoder_wiki_large.json",
            "fast": True, # set this to be true if speed is a concern
            "output_path": "logs/" # logging directory
        }

        self.args = argparse.Namespace(**config)

        self.models = main_dense.load_models(self.args, logger=None)
        self.ner_model = NER.get_model()

    def link_entities(self, text):
        samples = main_dense._annotate(self.ner_model, [text])
        if not samples:
            return []

        _, _, _, _, _, predictions, scores, = main_dense.run(self.args, None, *self.models, test_data=samples)

        return itertools.chain.from_iterable(predictions)
