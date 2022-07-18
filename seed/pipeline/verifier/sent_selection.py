from sentence_transformers import SentenceTransformer, util
import os

dirname = os.path.dirname(__file__)


class SentenceSelector:
    def __init__(self) -> None:
        self.sentence_selector = SentenceTransformer(os.path.join(dirname, "../../../models/bart-triplet-2"))

    def check_select(self, sent1, sent2):
        embeddings1 = self.sentence_selector.encode(sent1, convert_to_tensor=True)
        embeddings2 = self.sentence_selector.encode(sent2, convert_to_tensor=True)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)[0][0].item()
        if cosine_scores < 0.7:
            return False
        return True