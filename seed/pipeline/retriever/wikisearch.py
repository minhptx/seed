import orjson as json
from elasticsearch_dsl import MultiSearch, Search, connections
from flair.data import Sentence
from flair.models import SequenceTagger
import wikitextparser as wtp

# make a sentence


class WikiRetriever:
    def __init__(self, mode="hybrid", cache_file="") -> None:
        connections.create_connection(hosts=["http://ckg05:9200/"], timeout=100)

        self.ner = SequenceTagger.load('ner')

    def process_content(self, raw_content: str) -> str:
        title, content = json.loads(raw_content)["contents"].split("\n", 1)
        return {"title": title[1:-1], "text": content}

    def search(self, df, k: int = 10) -> list:
        query = " and ".join([x for x in df.values.flatten().tolist() if x])
        ms = MultiSearch(index="test_en")
        ms = ms.add(Search().query("match", text=query))

        sentence = Sentence(query)
        self.ner.predict(sentence)

        entity_count = 0
        entities = []
        for entity in sentence.get_spans('ner'):
            entity_count += 1
            ms = ms.add(Search().query("match", title=entity.text))
            entities.append(entity.text)


        for value in df.values.flatten().tolist():
            # if value.isdigit():
            #     continue
            if value not in entities:
                ms = ms.add(Search().query("match", title=value))
        
        responses = ms.execute()
        result = []
        titles = set()
        for response in responses:
            if not response.hits:
                continue
            for hit in response.hits:
                if  hit["title"] not in titles:
                    result.append({"title": hit["title"], "text": wtp.remove_markup(hit["text"])})
                    titles.add(hit["title"])

        return result
