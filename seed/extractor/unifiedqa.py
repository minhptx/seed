from transformers import T5ForConditionalGeneration, T5Tokenizer
from flair.data import Sentence
from flair.models import SequenceTagger
import dateparser
from price_parser import parse_price
from number_parser import parse


class UQAExtractor:
    def __init__(self) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-v2-t5-base-1251000")
        self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-v2-t5-base-1251000")
        self.tagger = SequenceTagger.load('ner')

    def check_date(self, values):
        for value in values:
            if value and dateparser.parse(value) is None:
                return False
        return True

    def check_number(self, values):
        for value in values:
            try:
                float(value)
            except:
                parse
                return False

    def check_price(self, values):
        for value in values:
            if parse_price(value).amount is None:
                return False
        return True

    def ner_tag(self, values):
        types = []
        for value in values:
            sentence = Sentence(values)
            self.tagger.predict(sentence)

            for entity in sentence.get_spans('ner'):
                types.append(entity.get_label("ner").value)

        return max(set(types), key=types.count)

    def check_types(self, values):
        if self.check_date(values):
            return "DATE"
        if self.check_price(values):
            return "PRICE"
        if self.check_number(values):
            return "NUMBER"
        return self.ner_tag(values)

    def run_unifiedqa(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def extract(self, sentence, df, column):
        values = df[column].values.tolist()
        type = self.check_types(values)

        if type == "DATE":
            question = "When ?"
        elif type == "PRICE":
            question = "How much ?"
        elif type == "NUMBER":
            question = "How many ?"
        elif type == "PER":
            question = "Who ?"
        elif type == "LOC":
            question = "Where ?"
        elif type == "ORG":
            question = f"Which {column} ?"
        elif type == "MISC":
            question = f"Which {column}?"

        print(question)
        print(df)
        return self.run_unifiedqa(f"{question} \\n {sentence}")
        
    def compare(self, sentence, value, df, column):
        prediction = self.extract(sentence, df, column)
        print(prediction)
        return prediction[0].lower() == value.lower()
