from fugashi import Tagger


class Tokenizer():
    def __init__(self):
        self.tagger = Tagger("-Owakati")

    def tokenize(self, text):
        tokens = self.tagger.parse(text).split(" ")
        return tokens
