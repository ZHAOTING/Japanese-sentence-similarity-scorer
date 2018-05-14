import MeCab

class MecabSegmentor():
    def __init__(self):
        self.tagger = MeCab.Tagger("-Ochasen")

    def get_tokens(self, text):
        self.tagger.parse('')

        parsed = self.tagger.parseToNode(text)
        tokens = []

        while parsed:
            tokens.append(parsed.surface)
            parsed = parsed.next

        tokens = list(filter(lambda w:w != "", tokens))

        return tokens
