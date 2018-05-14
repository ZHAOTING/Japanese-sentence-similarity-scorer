import MeCab

class MecabSegmentor():
    def __init__():
        tagger = MeCab.Tagger('-d /usr/lib/mecab/dic/mecab-ipadic-neologd')

    def get_tokens(text):
        tagger.parse('')

        parsed = tagger.parseToNode(text)
        tokens = []

        while parsed:
            tokens.append(parsed.surface)
            parsed = parsed.next

        return tokens
