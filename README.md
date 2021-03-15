# Japanese-sentence-similarity-scorer

A tool for calculating Japanese sentence similarity.

Metrics include: 
* word overlapping score
  * BLEU-n (BLEU-2 by default)
* cosine similarity scores between word-embedding-based sentence representations: 
  * Greedy matching
  * Vector extrema
  * Average embedding

Tokenization is based on [Fugashi](https://github.com/polm/fugashi), a mecab-based tokenizer.

Japanese word embeddings are from [Fasttext](https://fasttext.cc/docs/en/crawl-vectors.html).

## Install

* Get python 3
* Run `pip install -r ./requirements.txt` to obtain required packages.
* Run `python -m unidic download` to obtain Japanese dictionary for Fugashi tokenizer.
* Run `cd src; python download_embeddings.py` to obtain Japanese word embeddings.

## Example

Run 
```
cd src/
python test.py
```

You are supposed to see the tokenization results and similarity scores of the example sentence pairs as following:

```
Building tokenizer
Building scorer
Embedding-based scorer loaded 18 embeddings from ../data/cc.ja.300.vec
オリンピック は 見 ます か || 映画 を 見 ます か
> BLEU 2: 0.6438, greedy: 0.778, extrema: 0.7814, average: 0.9111, mean embedding: 0.823488

オリンピック は 見 ます か || 映画 は どんな の を 見 ます か
> BLEU 2: 0.5532, greedy: 0.8227, extrema: 0.7691, average: 0.9141, mean embedding: 0.835288

オリンピック は 見 ます か || 映画 は どれ くらい 見 ます か
> BLEU 2: 0.6203, greedy: 0.8324, extrema: 0.7551, average: 0.9247, mean embedding: 0.83743

オリンピック は 見 ます か || ゴルフ は 見 ます か
> BLEU 2: 0.7108, greedy: 0.8645, extrema: 0.9324, average: 0.9769, mean embedding: 0.924599

オリンピック は 見 ます か || サッカー は 見 ます か
> BLEU 2: 0.7108, greedy: 0.8942, extrema: 0.9906, average: 0.9912, mean embedding: 0.958666

オリンピック は 見 ます か || ゴルフ で 好き な 選手 は い ます か
> BLEU 2: 0.46, greedy: 0.751, extrema: 0.6163, average: 0.885, mean embedding: 0.750759
```

## Dependencies:

* python==3.7
* fugashi
* unidic
* numpy
* sklearn
* nltk