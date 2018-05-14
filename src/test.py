# encoding="utf-8"
import code
import argparse

import numpy as np

from scorer import WordOverlappingScorer, EmbeddingBasedScorer, SkipThoughtsScorer
from word_segmentor import MecabSegmentor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_path", help="path to w2v data", required=True)
    args = parser.parse_args()

    ## Build segmentor provided
    word_segmentor = MecabSegmentor()

    ## Process sentence pairs
    sentences_1 = [
    "映画を見ますか",
    "映画はどんなのを見ますか",
    "映画はどれくらい見ますか",
    "ゴルフは見ますか",
    "サッカーは見ますか",
    "ゴルフで好きな選手はいますか"]
    sentences_2 = ["オリンピックは見ますか"]*len(sentences_1)
    inputs_1 = [word_segmentor.get_tokens(sent) for sent in sentences_1]
    inputs_2 = [word_segmentor.get_tokens(sent) for sent in sentences_2]

    ## You can pass vocab as a parameter of EmbeddingBasedScorer() to accelerate loading w2v data (so that out-of-vocab word embeddings are not loaded).
    ##   If vocab is None or not passed, complete w2v data will be loaded, which will take a longer time
    vocab = {}
    for sent in inputs_1+inputs_2:
        for token in sent:
            vocab[token] = 1

    ## Build scorers
    word_overlap_scorer = WordOverlappingScorer()
    embedding_based_scorer = EmbeddingBasedScorer(args.w2v_path, vocab=vocab)

    ## Calculate similarities of (segmented) sentence pairs
    bleu_2_scores = word_overlap_scorer.score(inputs_1, inputs_2, n=2)
    greedy_scores, extrema_scores, average_scores = embedding_based_scorer.score(inputs_1, inputs_2)

    ## Output results
    for pair_idx in range(len(inputs_1)):
        score_1 = bleu_2_scores[pair_idx]
        score_2 = greedy_scores[pair_idx]
        score_3 = extrema_scores[pair_idx]
        score_4 = average_scores[pair_idx]
        score_5 = np.mean([score_2, score_3, score_4])
        tokens_1 = " ".join(inputs_1[pair_idx])
        tokens_2 = " ".join(inputs_2[pair_idx])
        print("{} || {}".format(tokens_1, tokens_2))
        print("> BLEU 2: {:.4g}, greedy embedding: {:.4g}, extrema embedding: {:.4g}, average embedding: {:.4g}, mean embedding-based: {:4g}".format(score_1, score_2, score_3, score_4, score_5))
