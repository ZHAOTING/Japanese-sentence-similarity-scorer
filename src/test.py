import code
import argparse

from scorer import WordOverlappingScorer, EmbeddingBasedScorer, SkipThoughtsScorer
from word_segmentor import MecabSegmentor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_path", help="path to w2v data", required=True)
    args = parser.parse_args()

    word_segmentor = MecabSegmentor()
    word_overlap_scorer = WordOverlappingScorer()
    embedding_based_scorer = EmbeddingBasedScorer(args.w2v_path)

    sentences_1 = [
    "映画を見ますか",
    "映画はどんなのを見ますか",
    "映画はどれくらい見ますか",
    "ゴルフはみますか",
    "サッカーは見ますか"]

    sentences_2 = ["オリンピックは見ますか"]*len(sentences_1)

    inputs_1 = [word_segmentor.get_tokens(sent) for sent in sentences_1]
    inputs_2 = [word_segmentor.get_tokens(sent) for sent in sentences_2]
    print(inputs_1)
    print(inputs_2)

    bleu_2_scores = word_overlap_scorer(inputs_1, inputs_2, n=2)
    greedy_scores, extrema_scores, average_scores = embedding_based_scorer(inputs_1, inputs_2)
    for pair_idx in range(len(inputs_1)):
        print(" ".join(inputs_1[pair_idx]))
        print(" ".join(inputs_2[pair_idx]))
        print("BLEU 2: {}, greedy: {}, extrema: {}, average: {}".format(bleu_2_scores[pair_idx], greedy_scores[pair_idx], extrema_scores[pair_idx], average_scores[pair_idx]))
