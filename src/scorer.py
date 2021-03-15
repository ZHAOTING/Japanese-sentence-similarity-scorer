import code
import io

import numpy as np
import sklearn.metrics
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from config import Config


class WordOverlappingScorer():
    def __init__(self):
        pass

    def score(self, refs, hyps, n=2):
        return self.bleu_n_score(refs, hyps, n)

    def bleu_n_score(self, refs, hyps, n=2):
        assert n in [1, 2, 3, 4]
        scores = []
        for ref, hyp in zip(refs, hyps):
            try:
                score = sentence_bleu(
                    ref, 
                    hyp, 
                    smoothing_function=SmoothingFunction().method7, 
                    weights=[1.0/n]*n
                )
            except Exception as e:
                print(e)
                score = 0.0
            scores.append(score)
        return scores


class EmbeddingBasedScorer():
    def __init__(self, embedding_path=Config.embedding_filepath, vocab=None):
        self.w2v = self.get_w2v(embedding_path, vocab)
        self.vocab_size = len(self.w2v)
        self.emb_dim = next(iter(self.w2v.values())).shape[0]
        print(f"Embedding-based scorer loaded {len(self.w2v)} embeddings from {embedding_path}")

    def get_w2v(self, path, vocab=None):
        fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        w2v = {}
        n_covered_vocab = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            if vocab is None:
                w2v[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
            else:
                if tokens[0] in vocab:
                    w2v[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
                    n_covered_vocab += 1
                if n_covered_vocab >= len(vocab):
                    break
        return w2v

    def greedy_matching_score(self, refs, hyps):
        res1 = self.oneside_greedy_matching_score(refs, hyps)
        res2 = self.oneside_greedy_matching_score(hyps, refs)
        res_sum = (res1 + res2)/2.0
        return res_sum

    def oneside_greedy_matching_score(self, refs, hyps):
        scores = []
        for ref, hyp in zip(refs, hyps):
            y_count = 0
            x_count = 0
            o = 0.0
            Y = np.zeros((self.emb_dim, 1))
            for tok in hyp:
                if tok in self.w2v:
                    Y = np.hstack((Y, (self.w2v[tok].reshape((self.emb_dim, 1)))))
                    y_count += 1

            for tok in ref:
                if tok in self.w2v:
                    x = self.w2v[tok]
                    tmp = sklearn.metrics.pairwise.cosine_similarity(Y.T, x.reshape(1, -1))
                    o += np.max(tmp)
                    x_count += 1

            # if none of the words in response or ground truth have embeddings, count result as zero
            if x_count < 1 or y_count < 1:
                scores.append(0)
                continue

            o /= float(x_count)
            scores.append(o)

        return np.asarray(scores)

    def vector_extrema_score(self, refs, hyps):
        scores = []
        for i, (ref, hyp) in enumerate(zip(refs, hyps)):
            X, Y = [], []
            x_cnt, y_cnt = 0, 0
            for tok in ref:
                if tok in self.w2v:
                    X.append(self.w2v[tok])
                    x_cnt += 1
            for tok in hyp:
                if tok in self.w2v:
                    Y.append(self.w2v[tok])
                    y_cnt += 1

            # if none of the words in ground truth have embeddings, skip
            if x_cnt == 0:
                continue

            # if none of the words have embeddings in response, count result as zero
            if y_cnt == 0:
                scores.append(0)
                continue

            xmax = np.max(X, 0)  # get positive max
            xmin = np.min(X, 0)  # get abs of min
            xtrema = []
            for i in range(len(xmax)):
                if np.abs(xmin[i]) > xmax[i]:
                    xtrema.append(xmin[i])
                else:
                    xtrema.append(xmax[i])
            X = np.array(xtrema)   # get extrema

            ymax = np.max(Y, 0)
            ymin = np.min(Y, 0)
            ytrema = []
            for i in range(len(ymax)):
                if np.abs(ymin[i]) > ymax[i]:
                    ytrema.append(ymin[i])
                else:
                    ytrema.append(ymax[i])
            Y = np.array(ytrema)

            o = sklearn.metrics.pairwise.cosine_similarity(Y.reshape(1, -1), X.reshape(1, -1))[0][0]

            scores.append(o)

        scores = np.asarray(scores)
        return scores

    def embedding_average_score(self, refs, hyps):
        scores = []
        for ref, hyp in zip(refs, hyps):
            X = np.zeros((self.emb_dim,))
            x_cnt, y_cnt = 0, 0
            for tok in ref:
                if tok in self.w2v:
                    X += self.w2v[tok]
                    x_cnt += 1
            Y = np.zeros((self.emb_dim,))
            for tok in hyp:
                if tok in self.w2v:
                    Y += self.w2v[tok]
                    y_cnt += 1

            # if none of the words in ground truth have embeddings, skip
            if x_cnt == 0:
                continue

            # if none of the words have embeddings in response, count result as zero
            if y_cnt == 0:
                scores.append(0)
                continue

            X = np.array(X)/x_cnt
            Y = np.array(Y)/y_cnt
            o = sklearn.metrics.pairwise.cosine_similarity(Y.reshape(1, -1), X.reshape(1, -1))[0][0]
            scores.append(o)

        scores = np.asarray(scores)
        return scores

    def score(self, refs, hyps):
        greedy_score = self.greedy_matching_score(refs, hyps)
        extrema_score = self.vector_extrema_score(refs, hyps)
        average_score = self.embedding_average_score(refs, hyps)
        return greedy_score, extrema_score, average_score
