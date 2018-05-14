import code

import numpy as np
import sklearn as sk
import sklearn.metrics
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# import skipthoughts

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
                scores.append(sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method7, weights=[1.0/n]*n))
            except:
                scores.append(0.0)
        return scores

class EmbeddingBasedScorer():
    def __init__(self, w2v_path, vocab=None):
        print("> loading word2vec...")
        self.w2v = self.get_w2v(w2v_path, vocab)
        print("> word2vec loaded.")

    def get_w2v(self, path, vocab=None):
        w2v = {}
        with open(path) as f:
            for line in f.readlines():
                items = line.strip().split(" ")
                if len(items) == 2: # skip gensim w2v head
                    continue
                word = items[0]
                vec = items[1:]
                if vocab is None:
                    w2v[word] = np.array([float(val) for val in vec])
                else:
                    if word in vocab:
                        w2v[word] = np.array([float(val) for val in vec])
        return w2v

    def greedy_matching_score(self, refs, hyps, w2v):
        res1 = self.oneside_greedy_matching_score(refs, hyps, w2v)
        res2 = self.oneside_greedy_matching_score(hyps, refs, w2v)
        res_sum = (res1 + res2)/2.0
        return res_sum

    def oneside_greedy_matching_score(self, refs, hyps, w2v):
        dim = list(w2v.values())[0].shape[0] # embedding dimensions

        scores = []
        for ref, hyp in zip(refs, hyps):
            y_count = 0
            x_count = 0
            o = 0.0
            Y = np.zeros((dim,1))
            for tok in hyp:
                if tok in w2v:
                    Y = np.hstack((Y,(w2v[tok].reshape((dim, 1)))))
                    y_count += 1

            for tok in ref:
                if tok in w2v:
                    x = w2v[tok]
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

    def vector_extrema_score(self, refs, hyps, w2v):
        scores = []
        for i, (ref, hyp) in enumerate(zip(refs, hyps)):
            X, Y = [], []
            x_cnt, y_cnt = 0, 0
            for tok in ref:
                if tok in w2v:
                    X.append(w2v[tok])
                    x_cnt += 1
            for tok in hyp:
                if tok in w2v:
                    Y.append(w2v[tok])
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

    def embedding_average_score(self, refs, hyps, w2v):
        dim = list(w2v.values())[0].shape[0] # embedding dimensions

        scores = []
        for ref, hyp in zip(refs, hyps):
            X = np.zeros((dim,))
            x_cnt, y_cnt = 0, 0
            for tok in ref:
                if tok in w2v:
                    X += w2v[tok]
                    x_cnt += 1
            Y = np.zeros((dim,))
            for tok in hyp:
                if tok in w2v:
                    Y += w2v[tok]
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
        greedy_score = self.greedy_matching_score(refs, hyps, self.w2v)
        extrema_score = self.vector_extrema_score(refs, hyps, self.w2v)
        average_score = self.embedding_average_score(refs, hyps, self.w2v)
        return greedy_score, extrema_score, average_score

class SkipThoughtsScorer():
    def __init__(self):
        print("> loading skip-thoughts model...")
        st_model = skipthoughts.load_model()
        self.st_encoder = skipthoughts.Encoder(st_model)
        print("> skip-thoughts model loaded.")

    def score(self, refs, hyps):
        refs = [" ".join(ref) for ref in refs]
        hyps = [" ".join(hyp) for hyp in hyps]
        ref_vectors = self.st_encoder.encode(refs, verbose=False)
        hyp_vectors = self.st_encoder.encode(hyps, verbose=False)

        scores = []
        for ref_v, hyp_v in zip(ref_vectors, hyp_vectors):
            o = sklearn.metrics.pairwise.cosine_similarity(ref_v.reshape(1, -1), hyp_v.reshape(1, -1))[0][0]
            scores.append(o)

        scores = np.asarray(scores)
        return scores
