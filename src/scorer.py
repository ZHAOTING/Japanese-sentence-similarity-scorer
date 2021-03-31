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
        return self.batch_bleu(hyps, refs, n)

    def multi_ref_score(self, mrefs, hyps, n=2):
        return self.batch_multi_ref_bleu(hyps, mrefs, n)

    def batch_bleu(self, hyps, refs, n=2):
        """Calculate BLEU-n scores in a batch

        Arguments:
            hyps {list of list} -- hypotheses, each hypothesis is a list of tokens
            refs {list of list} -- references, each reference is a list of tokens
            n {int} -- n for BLEU-n (default: 2)

        Returns:
            {list of float} list of BLEU scores
        """
        assert len(hyps) == len(refs)
        
        weights = [1./n]*n
        scores = []
        for hyp_tokens, ref_tokens in zip(hyps, refs):
            if len(hyp_tokens) == 0:
                score = 0.0
            else:
                try:
                    score = sentence_bleu(
                        ref_tokens,
                        hyp_tokens,
                        weights=weights,
                        smoothing_function=SmoothingFunction().method7
                    )
                except e:
                    raise Exception(f"BLEU score error: {e}")
            scores.append(score)
        return scores

    def batch_multi_ref_bleu(self, hyps, mrefs, n=2):
        """Calculate multiple-referenced BLEU-n scores in a batch

        Arguments:
            hyps {list of str} -- hypotheses, each hypothesis is a list of tokens
            mrefs {list of list of list} -- multi-references, each multi-reference is a list 
                                            of references, each reference is a list of tokens
            n {int} -- n for BLEU-n (default: 2)

        Returns:
            {list of float} list of BLEU scores
        """
        assert len(hyps) == len(mrefs)

        weights = [1./n]*n
        scores = []
        for hyp_tokens, mref_tokens in zip(hyps, mrefs):
            if len(hyp_tokens) == 0:
                score = 0.0
            else:
                try:
                    score = sentence_bleu(
                        mref_tokens,
                        hyp_tokens,
                        weights=weights,
                        smoothing_function=SmoothingFunction().method1
                    )
                except e:
                    raise Exception(f"BLEU score error: {e}")
            scores.append(score)
        return scores


class EmbeddingBasedScorer():
    def __init__(self, embedding_path=Config.embedding_filepath, vocab=None):
        self.w2v = self._get_w2v(embedding_path, vocab)
        self.vocab_size = len(self.w2v)
        self.emb_dim = next(iter(self.w2v.values())).shape[0]
        print(f"Embedding-based scorer loaded {len(self.w2v)} embeddings from {embedding_path}")

    def _get_w2v(self, path, vocab=None):
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

        missing_words = vocab.difference(set(w2v.keys()))
        print(f"Words not found in pretrained embeddings: {list(missing_words)}")
        
        return w2v

    def _tokens2emb(self, tokens):
        embs = [self.w2v[token] for token in tokens if token in self.w2v]
        if len(embs) == 0:
            embs = [[0.]*self.emb_mat.shape[1]]
        return embs

    def _cosine_similarity(self, hyps, refs):
        sims = np.sum(hyps * refs, axis=1) / (np.sqrt((np.sum(hyps * hyps, axis=1) * np.sum(refs * refs, axis=1))) + 1e-10)
        return sims

    def _embedding_metric(self, hyps_emb, refs_emb, method='average'):
        if method == 'average':
            hyps_avg_emb = [np.mean(hyp, axis=0) for hyp in hyps_emb]
            refs_avg_emb = [np.mean(ref, axis=0) for ref in refs_emb]
            sims = self._cosine_similarity(np.array(hyps_avg_emb), np.array(refs_avg_emb))
            return sims.tolist()
        elif method == "multi_ref_average":
            hyps_avg_emb = [np.mean(hyp, axis=0) for hyp in hyps_emb]
            mrefs_avg_emb = [[np.mean(ref, axis=0) for ref in refs] for refs in refs_emb]
            msims = []
            for hyp_avg_emb, mref_avg_emb in zip(hyps_avg_emb, mrefs_avg_emb):
                msim = self._cosine_similarity(np.array([hyp_avg_emb]*len(mref_avg_emb)), np.array(mref_avg_emb))
                msims.append(msim.tolist())
            return msims
        elif method == 'extrema':
            hyps_ext_emb = []
            refs_ext_emb = []
            for hyp, ref in zip(hyps_emb, refs_emb):
                h_max = np.max(hyp, axis=0)
                h_min = np.min(hyp, axis=0)
                h_plus = np.absolute(h_min) <= h_max
                h = h_max * h_plus + h_min * np.logical_not(h_plus)
                hyps_ext_emb.append(h)

                r_max = np.max(ref, axis=0)
                r_min = np.min(ref, axis=0)
                r_plus = np.absolute(r_min) <= r_max
                r = r_max * r_plus + r_min * np.logical_not(r_plus)
                refs_ext_emb.append(r)
            sims = self._cosine_similarity(np.array(hyps_ext_emb), np.array(refs_ext_emb))
            return sims.tolist()
        elif method == "multi_ref_extrema":
            hyps_ext_emb = []
            mrefs_ext_emb = []
            for hyp, mref in zip(hyps_emb, refs_emb):
                h_max = np.max(hyp, axis=0)
                h_min = np.min(hyp, axis=0)
                h_plus = np.absolute(h_min) <= h_max
                h = h_max * h_plus + h_min * np.logical_not(h_plus)
                hyps_ext_emb.append(h)

                mref_ext_emb = []
                for ref in mref:
                    r_max = np.max(ref, axis=0)
                    r_min = np.min(ref, axis=0)
                    r_plus = np.absolute(r_min) <= r_max
                    r = r_max * r_plus + r_min * np.logical_not(r_plus)
                    mref_ext_emb.append(r)
                mrefs_ext_emb.append(mref_ext_emb)
            msims = []
            for hyp_ext_emb, mref_ext_emb in zip(hyps_ext_emb, mrefs_ext_emb):
                msim = self._cosine_similarity(np.array([hyp_ext_emb]*len(mref_ext_emb)), np.array(mref_ext_emb))
                msims.append(msim.tolist())
            return msims
        elif method == 'greedy':
            sims = []
            for hyp, ref in zip(hyps_emb, refs_emb):
                hyp = np.array(hyp)
                ref = np.array(ref).T
                sim = (np.matmul(hyp, ref) / (np.sqrt(np.matmul(np.sum(hyp * hyp, axis=1, keepdims=True), np.sum(ref * ref, axis=0, keepdims=True)))+1e-10))
                sim = np.max(sim, axis=0).mean()
                sims.append(sim)
            return sims
        elif method == "multi_ref_greedy":
            msims = []
            for hyp, mref in zip(hyps_emb, refs_emb):
                hyp = np.array(hyp)
                msim = []
                for ref in mref:
                    ref = np.array(ref).T
                    sim = (np.matmul(hyp, ref) / (np.sqrt(np.matmul(np.sum(hyp * hyp, axis=1, keepdims=True), np.sum(ref * ref, axis=0, keepdims=True)))+1e-10))
                    sim = np.max(sim, axis=0).mean()
                    msim.append(sim)
                msims.append(msim)
            return msims
        else:
            raise NotImplementedError

    def score(self, refs, hyps):
        """Calculate Average/Extrema/Greedy embedding similarities in a batch

        Arguments:
            refs {list of list} -- references, each reference is a list of tokens
            hyps {list of list} -- hypotheses, each hypothesis is a list of tokens

        Returns:
            {list of float} Average similarities
            {list of float} Extrema similarities
            {list of float} Greedy similarities
        """
        assert len(hyps) == len(refs)
        
        hyps_emb = [self._tokens2emb(tokens) for tokens in hyps]
        refs_emb = [self._tokens2emb(tokens) for tokens in refs]

        emb_greedy_scores = self._embedding_metric(hyps_emb, refs_emb, "greedy")
        emb_ext_scores = self._embedding_metric(hyps_emb, refs_emb, "extrema")
        emb_avg_scores = self._embedding_metric(hyps_emb, refs_emb, "average")        

        return emb_greedy_scores, emb_ext_scores, emb_avg_scores

    def multi_ref_score(self, mrefs, hyps):
        """Calculate multi-referenced Average/Extrema/Greedy embedding similarities in a batch

        Arguments:
            mrefs {list of list of list} -- multi-references, each multi-reference is a list 
                                            of references, each reference is a list of tokens
            hyps {list of str} -- hypotheses, each hypothesis is a list of tokens

        Returns:
            {list of float} Average similarities
            {list of float} Extrema similarities
            {list of float} Greedy similarities
        """
        assert len(hyps) == len(mrefs)

        hyps_emb = [self._tokens2emb(tokens) for tokens in hyps]
        mrefs_emb = [[self._tokens2emb(tokens) for tokens in mref_tokens] for mref_tokens in mrefs]

        emb_greedy_scores = self._embedding_metric(hyps_emb, mrefs_emb, "multi_ref_greedy")
        emb_ext_scores = self._embedding_metric(hyps_emb, mrefs_emb, "multi_ref_extrema")
        emb_avg_scores = self._embedding_metric(hyps_emb, mrefs_emb, "multi_ref_average")        

        return emb_greedy_scores, emb_ext_scores, emb_avg_scores