from tokenizer import Tokenizer
from scorer import WordOverlappingScorer, EmbeddingBasedScorer

if __name__ == "__main__":
    # Build segmentor provided
    print("Building tokenizer")
    tokenizer = Tokenizer()

    # Process sentence pairs
    hyps = [
        "映画を見ますか",
        "映画はどんなのを見ますか",
        "映画はどれくらい見ますか",
        "ゴルフは見ますか",
        "サッカーは見ますか",
        "ゴルフで好きな選手はいますか"
    ]
    refs = ["オリンピックは見ますか"]*len(hyps)
    
    hyps = [tokenizer.tokenize(sent) for sent in hyps]
    refs = [tokenizer.tokenize(sent) for sent in refs]

    # You can pass `vocab` as an argument to EmbeddingBasedScorer() to load in-vocab words only for accelerating embedding loading.
    # If `vocab` is None (by default), all the embeddings will be loaded, which will take a longer time.
    vocab = set()
    for sent in hyps+refs:
        for token in sent:
            vocab.add(token)

    # Build scorers
    print("Building scorer")
    word_overlap_scorer = WordOverlappingScorer()
    embedding_based_scorer = EmbeddingBasedScorer(vocab=vocab)

    # Calculate similarities between sentence pairs
    b2_scores = word_overlap_scorer.score(refs, hyps, n=2)
    grd_scores, ext_scores, avg_scores = embedding_based_scorer.score(refs, hyps)

    # Print results
    for ref, hyp, b2, grd, ext, avg in zip(refs, hyps, b2_scores, grd_scores, ext_scores, avg_scores):
        mean_emb = (grd + ext + avg) / 3.0
        ref_tokens = " ".join(ref)
        hyp_tokens = " ".join(hyp)
        print("{} || {}".format(ref_tokens, hyp_tokens))
        print("> BLEU 2: {:.4g}, greedy: {:.4g}, extrema: {:.4g}, average: {:.4g}, mean embedding: {:4g}".format(b2, grd, ext, avg, mean_emb))
        print()
