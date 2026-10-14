"""BM25 tokenization and index behaviour."""

from backend.retrieval.bm25_search import BM25Index, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Kernel of an SVR") == ["kernel", "svr"]


def test_tokenize_keeps_compounds_and_their_parts():
    assert tokenize("F1-score") == ["f1-score", "f1", "score"]
    assert tokenize("learning rate 1e-3") == ["learning", "rate", "1e-3", "1e", "3"]


def test_tokenize_keeps_technical_tokens_and_numbers():
    assert tokenize("L2 penalty, C=10, epsilon 0.1") == ["l2", "penalty", "c", "10", "epsilon", "0.1"]


def test_plain_numbers_are_not_split():
    # Regression: "0.06248" used to add "0" (matching every decimal) and "06248".
    assert tokenize("GOOGL 0.06248 0.03752 v2.5.1") == ["googl", "0.06248", "0.03752", "v2.5.1", "v2", "5", "1"]
    assert tokenize("version 2.5.1") == ["version", "2.5.1"]


def test_number_noise_does_not_favour_numeric_chunks():
    index = BM25Index()
    index.add([
        (1, "GOOGL 0.06248 0.03752"),
        (2, " ".join(["0.11111"] * 30) + " linear"),  # many decimals, no matching values
    ])

    assert [vector_id for vector_id, _ in index.search("GOOGL 0.06248", top_k=2)] == [1]


def test_tokenize_handles_non_ascii_words():
    assert "ε" in tokenize("the ε-insensitive loss")


def make_index() -> BM25Index:
    index = BM25Index()
    index.add([
        (1, "Support vector regression uses an epsilon-insensitive loss function."),
        (2, "The RBF kernel maps inputs into a high-dimensional feature space."),
        (3, "Stock prices were collected daily from the exchange."),
        (4, "Neural networks were trained with dropout to reduce overfitting."),
    ])
    return index


def test_exact_rare_term_ranks_its_chunk_first():
    hits = make_index().search("RBF kernel", top_k=3)

    assert hits[0][0] == 2
    assert all(score > 0 for _, score in hits)


def test_only_matching_chunks_are_returned():
    hits = make_index().search("dropout", top_k=10)

    assert [vector_id for vector_id, _ in hits] == [4]


def test_compound_query_matches_split_text():
    hits = make_index().search("epsilon insensitive", top_k=1)

    assert hits[0][0] == 1


def test_allowed_ids_restrict_results():
    hits = make_index().search("kernel loss stock", top_k=10, allowed_ids={1, 3})

    assert {vector_id for vector_id, _ in hits} <= {1, 3}
    assert hits


def test_stopword_only_query_returns_nothing():
    assert make_index().search("the of and", top_k=5) == []


def test_empty_index_returns_nothing():
    assert BM25Index().search("kernel", top_k=5) == []


def test_remove_and_add_update_the_index():
    index = make_index()
    index.remove([2])

    assert index.search("RBF kernel", top_k=5) == []
    assert index.count == 3

    index.add([(9, "A polynomial kernel is another choice.")])
    assert index.search("kernel", top_k=5)[0][0] == 9


def test_term_in_half_the_chunks_still_scores():
    # Regression: rank-bm25's default IDF is exactly 0 here (N=2, n=1).
    index = BM25Index()
    index.add([(1, "gradient descent updates parameters"), (2, "regularization reduces overfitting")])

    assert [vector_id for vector_id, _ in index.search("gradient descent", top_k=2)] == [1]


def test_term_in_most_chunks_keeps_positive_weight():
    index = BM25Index()
    index.add([(1, "kernel alpha"), (2, "kernel beta"), (3, "kernel gamma")])

    hits = index.search("kernel", top_k=3)

    assert len(hits) == 3 and all(score > 0 for _, score in hits)


def test_rarer_terms_weigh_more():
    index = BM25Index()
    index.add([(1, "model kernel"), (2, "model loss"), (3, "model data")])

    # "kernel" (1 chunk) is rarer than "model" (3 chunks), so chunk 1 ranks first.
    assert index.search("model kernel", top_k=3)[0][0] == 1


def test_results_are_sorted_and_truncated():
    hits = make_index().search("support vector stock prices neural kernel", top_k=2)

    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]
