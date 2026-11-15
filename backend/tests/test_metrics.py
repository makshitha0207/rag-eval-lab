from app.evaluation.metrics import hit_at_k, is_relevant, precision_at_k, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k([False, True, False], 1) == 0.0
    assert hit_at_k([False, True, False], 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([False, False]) == 0.0


def test_precision_at_k():
    assert precision_at_k([True, False, True, False], 4) == 0.5


def test_is_relevant():
    evidence = "the quick brown fox jumps over the lazy dog"
    assert is_relevant("a quick brown fox jumps over a lazy dog today", evidence)
    assert not is_relevant("completely unrelated text about cooking", evidence)