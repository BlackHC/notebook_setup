import operator

from blackhc.project.utils.early_stopping import EarlyStopping


def test_early_stopping_improves():
    # Test case where score improves every time
    early_stopping = EarlyStopping(patience=3, lower_is_better=True)
    scores = [0.5, 0.4, 0.3, 0.2, 0.1]
    for score in scores:
        assert early_stopping.step(score) > 0
    assert early_stopping.best_score == 0.1
    assert early_stopping.cmp == operator.lt


def test_early_stopping_improves_with_higher_is_better():
    # Test case where score improves every time and higher is better
    early_stopping = EarlyStopping(patience=3, lower_is_better=False)
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    for score in scores:
        assert early_stopping.step(score) > 0
    assert early_stopping.best_score == 0.5
    assert early_stopping.cmp == operator.gt


def test_early_stopping_does_not_improve():
    import pytest

    # Test case where score never improves
    early_stopping = EarlyStopping(patience=2, lower_is_better=True)
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    for i, score in enumerate(scores):
        if i < 3:
            assert early_stopping.step(score)
        elif i == 3:
            assert not early_stopping.step(score)
        else:
            with pytest.raises(AssertionError):
                early_stopping.step(score)
    assert early_stopping.best_score == 0.5
    assert early_stopping.cmp == operator.lt


def test_early_stopping_improves_sometimes():
    # Test case where score improves sometimes
    early_stopping = EarlyStopping(patience=1, lower_is_better=True)
    scores = [0.5, 0.6, 0.4, 0.7, 0.3]
    for i, score in enumerate(scores):
        assert early_stopping.step(score)
        assert early_stopping.patience != (i % 2)
    assert early_stopping.best_score == 0.3
    assert early_stopping.cmp == operator.lt


def test_early_stopping_iter():
    # Test case for __iter__ method
    early_stopping = EarlyStopping(patience=1, lower_is_better=True)
    scores = [0.5, 0.6, 0.7, 0.8]
    for score, _ in zip(scores, early_stopping):
        early_stopping.step(score)
    assert early_stopping.best_score == 0.5
    assert early_stopping.cmp == operator.lt
    assert score == 0.7


def test_early_stopping_to_str():
    # Test case for __str__ method
    early_stopping = EarlyStopping(patience=1, lower_is_better=True)
    assert str(early_stopping) == "EarlyStopping(patience=1, lower_is_better=True)"
