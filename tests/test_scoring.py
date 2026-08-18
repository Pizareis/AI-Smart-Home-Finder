import pandas as pd
import pytest

from scoring import _normalize, _score, recommend_listings, recommend_neighborhoods


def test_normalize_higher_is_better_maps_min_max_to_0_and_1():
    series = pd.Series([10, 20, 30])
    normed = _normalize(series, higher_is_better=True)
    assert normed.iloc[0] == pytest.approx(0.0)
    assert normed.iloc[-1] == pytest.approx(1.0)


def test_normalize_lower_is_better_inverts():
    series = pd.Series([10, 20, 30])
    normed = _normalize(series, higher_is_better=False)
    assert normed.iloc[0] == pytest.approx(1.0)
    assert normed.iloc[-1] == pytest.approx(0.0)


def test_normalize_constant_series_returns_all_ones():
    series = pd.Series([5, 5, 5])
    normed = _normalize(series, higher_is_better=True)
    assert (normed == 1.0).all()


def test_score_raises_when_no_positive_weights():
    df = pd.DataFrame({"safety_score": [1, 2, 3]})
    with pytest.raises(ValueError):
        _score(df, {"safety": 0}, granularity="neighborhood")


def test_recommend_neighborhoods_sorted_descending_and_bounded():
    result = recommend_neighborhoods({"affordability": 1, "safety": 1}, top_n=3)
    assert len(result) == 3
    assert result["match_score"].is_monotonic_decreasing
    assert result["match_score"].between(0, 100).all()


def test_recommend_listings_respects_budget_and_room_type():
    result = recommend_listings({"affordability": 1}, budget_try=20000, room_type="2+1")
    assert (result["rent_try"] <= 20000).all()
    assert (result["room_type"] == "2+1").all()


def test_recommend_listings_empty_when_budget_impossible():
    result = recommend_listings({"affordability": 1}, budget_try=1)
    assert result.empty
