from src.scoring import DEFAULT_WEIGHTS, adjust_weights, compute_listing_score, normalize_budget_fit


def test_adjust_weights_no_priorities_returns_defaults():
    assert adjust_weights([]) == DEFAULT_WEIGHTS


def test_adjust_weights_sums_to_one():
    weights = adjust_weights(["transport", "safety"])
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_adjust_weights_boosts_prioritized_keys():
    weights = adjust_weights(["transport"])
    assert weights["transport"] > DEFAULT_WEIGHTS["transport"]
    assert weights["social"] < DEFAULT_WEIGHTS["social"]


def test_compute_listing_score_in_range():
    listing = {"budget": 1.0, "transport": 0.8, "distance": 0.5, "safety": 0.9, "features": 0.6, "social": 0.3}
    score = compute_listing_score(listing, DEFAULT_WEIGHTS)
    assert 0.0 <= score <= 1.0


def test_normalize_budget_fit_within_budget():
    assert normalize_budget_fit(15000, 20000) == 1.0


def test_normalize_budget_fit_over_budget():
    assert normalize_budget_fit(30000, 20000) == 0.5


def test_normalize_budget_fit_no_budget_given():
    assert normalize_budget_fit(50000, None) == 1.0
