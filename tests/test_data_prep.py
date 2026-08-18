import numpy as np
import pytest

from src.data_prep import (
    DISTRICT_PROFILES,
    LISTINGS_PER_DISTRICT,
    ROOM_TYPES,
    build_district_stats,
    engineer_features,
    generate_raw_listings,
)

SUB_SCORE_COLS = ["budget", "transport", "distance", "safety", "features", "social"]


@pytest.fixture(scope="module")
def raw_listings():
    return generate_raw_listings(np.random.default_rng(0))


@pytest.fixture(scope="module")
def engineered(raw_listings):
    return engineer_features(raw_listings)


def test_row_count_matches_districts_times_listings(raw_listings):
    assert len(raw_listings) == len(DISTRICT_PROFILES) * LISTINGS_PER_DISTRICT


def test_raw_listings_no_missing_values(raw_listings):
    assert not raw_listings.isna().any().any()


def test_price_is_positive_and_above_floor(raw_listings):
    assert (raw_listings["price"] >= 3000).all()


def test_room_counts_are_valid(raw_listings):
    assert set(raw_listings["room_count"].unique()) <= set(ROOM_TYPES)


def test_districts_match_profile_keys(raw_listings):
    assert set(raw_listings["district"].unique()) == set(DISTRICT_PROFILES.keys())


def test_engineered_sub_scores_within_unit_range(engineered):
    for col in SUB_SCORE_COLS:
        assert engineered[col].between(0, 1).all()


def test_engineered_has_no_missing_values(engineered):
    assert not engineered.isna().any().any()


def test_engineered_preserves_row_count_and_keeps_display_columns(raw_listings, engineered):
    assert len(engineered) == len(raw_listings)
    assert set(["district", "price", "area", "room_count"]) <= set(engineered.columns)


def test_district_stats_has_one_row_per_district(engineered):
    stats = build_district_stats(engineered)
    assert len(stats) == len(DISTRICT_PROFILES)
    assert set(stats["district"]) == set(DISTRICT_PROFILES.keys())
    for col in SUB_SCORE_COLS:
        assert stats[col].between(0, 1).all()
