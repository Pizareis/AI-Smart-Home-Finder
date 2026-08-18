import numpy as np
import pytest

from data_prep import (
    LISTINGS_PER_NEIGHBORHOOD,
    NEIGHBORHOOD_PROFILES,
    ROOM_TYPES,
    build_neighborhood_profile_table,
    generate_listings,
)


@pytest.fixture(scope="module")
def listings():
    return generate_listings(np.random.default_rng(0))


def test_row_count_matches_neighborhoods_times_listings(listings):
    expected = len(NEIGHBORHOOD_PROFILES) * LISTINGS_PER_NEIGHBORHOOD
    assert len(listings) == expected


def test_no_missing_values(listings):
    assert not listings.isna().any().any()


def test_rent_is_positive_and_above_floor(listings):
    assert (listings["rent_try"] >= 3000).all()


def test_room_types_are_valid(listings):
    assert set(listings["room_type"].unique()) <= set(ROOM_TYPES)


def test_neighborhoods_match_profile_keys(listings):
    assert set(listings["neighborhood"].unique()) == set(NEIGHBORHOOD_PROFILES.keys())


def test_distances_are_non_negative(listings):
    for col in ["distance_to_university_km", "distance_to_center_km", "distance_to_hospital_km"]:
        assert (listings[col] >= 0).all()


def test_neighborhood_profile_table_has_one_row_per_neighborhood():
    profiles = build_neighborhood_profile_table()
    assert len(profiles) == len(NEIGHBORHOOD_PROFILES)
    assert set(profiles["neighborhood"]) == set(NEIGHBORHOOD_PROFILES.keys())
