"""Weighted scoring: turn user priorities into a match score per neighborhood/listing.

User priorities are expressed as a dict of {priority_name: weight}, e.g.
{"affordability": 3, "university_proximity": 2, "safety": 1}. Missing priorities
default to weight 0. This is the core the Streamlit UI / recommendation layer
calls after collecting budget, room count, and priority sliders from the user.
"""

import sys
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Canonical priority name -> (neighborhood-level column, listing-level column, higher_is_better)
PRIORITY_MAP = {
    "affordability": ("avg_rent_per_m2_try", "rent_try", False),
    "university_proximity": ("avg_distance_to_university_km", "distance_to_university_km", False),
    "center_proximity": ("avg_distance_to_center_km", "distance_to_center_km", False),
    "hospital_proximity": ("avg_distance_to_hospital_km", "distance_to_hospital_km", False),
    "green_space": ("green_space_score", "green_space_score", True),
    "safety": ("safety_score", "safety_score", True),
    "public_transport": ("public_transport_score", "public_transport_score", True),
    "quietness": ("noise_level", "noise_level", False),
}


def load_neighborhood_profiles(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "processed" / "neighborhood_profiles_with_listing_stats.csv"
    return pd.read_csv(path)


def load_listings(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "raw" / "bursa_listings_synthetic.csv"
    return pd.read_csv(path)


def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    normed = (series - lo) / (hi - lo)
    return normed if higher_is_better else 1 - normed


def _score(df: pd.DataFrame, weights: dict, granularity: str) -> pd.DataFrame:
    col_index = 0 if granularity == "neighborhood" else 1
    active = {
        name: (col_map[col_index], col_map[2], weight)
        for name, weight in weights.items()
        if weight > 0 and (col_map := PRIORITY_MAP.get(name)) is not None
    }
    if not active:
        raise ValueError("No valid positive-weight priorities given")

    total_weight = sum(weight for _, _, weight in active.values())
    weighted_sum = pd.Series(0.0, index=df.index)
    for column, higher_is_better, weight in active.values():
        weighted_sum += _normalize(df[column], higher_is_better) * weight

    scored = df.copy()
    scored["match_score"] = (weighted_sum / total_weight * 100).round(1)
    return scored.sort_values("match_score", ascending=False).reset_index(drop=True)


def recommend_neighborhoods(weights: dict, top_n: int = 5) -> pd.DataFrame:
    profiles = load_neighborhood_profiles()
    scored = _score(profiles, weights, granularity="neighborhood")
    display_cols = ["neighborhood", "district", "match_score", "avg_rent_try", "listing_count"]
    return scored[display_cols].head(top_n)


def recommend_listings(
    weights: dict,
    budget_try: float | None = None,
    room_type: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    listings = load_listings()
    if budget_try is not None:
        listings = listings[listings["rent_try"] <= budget_try]
    if room_type is not None:
        listings = listings[listings["room_type"] == room_type]
    if listings.empty:
        return listings

    scored = _score(listings, weights, granularity="listing")
    display_cols = [
        "listing_id", "neighborhood", "room_type", "size_m2", "rent_try", "match_score",
    ]
    return scored[display_cols].head(top_n)


if __name__ == "__main__":
    student_priorities = {"affordability": 3, "university_proximity": 3, "public_transport": 1}
    print("Öğrenci profiline göre önerilen mahalleler:")
    print(recommend_neighborhoods(student_priorities))

    family_priorities = {"safety": 3, "green_space": 2, "quietness": 2, "hospital_proximity": 1}
    print("\nAile profiline göre önerilen ilanlar (bütçe <= 25000 TL, 3+1):")
    print(recommend_listings(family_priorities, budget_try=25000, room_type="3+1"))
