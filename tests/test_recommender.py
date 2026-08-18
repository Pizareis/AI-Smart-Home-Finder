import pandas as pd
import pytest

from src.nlp_parser import parse_user_input
from src.recommender import rank_listings, recommend

SAMPLE_LISTINGS = pd.DataFrame([
    {"district": "Nilüfer", "room_count": "1+1", "price": 19000, "budget": 1.0, "transport": 0.7, "distance": 0.9, "safety": 0.6, "features": 0.5, "social": 0.4},
    {"district": "Nilüfer", "room_count": "2+1", "price": 21000, "budget": 0.9, "transport": 0.8, "distance": 0.8, "safety": 0.7, "features": 0.6, "social": 0.5},
    {"district": "Nilüfer", "room_count": "1+1", "price": 47000, "budget": 0.1, "transport": 0.9, "distance": 0.9, "safety": 0.9, "features": 0.9, "social": 0.9},
    {"district": "Nilüfer", "room_count": "3+1", "price": 18000, "budget": 1.0, "transport": 0.9, "distance": 0.9, "safety": 0.9, "features": 0.9, "social": 0.9},
    {"district": "Görükle", "room_count": "1+1", "price": 15000, "budget": 1.0, "transport": 0.9, "distance": 0.7, "safety": 0.8, "features": 0.5, "social": 0.6},
    {"district": "Beşevler", "room_count": "2+1", "price": 14000, "budget": 1.0, "transport": 0.85, "distance": 0.6, "safety": 0.9, "features": 0.4, "social": 0.3},
])

SAMPLE_DISTRICT_STATS = pd.DataFrame([
    {"district": "Nilüfer", "avg_price": 20000, "budget": 0.95, "transport": 0.75, "distance": 0.85, "safety": 0.65, "features": 0.55, "social": 0.45},
    {"district": "Görükle", "avg_price": 15000, "budget": 1.0, "transport": 0.9, "distance": 0.7, "safety": 0.8, "features": 0.5, "social": 0.6},
    {"district": "Beşevler", "avg_price": 14000, "budget": 1.0, "transport": 0.85, "distance": 0.6, "safety": 0.9, "features": 0.4, "social": 0.3},
])

USER_TEXT = (
    "24 yaşındayım, Bursa'da öğrenciyim. Üniversitem Nilüfer'de. "
    "Aylık maksimum 20.000 TL verebilirim. 1+1 veya 2+1 istiyorum. "
    "Ulaşım ve güvenlik benim için önemli."
)


def test_rank_listings_sorted_descending():
    ranked = rank_listings(SAMPLE_LISTINGS)
    assert list(ranked["score"]) == sorted(ranked["score"], reverse=True)


def test_rank_listings_empty_input():
    empty = SAMPLE_LISTINGS.iloc[0:0]
    ranked = rank_listings(empty)
    assert ranked.empty


def test_recommend_target_group_matches_district():
    profile = parse_user_input(USER_TEXT)
    result = recommend(SAMPLE_LISTINGS, SAMPLE_DISTRICT_STATS, profile)
    assert (result["target"]["district"] == "Nilüfer").all()
    assert not result["target"].empty


def test_recommend_suggested_excludes_target_and_alternatives():
    profile = parse_user_input(USER_TEXT)
    result = recommend(SAMPLE_LISTINGS, SAMPLE_DISTRICT_STATS, profile)
    suggested_districts = set(result["suggested"]["district"])
    assert profile.target_district not in suggested_districts
    assert not suggested_districts.intersection(profile.alternative_districts)


def test_recommend_suggested_reason_not_empty_when_suggestions_exist():
    profile = parse_user_input(USER_TEXT)
    result = recommend(SAMPLE_LISTINGS, SAMPLE_DISTRICT_STATS, profile)
    if not result["suggested"].empty:
        assert result["suggested_reason"] != ""


def test_recommend_filters_out_listings_over_budget():
    profile = parse_user_input(USER_TEXT)  # butce: 20.000 TL
    result = recommend(SAMPLE_LISTINGS, SAMPLE_DISTRICT_STATS, profile)
    all_prices = pd.concat([result["target"]["price"], result["alternatives"]["price"], result["suggested"]["price"]])
    assert (all_prices <= profile.budget_max * 1.10).all()


def test_recommend_filters_out_wrong_room_type():
    profile = parse_user_input(USER_TEXT)  # oda: 1+1 veya 2+1
    result = recommend(SAMPLE_LISTINGS, SAMPLE_DISTRICT_STATS, profile)
    all_rooms = pd.concat([result["target"]["room_count"], result["alternatives"]["room_count"], result["suggested"]["room_count"]])
    assert set(all_rooms).issubset(set(profile.room_options))
