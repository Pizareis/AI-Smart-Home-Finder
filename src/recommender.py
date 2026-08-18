"""Siralama ve 3 gruplu semt oneri mantigi (Omer).

Grup 1: kullanicinin istedigi semt (target)
Grup 2: kullanicinin belirttigi alternatif semt(ler) (alternatives)
Grup 3: sistemin kendi onerdigi semt(ler), gerekceli (suggested)

Beklenen semalar (Yagmur'un feature engineering asamasindan gelir):
- listings: district, budget, transport, distance, safety, features, social (0-1 normalize alt skorlar)
- district_stats: district, avg_price, budget, transport, distance, safety, features, social
  (semt bazli ortalama/normalize skorlar - bkz. plan bolum 6)
"""

import pandas as pd

from src.nlp_parser import UserProfile
from src.scoring import DEFAULT_WEIGHTS, adjust_weights, compute_listing_score

PRIORITY_LABELS_TR = {
    "budget": "bütçe",
    "transport": "ulaşım",
    "distance": "okula/işe uzaklık",
    "safety": "güvenlik",
    "features": "ev özellikleri",
    "social": "sosyal olanaklar",
}

SUGGESTION_TOP_N = 2
SUGGESTION_BUDGET_TOLERANCE = 1.10


def rank_listings(listings: pd.DataFrame, weights: dict[str, float] = DEFAULT_WEIGHTS) -> pd.DataFrame:
    if listings.empty:
        return listings.assign(score=[])

    ranked = listings.copy()
    ranked["score"] = ranked.apply(lambda row: compute_listing_score(row.to_dict(), weights), axis=1)
    return ranked.sort_values("score", ascending=False).reset_index(drop=True)


def _filter_by_district(ranked_listings: pd.DataFrame, districts: list[str]) -> pd.DataFrame:
    if not districts or ranked_listings.empty:
        return ranked_listings.iloc[0:0]
    return ranked_listings[ranked_listings["district"].isin(districts)].reset_index(drop=True)


def _build_suggestion_reason(row: pd.Series, target_row: pd.Series | None, profile: UserProfile) -> str:
    priority_names = ", ".join(PRIORITY_LABELS_TR.get(p, p) for p in profile.priorities) or "bütçe ve ulaşım"

    if target_row is not None and target_row.get("avg_price", 0) > 0:
        diff_pct = round((target_row["avg_price"] - row["avg_price"]) / target_row["avg_price"] * 100)
        price_clause = f"ortalama kira %{diff_pct} daha uygun" if diff_pct > 0 else "ortalama kira benzer"
    else:
        price_clause = f"ortalama kira {row['avg_price']:.0f} TL civarında"

    return (
        f"Bütçeniz ve {priority_names} önceliğiniz nedeniyle {row['district']}'i de "
        f"değerlendirmenizi öneriyoruz: {price_clause}."
    )


def _suggest_districts(
    district_stats: pd.DataFrame,
    profile: UserProfile,
    weights: dict[str, float],
    exclude: set[str],
) -> tuple[pd.DataFrame, str]:
    if district_stats.empty:
        return district_stats.iloc[0:0], ""

    candidates = district_stats[~district_stats["district"].isin(exclude)].copy()

    if profile.budget_max:
        candidates = candidates[candidates["avg_price"] <= profile.budget_max * SUGGESTION_BUDGET_TOLERANCE]

    if candidates.empty:
        return candidates, ""

    candidates["district_score"] = candidates.apply(lambda row: compute_listing_score(row.to_dict(), weights), axis=1)
    candidates = candidates.sort_values("district_score", ascending=False).head(SUGGESTION_TOP_N)

    target_row = None
    if profile.target_district is not None:
        target_matches = district_stats[district_stats["district"] == profile.target_district]
        if not target_matches.empty:
            target_row = target_matches.iloc[0]

    reasons = [_build_suggestion_reason(row, target_row, profile) for _, row in candidates.iterrows()]
    return candidates.reset_index(drop=True), " ".join(reasons)


def build_listing_comment(row: pd.Series, threshold: float = 0.8) -> str:
    strengths = []
    if row.get("budget", 0) >= threshold:
        strengths.append("bütçenize uygun")
    if row.get("transport", 0) >= threshold:
        strengths.append("ulaşıma yakın")
    if row.get("distance", 0) >= threshold:
        strengths.append("okula/işe yakın")
    if row.get("safety", 0) >= threshold:
        strengths.append("güvenli bir bölgede")

    if not strengths:
        return "Orta düzeyde uygun bir seçenek."
    return "Bu ilan " + ", ".join(strengths) + "."


def recommend(listings: pd.DataFrame, district_stats: pd.DataFrame, profile: UserProfile) -> dict:
    """Returns {"target": df, "alternatives": df, "suggested": df, "suggested_reason": str}."""
    weights = adjust_weights(profile.priorities)
    ranked = rank_listings(listings, weights)

    target_districts = [profile.target_district] if profile.target_district else []
    target_listings = _filter_by_district(ranked, target_districts)
    alternative_listings = _filter_by_district(ranked, profile.alternative_districts)

    excluded = set(filter(None, [profile.target_district, *profile.alternative_districts]))
    suggested_stats, suggested_reason = _suggest_districts(district_stats, profile, weights, excluded)
    suggested_listings = _filter_by_district(ranked, suggested_stats["district"].tolist())

    return {
        "target": target_listings,
        "alternatives": alternative_listings,
        "suggested": suggested_listings,
        "suggested_reason": suggested_reason,
    }
