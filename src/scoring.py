"""Agirlikli ev skoru hesaplama (Omer + Yagmur).

Ev Skoru = %30 Butce + %25 Ulasim + %20 Okula/ise uzaklik + %10 Guvenlik
           + %10 Ev ozellikleri + %5 Sosyal olanaklar

Kullanici bir onceligi vurgularsa (orn. "ulasim") ilgili agirlik artirilir,
digerleri toplam 1.0'e sabit kalacak sekilde orantili azaltilir (normalize).

Not: compute_listing_score, listing icindeki alt skorlarin (budget, transport,
distance, safety, features, social) 0-1 araliginda ONCEDEN normalize edilmis
oldugunu varsayar. Ham veriden (price, transport_distance, ...) bu alt
skorlari uretmek data_prep/feature engineering asamasinin isidir.
"""

SCORE_KEYS = ("budget", "transport", "distance", "safety", "features", "social")

DEFAULT_WEIGHTS = {
    "budget": 0.30,
    "transport": 0.25,
    "distance": 0.20,
    "safety": 0.10,
    "features": 0.10,
    "social": 0.05,
}

PRIORITY_BOOST = 0.10


def adjust_weights(priorities: list[str], base_weights: dict[str, float] = DEFAULT_WEIGHTS) -> dict[str, float]:
    if not priorities:
        return dict(base_weights)

    weights = dict(base_weights)
    for priority in priorities:
        if priority in weights:
            weights[priority] += PRIORITY_BOOST

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def normalize_budget_fit(price: float, budget_max: float | None) -> float:
    """Price <= budget -> 1.0; the more it exceeds budget, the closer to 0.0."""
    if budget_max is None or budget_max <= 0:
        return 1.0
    if price <= budget_max:
        return 1.0
    over_ratio = (price - budget_max) / budget_max
    return max(0.0, 1.0 - over_ratio)


def compute_listing_score(listing: dict, weights: dict[str, float] = DEFAULT_WEIGHTS) -> float:
    return sum(weights.get(key, 0.0) * listing.get(key, 0.0) for key in SCORE_KEYS)
