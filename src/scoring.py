"""Agirlikli ev skoru hesaplama (Omer + Yagmur).

Ev Skoru = %30 Butce + %25 Ulasim + %20 Okula/ise uzaklik + %10 Guvenlik
           + %10 Ev ozellikleri + %5 Sosyal olanaklar

Kullanici bir onceligi vurgularsa (orn. "ulasim") ilgili agirlik artirilir,
digerleri toplam 100'e sabit kalacak sekilde orantili azaltilir.
"""

DEFAULT_WEIGHTS = {
    "budget": 0.30,
    "transport": 0.25,
    "distance": 0.20,
    "safety": 0.10,
    "features": 0.10,
    "social": 0.05,
}


def adjust_weights(priorities: list[str], base_weights: dict[str, float] = DEFAULT_WEIGHTS) -> dict[str, float]:
    raise NotImplementedError


def compute_listing_score(listing: dict, weights: dict[str, float]) -> float:
    raise NotImplementedError
