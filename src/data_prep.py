"""Sentetik veri uretimi + feature engineering (Yagmur).

Gercek scraping yerine, arastirmaci tarafindan secilen gercekci taban
istatistiklere (semt basina kira, mesafe, guvenlik vb.) gore sentetik
ilanlar uretilir. `main()` iki kademede calisir:

1. generate_raw_listings()  -> data/raw/bursa_listings_synthetic.csv
   Ham, insan-okunur olceklerde (TL, km, 0-10 skor) ilan verisi.
2. engineer_features()      -> data/processed/listings.csv
   build_district_stats()   -> data/processed/district_stats.csv
   src/scoring.py + src/recommender.py'nin bekledigi semayi (district,
   price, area, room_count, budget/transport/distance/safety/features/social
   0-1 normalize alt skorlar) uretir.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
LISTINGS_PER_DISTRICT = 150

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# district -> (avg rent/m2 TRY, dist_to_university_km, dist_to_center_km,
# dist_to_hospital_km, green_space, safety, transport, noise) -- son 4 skor 0-10
DISTRICT_PROFILES = {
    "Görükle":  (170,  1.0, 14.0, 6.0, 6.0, 6.5, 6.0, 6.5),
    "Özlüce":   (190,  6.0,  9.0, 4.0, 6.5, 7.0, 6.5, 5.0),
    "Beşevler": (230,  4.0,  6.0, 3.0, 6.0, 7.5, 8.5, 6.0),
    "Nilüfer":  (260,  7.5,  7.0, 3.5, 7.5, 8.0, 7.5, 4.5),
    "Çekirge":  (280,  9.0,  3.0, 1.0, 7.0, 8.5, 7.0, 5.0),
    "Heykel":   (250, 10.0,  0.5, 2.0, 4.5, 6.5, 9.0, 8.0),
    "Yıldırım": (175, 12.0,  6.0, 4.5, 5.0, 6.0, 6.0, 6.0),
    "Mudanya":  (210, 22.0, 20.0, 8.0, 8.5, 7.5, 4.0, 3.0),
}

ROOM_TYPES = ["1+1", "2+1", "3+1", "4+1"]
ROOM_TYPE_SIZE = {"1+1": (45, 65), "2+1": (70, 95), "3+1": (100, 130), "4+1": (135, 170)}
ROOM_TYPE_WEIGHTS = [0.25, 0.40, 0.25, 0.10]


def _jitter(rng, base, spread, size=1, low=0.0, high=10.0):
    return np.clip(rng.normal(base, spread, size), low, high)


def generate_raw_listings(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    listing_id = 1
    for district, profile in DISTRICT_PROFILES.items():
        rent_per_m2, dist_uni, dist_center, dist_hosp, green, safety, transport, noise = profile
        room_types = rng.choice(ROOM_TYPES, size=LISTINGS_PER_DISTRICT, p=ROOM_TYPE_WEIGHTS)

        for room_type in room_types:
            low, high = ROOM_TYPE_SIZE[room_type]
            area = rng.uniform(low, high)
            building_age = max(0, int(rng.exponential(8)))
            total_floors = int(rng.integers(3, 12))
            floor = int(rng.integers(0, total_floors + 1))

            age_discount = 1 - min(building_age, 30) / 100
            price_noise = rng.normal(1.0, 0.12)
            price = round(area * rent_per_m2 * age_discount * price_noise, -1)

            rows.append({
                "listing_id": listing_id,
                "district": district,
                "room_count": room_type,
                "area": round(area, 1),
                "building_age": building_age,
                "floor": floor,
                "total_floors": total_floors,
                "price": max(price, 3000),
                "distance_to_university_km": round(float(_jitter(rng, dist_uni, 1.5, low=0, high=40)[0]), 1),
                "distance_to_center_km": round(float(_jitter(rng, dist_center, 1.2, low=0, high=40)[0]), 1),
                "distance_to_hospital_km": round(float(_jitter(rng, dist_hosp, 0.8, low=0, high=20)[0]), 1),
                "green_space_score": round(float(_jitter(rng, green, 0.8)[0]), 1),
                "safety_score": round(float(_jitter(rng, safety, 0.6)[0]), 1),
                "public_transport_score": round(float(_jitter(rng, transport, 0.7)[0]), 1),
                "noise_level": round(float(_jitter(rng, noise, 0.9)[0]), 1),
            })
            listing_id += 1

    return pd.DataFrame(rows)


def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    normed = (series - lo) / (hi - lo)
    return normed if higher_is_better else 1 - normed


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Ham olcekleri src/scoring.py'nin bekledigi 0-1 alt skorlara cevirir.

    budget    <- price (dusuk = iyi)
    transport <- public_transport_score
    distance  <- distance_to_university_km (dusuk = iyi, "okula yakinlik" proxy'si)
    safety    <- safety_score
    features  <- yeni bina + cok katli site (asansor ihtimali) proxy'si
    social    <- yesil alan + merkeze yakinlik karisimi
    """
    engineered = raw[["district", "price", "area", "room_count"]].copy()
    engineered["budget"] = _normalize(raw["price"], higher_is_better=False)
    engineered["transport"] = raw["public_transport_score"] / 10
    engineered["distance"] = _normalize(raw["distance_to_university_km"], higher_is_better=False)
    engineered["safety"] = raw["safety_score"] / 10
    engineered["features"] = (
        0.7 * _normalize(raw["building_age"], higher_is_better=False)
        + 0.3 * _normalize(raw["total_floors"], higher_is_better=True)
    )
    engineered["social"] = (
        0.5 * (raw["green_space_score"] / 10)
        + 0.5 * _normalize(raw["distance_to_center_km"], higher_is_better=False)
    )
    return engineered


def build_district_stats(listings: pd.DataFrame) -> pd.DataFrame:
    stats = (
        listings.groupby("district")
        .agg(
            avg_price=("price", "mean"),
            budget=("budget", "mean"),
            transport=("transport", "mean"),
            distance=("distance", "mean"),
            safety=("safety", "mean"),
            features=("features", "mean"),
            social=("social", "mean"),
        )
        .reset_index()
    )
    return stats


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = generate_raw_listings(rng)
    raw.to_csv(RAW_DIR / "bursa_listings_synthetic.csv", index=False)

    listings = engineer_features(raw)
    listings.to_csv(PROCESSED_DIR / "listings.csv", index=False)

    district_stats = build_district_stats(listings)
    district_stats.to_csv(PROCESSED_DIR / "district_stats.csv", index=False)

    print(f"Wrote {len(raw)} raw listings -> {RAW_DIR / 'bursa_listings_synthetic.csv'}")
    print(f"Wrote {len(listings)} engineered listings -> {PROCESSED_DIR / 'listings.csv'}")
    print(f"Wrote {len(district_stats)} district stats -> {PROCESSED_DIR / 'district_stats.csv'}")


if __name__ == "__main__":
    main()
