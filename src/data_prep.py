"""Generate a realistic synthetic housing dataset for Bursa neighborhoods.

No real scraped listings exist for this project; instead we sample individual
"listings" around researcher-chosen, realistic base statistics per neighborhood
(rent level, distance to university/center, green space, safety, transport).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
LISTINGS_PER_NEIGHBORHOOD = 150

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Base profile per neighborhood: (district, avg rent/m2 TRY, dist_to_university_km,
# dist_to_center_km, dist_to_hospital_km, green_space, safety, transport, noise) -- scores 0-10
NEIGHBORHOOD_PROFILES = {
    "Görükle":  ("Nilüfer",   170,  1.0, 14.0, 6.0, 6.0, 6.5, 6.0, 6.5),
    "Özlüce":   ("Nilüfer",   190,  6.0,  9.0, 4.0, 6.5, 7.0, 6.5, 5.0),
    "Beşevler": ("Nilüfer",   230,  4.0,  6.0, 3.0, 6.0, 7.5, 8.5, 6.0),
    "Nilüfer":  ("Nilüfer",   260,  7.5,  7.0, 3.5, 7.5, 8.0, 7.5, 4.5),
    "Çekirge":  ("Osmangazi", 280,  9.0,  3.0, 1.0, 7.0, 8.5, 7.0, 5.0),
    "Heykel":   ("Osmangazi", 250, 10.0,  0.5, 2.0, 4.5, 6.5, 9.0, 8.0),
    "Yıldırım": ("Yıldırım",  175, 12.0,  6.0, 4.5, 5.0, 6.0, 6.0, 6.0),
    "Mudanya":  ("Mudanya",   210, 22.0, 20.0, 8.0, 8.5, 7.5, 4.0, 3.0),
}

ROOM_TYPES = ["1+1", "2+1", "3+1", "4+1"]
ROOM_TYPE_SIZE = {"1+1": (45, 65), "2+1": (70, 95), "3+1": (100, 130), "4+1": (135, 170)}
ROOM_TYPE_WEIGHTS = [0.25, 0.40, 0.25, 0.10]


def _jitter(rng, base, spread, size, low=0.0, high=10.0):
    return np.clip(rng.normal(base, spread, size), low, high)


def generate_listings(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    listing_id = 1
    for neighborhood, profile in NEIGHBORHOOD_PROFILES.items():
        district, rent_per_m2, dist_uni, dist_center, dist_hosp, green, safety, transport, noise = profile
        n = LISTINGS_PER_NEIGHBORHOOD
        room_types = rng.choice(ROOM_TYPES, size=n, p=ROOM_TYPE_WEIGHTS)

        for room_type in room_types:
            low, high = ROOM_TYPE_SIZE[room_type]
            size_m2 = rng.uniform(low, high)
            building_age = max(0, int(rng.exponential(8)))
            total_floors = rng.integers(3, 12)
            floor = rng.integers(0, total_floors + 1)

            age_discount = 1 - min(building_age, 30) / 100
            price_noise = rng.normal(1.0, 0.12)
            rent_try = round(size_m2 * rent_per_m2 * age_discount * price_noise, -1)

            rows.append({
                "listing_id": listing_id,
                "neighborhood": neighborhood,
                "district": district,
                "room_type": room_type,
                "size_m2": round(size_m2, 1),
                "building_age": building_age,
                "floor": int(floor),
                "total_floors": int(total_floors),
                "rent_try": max(rent_try, 3000),
                "distance_to_university_km": round(float(_jitter(rng, dist_uni, 1.5, 1, 0, 40)[0]), 1),
                "distance_to_center_km": round(float(_jitter(rng, dist_center, 1.2, 1, 0, 40)[0]), 1),
                "distance_to_hospital_km": round(float(_jitter(rng, dist_hosp, 0.8, 1, 0, 20)[0]), 1),
                "green_space_score": round(float(_jitter(rng, green, 0.8, 1)[0]), 1),
                "safety_score": round(float(_jitter(rng, safety, 0.6, 1)[0]), 1),
                "public_transport_score": round(float(_jitter(rng, transport, 0.7, 1)[0]), 1),
                "noise_level": round(float(_jitter(rng, noise, 0.9, 1)[0]), 1),
            })
            listing_id += 1

    return pd.DataFrame(rows)


def build_neighborhood_profile_table() -> pd.DataFrame:
    records = []
    for neighborhood, profile in NEIGHBORHOOD_PROFILES.items():
        district, rent_per_m2, dist_uni, dist_center, dist_hosp, green, safety, transport, noise = profile
        records.append({
            "neighborhood": neighborhood,
            "district": district,
            "avg_rent_per_m2_try": rent_per_m2,
            "avg_distance_to_university_km": dist_uni,
            "avg_distance_to_center_km": dist_center,
            "avg_distance_to_hospital_km": dist_hosp,
            "green_space_score": green,
            "safety_score": safety,
            "public_transport_score": transport,
            "noise_level": noise,
        })
    return pd.DataFrame(records)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    listings = generate_listings(rng)
    listings.to_csv(RAW_DIR / "bursa_listings_synthetic.csv", index=False)

    profiles = build_neighborhood_profile_table()
    profiles.to_csv(PROCESSED_DIR / "neighborhood_profiles.csv", index=False)

    listing_summary = (
        listings.groupby("neighborhood")
        .agg(avg_rent_try=("rent_try", "mean"), listing_count=("listing_id", "count"))
        .reset_index()
    )
    profiles_with_actuals = profiles.merge(listing_summary, on="neighborhood")
    profiles_with_actuals.to_csv(PROCESSED_DIR / "neighborhood_profiles_with_listing_stats.csv", index=False)

    print(f"Wrote {len(listings)} listings -> {RAW_DIR / 'bursa_listings_synthetic.csv'}")
    print(f"Wrote {len(profiles)} neighborhood profiles -> {PROCESSED_DIR / 'neighborhood_profiles.csv'}")


if __name__ == "__main__":
    main()
