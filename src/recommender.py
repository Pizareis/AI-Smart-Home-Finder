"""Siralama ve 3 gruplu semt oneri mantigi (Omer).

Grup 1: kullanicinin istedigi semt
Grup 2: kullanicinin belirttigi alternatif semt(ler)
Grup 3: sistemin kendi onerdigi semt(ler), gerekceli
"""

import pandas as pd

from src.nlp_parser import UserProfile


def rank_listings(listings: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    raise NotImplementedError


def recommend(listings: pd.DataFrame, district_stats: pd.DataFrame, profile: UserProfile) -> dict:
    """Returns {"target": df, "alternatives": df, "suggested": df, "suggested_reason": str}."""
    raise NotImplementedError
