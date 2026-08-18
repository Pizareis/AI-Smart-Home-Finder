"""Veri yukleme ve temizleme (Yagmur)."""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    raise NotImplementedError


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
