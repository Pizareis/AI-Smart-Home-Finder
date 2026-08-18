"""Streamlit arayuzu: profil girisi -> 3 gruplu semt onerisi."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.nlp_parser import KNOWN_DISTRICTS, parse_user_input
from src.recommender import build_listing_comment, recommend

st.set_page_config(page_title="AI-Powered Smart Home Finder", page_icon="\U0001F3E0")

LISTINGS_PATH = Path("data/processed/listings.csv")
DISTRICT_STATS_PATH = Path("data/processed/district_stats.csv")


@st.cache_data
def load_data():
    if not LISTINGS_PATH.exists() or not DISTRICT_STATS_PATH.exists():
        return None, None
    return pd.read_csv(LISTINGS_PATH), pd.read_csv(DISTRICT_STATS_PATH)


def render_listing_card(row: pd.Series) -> None:
    with st.container(border=True):
        st.markdown(f"**{row['district']}** — {row['price']:.0f} TL")
        cols = st.columns(3)
        cols[0].metric("Alan", f"{row.get('area', '—')} m²")
        cols[1].metric("Oda", row.get("room_count", "—"))
        cols[2].metric("Skor", f"{row['score']:.2f}")
        st.caption(build_listing_comment(row))


st.title("AI-Powered Smart Home Finder")
st.write("Kendinizi anlatin: yas, ogrenci/calisan, okul/is konumu, butce, oda tercihi, oncelikler.")

user_text = st.text_area(
    "Kendinizi tanitin",
    placeholder="24 yaşındayım, Bursa'da öğrenciyim. Üniversitem Nilüfer'de. "
    "Aylık maksimum 20.000 TL verebilirim. 1+1 veya 2+1 istiyorum. "
    "Ulaşım ve güvenlik benim için önemli.",
    height=120,
)

if st.button("Öneri al", type="primary"):
    if not user_text.strip():
        st.warning("Lütfen önce kendinizi tanıtan bir metin girin.")
    else:
        listings, district_stats = load_data()
        if listings is None:
            st.info(
                "Veri seti henüz hazır değil (data/processed/listings.csv ve "
                "district_stats.csv bekleniyor) — dataset hazırlığı tamamlanınca "
                "sonuçlar burada görünecek."
            )
        else:
            profile = parse_user_input(user_text, KNOWN_DISTRICTS)
            result = recommend(listings, district_stats, profile)

            tab_target, tab_alt, tab_suggested = st.tabs(
                ["🎯 İstediğiniz Semt", "🔁 Alternatif Semtler", "✨ Sistem Önerisi"]
            )

            with tab_target:
                if result["target"].empty:
                    st.write("Metninizde bilinen bir semt tespit edemedim.")
                else:
                    for _, row in result["target"].iterrows():
                        render_listing_card(row)

            with tab_alt:
                if result["alternatives"].empty:
                    st.write("Belirttiğiniz alternatif semt bulunamadı.")
                else:
                    for _, row in result["alternatives"].iterrows():
                        render_listing_card(row)

            with tab_suggested:
                if result["suggested"].empty:
                    st.write("Şu an için ek bir semt önerisi yok.")
                else:
                    st.info(result["suggested_reason"])
                    for _, row in result["suggested"].iterrows():
                        render_listing_card(row)
