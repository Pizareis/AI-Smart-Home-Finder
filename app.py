"""Streamlit arayuzu: profil girisi -> 3 gruplu semt onerisi (premium glassmorphism UI)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.nlp_parser import KNOWN_DISTRICTS, parse_user_input
from src.recommender import build_listing_comment, recommend

st.set_page_config(page_title="AI-Powered Smart Home Finder", page_icon="\U0001F3E0", layout="centered")

LISTINGS_PATH = Path("data/processed/listings.csv")
DISTRICT_STATS_PATH = Path("data/processed/district_stats.csv")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(139, 92, 246, 0.28), transparent 42%),
        radial-gradient(circle at 88% 15%, rgba(6, 182, 212, 0.22), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(236, 72, 153, 0.14), transparent 50%),
        #0b0d17;
}

.block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 760px; }

/* Hero */
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
    padding: 6px 14px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin: 0 0 10px 0;
    background: linear-gradient(90deg, #ffffff 20%, #c4b5fd 60%, #67e8f9 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: #a1a1b5; font-size: 1.02rem; line-height: 1.55; margin-bottom: 1.6rem; }

/* Glass containers (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.055) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    padding: 6px 2px;
    margin-bottom: 16px;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(139, 92, 246, 0.15);
}

/* Text area */
[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px !important;
    color: #f2f2f7 !important;
    font-size: 0.96rem;
    padding: 14px !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.22) !important;
}
[data-testid="stTextArea"] label { color: #cfcfe0 !important; font-weight: 600; }

/* Primary button */
div.stButton > button {
    background: linear-gradient(90deg, #8b5cf6, #6366f1 55%, #06b6d4);
    color: white; border: none; border-radius: 999px;
    padding: 0.65rem 1.6rem; font-weight: 700; letter-spacing: 0.01em;
    box-shadow: 0 6px 24px rgba(139, 92, 246, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.5);
    color: white;
}
div.stButton > button:active { transform: translateY(0); }

/* Tabs */
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(255,255,255,0.1); }
[data-baseweb="tab"] {
    color: #a1a1b5 !important; font-weight: 600; border-radius: 10px 10px 0 0 !important;
}
[data-baseweb="tab"] p { font-size: 0.94rem; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #ffffff !important;
    background: rgba(139, 92, 246, 0.14) !important;
}
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important; height: 3px !important; }

/* Metrics */
[data-testid="stMetricLabel"] { color: #8f8fa8 !important; font-size: 0.78rem !important; letter-spacing: 0.03em; }
[data-testid="stMetricValue"] { color: #f2f2f7 !important; font-size: 1.15rem !important; }

/* Card title row */
.card-title { font-size: 1.1rem; font-weight: 700; color: #f2f2f7; margin: 4px 0 2px 4px; }
.card-price { color: #67e8f9; font-weight: 700; }
.card-comment { color: #b8b8cc; font-size: 0.9rem; padding: 0 4px 6px 4px; }

/* Alerts */
[data-testid="stInfo"], [data-testid="stAlert"] {
    background: rgba(139, 92, 246, 0.1) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px);
}

/* Progress bar (score) */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important;
}

footer, #MainMenu { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not LISTINGS_PATH.exists() or not DISTRICT_STATS_PATH.exists():
        return None, None
    return pd.read_csv(LISTINGS_PATH), pd.read_csv(DISTRICT_STATS_PATH)


MAX_CARDS_PER_GROUP = 10


def render_listing_group(listings: pd.DataFrame) -> None:
    if len(listings) > MAX_CARDS_PER_GROUP:
        st.caption(f"En uygun {MAX_CARDS_PER_GROUP} ilan gösteriliyor · toplam {len(listings)} sonuç bulundu")
    for _, row in listings.head(MAX_CARDS_PER_GROUP).iterrows():
        render_listing_card(row)


def render_listing_card(row: pd.Series) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title">{row["district"]} '
            f'<span class="card-price">· {row["price"]:,.0f} TL</span></div>'.replace(",", "."),
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        cols[0].metric("Alan", f"{row.get('area', '—'):.0f} m²" if "area" in row else "—")
        cols[1].metric("Oda", row.get("room_count", "—"))
        cols[2].metric("Skor", f"{row['score']:.2f}")
        st.progress(min(max(row["score"], 0.0), 1.0))
        st.markdown(f'<div class="card-comment">{build_listing_comment(row)}</div>', unsafe_allow_html=True)


st.markdown('<div class="hero-badge">✨ AI-Powered · Bursa</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Smart Home Finder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Kendinizi anlatın — yaş, öğrenci/çalışan, okul/iş konumu, bütçe, '
    'oda tercihi, öncelikler. Yapay zeka destekli öneri sistemimiz size en uygun semtleri bulsun.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    user_text = st.text_area(
        "Kendinizi tanıtın",
        placeholder="24 yaşındayım, Bursa'da öğrenciyim. Üniversitem Nilüfer'de. "
        "Aylık maksimum 20.000 TL verebilirim. 1+1 veya 2+1 istiyorum. "
        "Ulaşım ve güvenlik benim için önemli.",
        height=120,
        label_visibility="collapsed",
    )
    submitted = st.button("✨ Öneri Al", type="primary")

if submitted:
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
                    st.write("Metninizde bilinen bir semt tespit edemedim veya kriterlerinize uyan ilan yok.")
                else:
                    render_listing_group(result["target"])

            with tab_alt:
                if result["alternatives"].empty:
                    st.write("Belirttiğiniz alternatif semtte kriterlerinize uyan ilan bulunamadı.")
                else:
                    render_listing_group(result["alternatives"])

            with tab_suggested:
                if result["suggested"].empty:
                    st.write("Şu an için ek bir semt önerisi yok.")
                else:
                    st.info(result["suggested_reason"])
                    render_listing_group(result["suggested"])
