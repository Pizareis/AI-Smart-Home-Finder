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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(139, 92, 246, 0.35), transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(6, 182, 212, 0.28), transparent 45%),
        radial-gradient(circle at 50% 105%, rgba(236, 72, 153, 0.20), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(34, 211, 238, 0.10), transparent 50%),
        #08090f;
    background-attachment: fixed;
}

.block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 780px; }

[data-testid="stMarkdownContainer"] p { color: #d4d4e0; }

/* ---------- Hero ---------- */
.hero-wrap { text-align: center; margin-bottom: 2rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: linear-gradient(90deg, rgba(139,92,246,0.18), rgba(6,182,212,0.18));
    border: 1px solid rgba(196, 181, 253, 0.35);
    color: #d8c9ff;
    padding: 7px 16px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    margin-bottom: 18px;
}
.hero-title {
    font-size: 3rem; font-weight: 900; line-height: 1.1; margin: 0 0 12px 0; letter-spacing: -0.02em;
    background: linear-gradient(100deg, #ffffff 15%, #d8c9ff 45%, #67e8f9 85%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    color: #9d9db4; font-size: 1.04rem; line-height: 1.6; margin: 0 auto 22px auto; max-width: 540px;
}
.hero-stats { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.hero-stat {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 8px 16px;
    font-size: 0.82rem; color: #c3c3d6; font-weight: 600;
    backdrop-filter: blur(10px);
}
.hero-stat b { color: #ffffff; font-weight: 800; }

/* ---------- Glass form panel ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02)) !important;
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(255, 255, 255, 0.13) !important;
    border-radius: 22px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.08);
    padding: 8px 4px;
    margin-bottom: 18px;
}

/* Text area */
[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.045) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 14px !important;
    color: #f2f2f7 !important;
    font-size: 0.97rem;
    padding: 14px !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.25) !important;
}
[data-testid="stTextArea"] label { color: #cfcfe0 !important; font-weight: 600; }

/* Primary button */
div.stButton > button {
    background: linear-gradient(90deg, #8b5cf6, #6366f1 50%, #06b6d4);
    background-size: 160% 100%;
    color: white; border: none; border-radius: 999px;
    padding: 0.7rem 1.6rem; font-weight: 700; font-size: 1rem; letter-spacing: 0.01em;
    box-shadow: 0 8px 28px rgba(139, 92, 246, 0.4);
    transition: transform 0.15s ease, box-shadow 0.15s ease, background-position 0.4s ease;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 34px rgba(139, 92, 246, 0.55);
    background-position: 100% 0;
    color: white;
}
div.stButton > button:active { transform: translateY(0); }

/* ---------- Tabs ---------- */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 6px; }
[data-baseweb="tab"] {
    color: #8f8fa8 !important; font-weight: 700; border-radius: 12px 12px 0 0 !important;
    padding: 10px 16px !important;
}
[data-baseweb="tab"] p { font-size: 0.93rem; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #ffffff !important;
    background: linear-gradient(180deg, rgba(139,92,246,0.22), rgba(139,92,246,0.06)) !important;
}
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important; height: 3px !important; border-radius: 3px; }

/* ---------- Listing cards (fully custom HTML) ---------- */
.result-caption {
    color: #7d7d95; font-size: 0.82rem; font-weight: 600; margin: 4px 0 14px 2px;
}
.listing-card {
    position: relative;
    background: linear-gradient(160deg, rgba(255,255,255,0.065), rgba(255,255,255,0.015));
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 18px;
    padding: 18px 20px 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.32);
    overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.listing-card:hover {
    border-color: rgba(255,255,255,0.22);
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}
.listing-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.listing-district { font-size: 1.05rem; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 6px; }
.listing-district .pin { opacity: 0.75; font-size: 0.9rem; }
.match-badge {
    flex-shrink: 0; font-size: 0.76rem; font-weight: 800; padding: 5px 11px; border-radius: 999px;
    white-space: nowrap;
}
.listing-price { font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-bottom: 12px; }
.listing-price span { font-size: 0.85rem; font-weight: 600; color: #8f8fa8; margin-left: 3px; }
.listing-pills { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.pill {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    color: #c3c3d6; font-size: 0.8rem; font-weight: 600;
    padding: 5px 11px; border-radius: 10px;
}
.score-track { height: 6px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; margin-bottom: 12px; }
.score-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #8b5cf6, #06b6d4); }
.listing-comment { color: #a8a8bd; font-size: 0.88rem; line-height: 1.5; }

/* ---------- Alerts ---------- */
[data-testid="stInfo"], [data-testid="stAlert"] {
    background: rgba(139, 92, 246, 0.1) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
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


def format_try(amount: float) -> str:
    return f"{amount:,.0f}".replace(",", ".")


def score_tier(score: float) -> dict:
    if score >= 0.8:
        return {"color": "#34d399", "bg": "rgba(52,211,153,0.14)", "border": "rgba(52,211,153,0.4)", "glow": "rgba(52,211,153,0.25)", "label": "Mükemmel eşleşme"}
    if score >= 0.6:
        return {"color": "#c4b5fd", "bg": "rgba(139,92,246,0.16)", "border": "rgba(139,92,246,0.4)", "glow": "rgba(139,92,246,0.25)", "label": "İyi eşleşme"}
    return {"color": "#67e8f9", "bg": "rgba(103,232,249,0.12)", "border": "rgba(103,232,249,0.35)", "glow": "rgba(103,232,249,0.2)", "label": "Uygun seçenek"}


MAX_CARDS_PER_GROUP = 10


def render_listing_group(listings: pd.DataFrame) -> None:
    if len(listings) > MAX_CARDS_PER_GROUP:
        st.markdown(
            f'<div class="result-caption">En uygun {MAX_CARDS_PER_GROUP} ilan gösteriliyor '
            f"· toplam {len(listings)} sonuç bulundu</div>",
            unsafe_allow_html=True,
        )
    for _, row in listings.head(MAX_CARDS_PER_GROUP).iterrows():
        render_listing_card(row)


def render_listing_card(row: pd.Series) -> None:
    score = min(max(row["score"], 0.0), 1.0)
    tier = score_tier(score)
    area = f"{row['area']:.0f} m²" if "area" in row and pd.notna(row["area"]) else "—"
    room = row.get("room_count", "—")

    card_style = f"border-left: 4px solid {tier['color']};"
    badge_style = f"background:{tier['bg']}; color:{tier['color']}; border:1px solid {tier['border']};"
    html = f"""
    <div class="listing-card" style="{card_style}">
        <div class="listing-head">
            <div class="listing-district"><span class="pin">📍</span>{row['district']}</div>
            <div class="match-badge" style="{badge_style}">{round(score * 100)}% · {tier['label']}</div>
        </div>
        <div class="listing-price">{format_try(row['price'])} TL<span>/ay</span></div>
        <div class="listing-pills">
            <span class="pill">🛏️ {room}</span>
            <span class="pill">📐 {area}</span>
        </div>
        <div class="score-track"><div class="score-fill" style="width:{score * 100:.0f}%;"></div></div>
        <div class="listing-comment">{build_listing_comment(row)}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">✨ AI-Powered · Bursa</div>
        <div class="hero-title">Smart Home Finder</div>
        <div class="hero-subtitle">
            Kendinizi anlatın — yaş, öğrenci/çalışan, okul/iş konumu, bütçe, oda tercihi, öncelikler.
            Yapay zeka destekli öneri sistemimiz size en uygun semtleri bulsun.
        </div>
        <div class="hero-stats">
            <div class="hero-stat"><b>8</b> semt</div>
            <div class="hero-stat"><b>1.200+</b> ilan</div>
            <div class="hero-stat"><b>3</b> gruplu öneri</div>
        </div>
    </div>
    """,
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
