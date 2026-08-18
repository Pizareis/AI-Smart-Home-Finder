"""Streamlit arayuzu: profil girisi -> 3 gruplu semt onerisi (premium glassmorphism UI)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.nlp_parser import KNOWN_DISTRICTS, parse_user_input
from src.recommender import build_listing_comment, recommend

st.set_page_config(page_title="AI-Powered Smart Home Finder", page_icon="\U0001F3E0", layout="wide")

LISTINGS_PATH = Path("data/processed/listings.csv")
DISTRICT_STATS_PATH = Path("data/processed/district_stats.csv")
REPO_URL = "https://github.com/Pizareis/AI-Smart-Home-Finder"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulseGlow { 0%, 100% { opacity: 0.55; } 50% { opacity: 0.9; } }

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(139, 92, 246, 0.38), transparent 38%),
        radial-gradient(circle at 92% 8%, rgba(6, 182, 212, 0.30), transparent 42%),
        radial-gradient(circle at 50% 100%, rgba(236, 72, 153, 0.22), transparent 55%),
        radial-gradient(circle at 100% 60%, rgba(34, 211, 238, 0.14), transparent 45%),
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        #08090f;
    background-size: auto, auto, auto, auto, 44px 44px, 44px 44px, auto;
    background-attachment: fixed;
}

[data-testid="stToolbar"] { visibility: hidden; }
footer, #MainMenu { visibility: hidden; }

.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1120px; margin: 0 auto; }
[data-testid="stMarkdownContainer"] p { color: #d4d4e0; }

/* ---------- Top nav ---------- */
.topnav {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 4px 28px 4px; margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.topnav-brand { font-weight: 800; font-size: 1.05rem; color: #f2f2f7; display: flex; align-items: center; gap: 8px; }
.topnav-link {
    font-size: 0.85rem; font-weight: 600; color: #a1a1b5; text-decoration: none;
    border: 1px solid rgba(255,255,255,0.14); padding: 6px 14px; border-radius: 999px;
    transition: color 0.15s ease, border-color 0.15s ease;
}
.topnav-link:hover { color: #ffffff; border-color: rgba(139,92,246,0.5); }

/* ---------- Hero ---------- */
.hero-wrap { text-align: center; margin: 1.5rem 0 2.5rem 0; animation: fadeInUp 0.7s ease both; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: linear-gradient(90deg, rgba(139,92,246,0.18), rgba(6,182,212,0.18));
    border: 1px solid rgba(196, 181, 253, 0.35);
    color: #d8c9ff;
    padding: 7px 16px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 3.4rem; font-weight: 900; line-height: 1.08; margin: 0 0 14px 0; letter-spacing: -0.02em;
    background: linear-gradient(100deg, #ffffff 15%, #d8c9ff 45%, #67e8f9 85%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    color: #9d9db4; font-size: 1.08rem; line-height: 1.65; margin: 0 auto 26px auto; max-width: 600px;
}
.hero-stats { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.hero-stat {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 9px 18px;
    font-size: 0.84rem; color: #c3c3d6; font-weight: 600;
    backdrop-filter: blur(10px);
}
.hero-stat b { color: #ffffff; font-weight: 800; }

/* ---------- Feature strip ---------- */
.feature-row { animation: fadeInUp 0.8s ease both; margin-bottom: 2rem; }
.feature-card {
    background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px; padding: 20px 18px; height: 100%;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.feature-card:hover { transform: translateY(-3px); border-color: rgba(139,92,246,0.4); }
.feature-icon {
    width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(6,182,212,0.25));
    font-size: 1.2rem; margin-bottom: 12px;
}
.feature-title { font-weight: 700; color: #f2f2f7; font-size: 0.98rem; margin-bottom: 6px; }
.feature-desc { color: #9898ac; font-size: 0.85rem; line-height: 1.5; }

/* ---------- Glass form panel ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02)) !important;
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(255, 255, 255, 0.13) !important;
    border-radius: 22px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.08);
    padding: 10px 6px;
    margin-bottom: 18px;
}

/* Text area */
[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.045) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 14px !important;
    color: #f2f2f7 !important;
    font-size: 0.98rem;
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
    padding: 0.75rem 1.6rem; font-weight: 700; font-size: 1.02rem; letter-spacing: 0.01em;
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
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 10px; }
[data-baseweb="tab"] {
    color: #8f8fa8 !important; font-weight: 700; border-radius: 12px 12px 0 0 !important;
    padding: 10px 18px !important;
}
[data-baseweb="tab"] p { font-size: 0.94rem; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #ffffff !important;
    background: linear-gradient(180deg, rgba(139,92,246,0.22), rgba(139,92,246,0.06)) !important;
}
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important; height: 3px !important; border-radius: 3px; }

/* ---------- Listing cards (fully custom HTML) ---------- */
.result-caption {
    color: #7d7d95; font-size: 0.84rem; font-weight: 600; margin: 4px 0 16px 2px;
}
.listing-card {
    position: relative;
    background: linear-gradient(160deg, rgba(255,255,255,0.065), rgba(255,255,255,0.015));
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 18px;
    padding: 18px 20px 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.32);
    overflow: hidden;
    animation: fadeInUp 0.5s ease both;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.listing-card:hover {
    border-color: rgba(255,255,255,0.24);
    transform: translateY(-4px);
    box-shadow: 0 18px 44px rgba(0,0,0,0.42);
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

/* ---------- Footer ---------- */
.site-footer {
    margin-top: 3.5rem; padding-top: 24px; border-top: 1px solid rgba(255,255,255,0.08);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;
}
.footer-note { color: #6f6f85; font-size: 0.82rem; line-height: 1.5; max-width: 560px; }
.footer-stack { display: flex; gap: 8px; flex-wrap: wrap; }
.stack-badge {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    color: #8f8fa8; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 8px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="topnav">
        <div class="topnav-brand">🏠 Smart Home Finder</div>
        <a class="topnav-link" href="{REPO_URL}" target="_blank">GitHub ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not LISTINGS_PATH.exists() or not DISTRICT_STATS_PATH.exists():
        return None, None
    return pd.read_csv(LISTINGS_PATH), pd.read_csv(DISTRICT_STATS_PATH)


def format_try(amount: float) -> str:
    return f"{amount:,.0f}".replace(",", ".")


def score_tier(score: float) -> dict:
    if score >= 0.8:
        return {"color": "#34d399", "bg": "rgba(52,211,153,0.14)", "border": "rgba(52,211,153,0.4)", "label": "Mükemmel eşleşme"}
    if score >= 0.6:
        return {"color": "#c4b5fd", "bg": "rgba(139,92,246,0.16)", "border": "rgba(139,92,246,0.4)", "label": "İyi eşleşme"}
    return {"color": "#67e8f9", "bg": "rgba(103,232,249,0.12)", "border": "rgba(103,232,249,0.35)", "label": "Uygun seçenek"}


MAX_CARDS_PER_GROUP = 10
CARD_COLUMNS = 2


def render_listing_group(listings: pd.DataFrame) -> None:
    if len(listings) > MAX_CARDS_PER_GROUP:
        st.markdown(
            f'<div class="result-caption">En uygun {MAX_CARDS_PER_GROUP} ilan gösteriliyor '
            f"· toplam {len(listings)} sonuç bulundu</div>",
            unsafe_allow_html=True,
        )
    rows = listings.head(MAX_CARDS_PER_GROUP).to_dict("records")
    cols = st.columns(CARD_COLUMNS, gap="medium")
    for i, row in enumerate(rows):
        with cols[i % CARD_COLUMNS]:
            render_listing_card(row)


def render_listing_card(row: dict) -> None:
    score = min(max(row["score"], 0.0), 1.0)
    tier = score_tier(score)
    area = f"{row['area']:.0f} m²" if row.get("area") is not None and pd.notna(row.get("area")) else "—"
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

st.markdown('<div class="feature-row">', unsafe_allow_html=True)
feat_cols = st.columns(3, gap="medium")
features = [
    ("📝", "Kendinizi Tanıtın", "Yaşınızı, bütçenizi, oda tercihinizi ve önceliklerinizi serbest metinle anlatın."),
    ("🧠", "AI Analiz Etsin", "Doğal dil işleme metninizi yapılandırılmış verilere çevirir, ağırlıklı puanlama devreye girer."),
    ("🎯", "3 Gruplu Öneri Alın", "İstediğiniz semt, alternatifleriniz ve sistemin gerekçeli önerileri ayrı sekmelerde sunulur."),
]
for col, (icon, title, desc) in zip(feat_cols, features):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

form_col, _ = st.columns([2, 1])
with form_col:
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

st.markdown(
    f"""
    <div class="site-footer">
        <div class="footer-note">
            Bu bir portföy/CV projesidir, gerçek bir emlak platformu değildir. İlan verileri
            araştırılan semt istatistiklerinden türetilmiş <b>sentetik</b> verilerdir.
        </div>
        <div class="footer-stack">
            <span class="stack-badge">Python</span>
            <span class="stack-badge">pandas</span>
            <span class="stack-badge">scikit-learn</span>
            <span class="stack-badge">Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
