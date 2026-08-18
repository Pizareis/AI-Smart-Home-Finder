import streamlit as st

from scoring import load_listings, recommend_listings, recommend_neighborhoods

PRIORITY_LABELS = {
    "affordability": "Uygun fiyat",
    "university_proximity": "Üniversiteye yakınlık",
    "center_proximity": "Şehir merkezine yakınlık",
    "hospital_proximity": "Hastaneye yakınlık",
    "green_space": "Yeşil alan",
    "safety": "Güvenlik",
    "public_transport": "Toplu taşıma",
    "quietness": "Sessizlik",
}

st.set_page_config(page_title="Bursa Ev Bulucu", page_icon="🏠", layout="wide")
st.title("🏠 AI Destekli Bursa Ev Bulucu")
st.caption("Bütçeni, oda tercihini ve önceliklerini gir; sana en uygun mahalle ve ilanları önerelim.")


@st.cache_data
def _cached_listings():
    return load_listings()


listings = _cached_listings()

with st.sidebar:
    st.header("Tercihlerin")
    budget = st.number_input(
        "Aylık bütçe (TL)",
        min_value=3000,
        max_value=int(listings["rent_try"].max()),
        value=20000,
        step=500,
    )
    room_type = st.selectbox("Oda tipi", ["Farketmez"] + sorted(listings["room_type"].unique()))

    st.subheader("Öncelikler")
    st.caption("0 = önemsiz, 5 = çok önemli")
    weights = {
        key: st.slider(label, 0, 5, 2)
        for key, label in PRIORITY_LABELS.items()
    }

room_type_filter = None if room_type == "Farketmez" else room_type
active_weights = {k: v for k, v in weights.items() if v > 0}

tab_neighborhoods, tab_listings = st.tabs(["📍 Mahalle Önerileri", "🏘️ Uygun İlanlar"])

with tab_neighborhoods:
    if not active_weights:
        st.info("En az bir önceliğe ağırlık ver (kaydırıcılardan).")
    else:
        top_neighborhoods = recommend_neighborhoods(active_weights, top_n=8)
        col_chart, col_table = st.columns([2, 3])
        with col_chart:
            st.bar_chart(top_neighborhoods.set_index("neighborhood")["match_score"])
        with col_table:
            st.dataframe(top_neighborhoods, hide_index=True, width="stretch")

with tab_listings:
    if not active_weights:
        st.info("En az bir önceliğe ağırlık ver (kaydırıcılardan).")
    else:
        top_listings = recommend_listings(
            active_weights, budget_try=budget, room_type=room_type_filter, top_n=15
        )
        if top_listings.empty:
            st.warning("Bu bütçe ve oda tipiyle eşleşen ilan bulunamadı. Bütçeyi artırmayı dene.")
        else:
            st.dataframe(top_listings, hide_index=True, width="stretch")
