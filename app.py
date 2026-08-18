"""Streamlit arayuzu: profil girisi -> 3 gruplu semt onerisi."""

import streamlit as st

st.set_page_config(page_title="AI-Powered Smart Home Finder", page_icon="\U0001F3E0")

st.title("AI-Powered Smart Home Finder")
st.write("Kendinizi anlatin: yas, ogrenci/calisan, okul/is konumu, butce, oda tercihi, oncelikler.")

user_text = st.text_area("Kendinizi tanitin", placeholder="24 yasindayim, Bursa'da ogrenciyim...")

if st.button("Oneri al"):
    st.info("Henuz baglanti kurulmadi - nlp_parser / recommender entegrasyonu bekleniyor.")
