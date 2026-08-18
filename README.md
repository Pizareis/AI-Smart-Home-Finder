# AI-Powered Smart Home Finder

Kullanicinin serbest metinle anlattigi tercihleri (yas, ogrenci/calisan, okul/is konumu,
butce, oda tercihi, oncelikler) yapilandirilmis veriye cevirip, bir ev veri seti uzerinde
agirlikli bir puanlama/oneri algoritmasi calistiran ve sonuclari uc grupta sunan bir
AI/Data Science recommendation sistemi prototipi (Bursa, tek sehir MVP).

Bu gercek bir emlak platformu degildir; portfoy/CV amacli bir prototiptir.

## Ozellikler

- Serbest metin -> yapilandirilmis kullanici profili (regex/keyword tabanli, stretch: LLM API)
- Agirlikli skorlama: butce, ulasim, okula/ise uzaklik, guvenlik, ev ozellikleri, sosyal olanaklar
- Uc gruplu semt sonuclari: istenen semt / alternatif semt(ler) / sistemin onerdigi semt(ler)
- Streamlit arayuzu

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Kullanim

```bash
# 1) Sentetik veri setini + islenmis dosyalari uret
python src/data_prep.py

# 2) Uygulamayi baslat
streamlit run app.py
```

## Testler

```bash
pytest
```

## Veri Seti

Gercek emlak scraping'i yapilmiyor: `src/data_prep.py`, Bursa'nin 8 semti icin
(Görükle, Özlüce, Beşevler, Nilüfer, Çekirge, Heykel, Yıldırım, Mudanya) arastirmaci
tarafindan secilen gercekci taban istatistiklere (m² basi kira, üniversiteye/merkeze/
hastaneye uzaklik, yesil alan, güvenlik, ulasim, gürültü) gore semt basina 150 sentetik
ilan uretir.

Iki asamali pipeline:

1. **Ham veri** (`data/raw/bursa_listings_synthetic.csv`) — insan-okunur olceklerde
   (TL, km, 0-10 skor) ilan verisi. `notebooks/01_eda.ipynb` bu veri uzerinde calisir.
2. **Feature engineering** (`data/processed/listings.csv`, `data/processed/district_stats.csv`) —
   `src/scoring.py`'nin bekledigi semaya (district, price, area, room_count,
   budget/transport/distance/safety/features/social — hepsi 0-1 normalize) cevrilmis hali.
   `notebooks/02_feature_engineering.ipynb` bu donusumu adim adim gosterir.

## Proje Yapisi

```
AI-Smart-Home-Finder/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering.ipynb
├── src/
│   ├── data_prep.py
│   ├── scoring.py
│   ├── recommender.py
│   └── nlp_parser.py
├── app.py
├── requirements.txt
└── README.md
```

## Ekip

- Yagmur — Data Science (veri seti, EDA, feature engineering)
- Omer — Software/AI (algoritma, NLP parsing, Streamlit arayuzu, proje yapisi)
