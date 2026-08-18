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
streamlit run app.py
```

## Testler

```bash
pytest
```

## Veri Seti

*(Doldurulacak: kaynak / sentetik uretim yontemi aciklamasi)*

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
