# AI-Powered Smart Home Finder

Kullanıcının bütçesini, oda tercihini ve önceliklerini (üniversiteye yakınlık, güvenlik, yeşil alan, ulaşım, sessizlik vb.) analiz edip Bursa'daki farklı mahalleleri/ilanları puanlayan bir öneri sistemi prototipi. Gerçek emlak scraping yapılmaz; gerçekçi taban istatistiklere dayalı sentetik bir veri seti kullanılır.

## Kapsam

- Şehir: Bursa
- Mahalleler: Görükle, Özlüce, Beşevler, Nilüfer, Çekirge, Heykel, Yıldırım, Mudanya

## Ekip

- **Yağmur** — Data Science: veri seti, temizlik, EDA, feature engineering, puanlama
- **Ömer** — Software/AI: proje yapısı, öneri algoritması, kullanıcı girişi, Streamlit arayüzü

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Kullanım

```bash
# 1) Sentetik veri setini üret
python src/data_prep.py

# 2) EDA'yı incele
jupyter notebook notebooks/01_eda.ipynb

# 3) Testleri çalıştır
pytest tests/

# 4) Uygulamayı başlat
streamlit run src/app.py
```

## Proje yapısı

```
data/
  raw/          -> ham sentetik ilan verisi (bursa_listings_synthetic.csv)
  processed/    -> mahalle bazlı özet profiller
notebooks/
  01_eda.ipynb  -> keşifsel veri analizi
src/
  data_prep.py  -> sentetik veri üretimi
  scoring.py    -> öncelik ağırlıklı puanlama motoru
  app.py        -> Streamlit arayüzü
tests/          -> pytest testleri
```
