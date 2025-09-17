# Dashboard Analisis Media Sosial

Dashboard interaktif untuk analisis media sosial dengan fitur lengkap termasuk analisis sentimen, engagement, social network analysis, dan trending topics.

## Fitur Dashboard

### 📈 Overview
- Metrics utama (total posts, views, likes, platforms)
- Distribusi platform
- Statistik umum

### 🎯 Engagement Analysis  
- Metrics engagement per platform (views, likes, shares, comments)
- Engagement rate calculation
- Perbandingan engagement antar platform

### 😊 Sentiment Analysis
- Distribusi sentimen keseluruhan
- Sentimen per platform
- Tren sentimen dari waktu ke waktu
- Perbandingan sentimen antar platform
- Quick emotion insight dan kombinasi emotion-sentiment

### 🎭 Emotion Analysis (NEW!)
- Overview metrics emotion (dominan, ter-engage)
- Distribusi emotion dan per platform
- Korelasi emotion vs sentiment (heatmap & charts)
- Tren emotion dari waktu ke waktu
- Engagement analysis per emotion
- Platform comparison untuk emotion
- Detail analysis per emotion dengan sample posts

### ☁️ Word Cloud
- Word cloud keseluruhan dengan filter sentiment & emotion
- Word cloud per platform
- Perbandingan WordCloud per sentiment
- Perbandingan WordCloud per emotion
- Multi-filter: platform + sentiment + emotion

### 🕸️ Social Network Analysis (SNA)
- Statistik jaringan (nodes, edges, density) 
- Visualisasi network dengan weighted nodes & edges
- Top influencers berdasarkan weighted in-degree
- Distribusi jenis relasi
- Interactive filters: platform, sentiment, emotion, issue, relation type
- Enhanced network visualization dengan sizing & coloring
- Issue-specific network analysis

### 🔥 Trending Topics
- Top issues/topics yang diperbincangkan
- Interactive topic analysis dengan drill-down
- Trending topics per platform
- Sentiment & emotion analysis per topic
- Temporal trend analysis
- Cross-platform comparison untuk topics
- Sample posts dengan emotion & sentiment info

## Cara Menjalankan

### 1. Aktivasi Virtual Environment
```bash
# Windows
.\VisualEnv\Scripts\Activate.ps1

# atau Command Prompt
VisualEnv\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Dashboard
```bash
streamlit run dashboard.py
```

### 4. Akses Dashboard
Dashboard akan terbuka di browser pada alamat: `http://localhost:8501`

## Struktur Data

### File dashboardsentimen_with_sentiment.csv
- `platform`: Platform media sosial
- `author`: Nama penulis
- `author_username`: Username penulis  
- `content_text`: Teks konten
- `post_url`: URL postingan
- `timestamp`: Waktu posting
- `likes`: Jumlah likes
- `shares`: Jumlah shares
- `comments`: Jumlah komentar
- `views`: Jumlah views
- `hashtags`: Hashtag
- `mentions`: Mention
- `scraped_at`: Waktu scraping
- `content_original`: Konten asli
- `emotion`: Emosi
- `sentiment`: Sentimen

### File dashboardsna_with_sentiment.csv
- `platform`: Platform media sosial
- `content_text`: Teks konten
- `source`: Sumber/pengirim
- `target`: Target/penerima
- `relation`: Jenis relasi
- `timestamp`: Waktu posting
- `scraped_at`: Waktu scraping
- `content_original`: Konten asli
- `sentiment`: Sentimen

## Fitur Filter

- **Filter Platform**: Pilih platform spesifik untuk analisis
- **Filter Tanggal**: Tentukan rentang waktu analisis
- **Perbandingan Platform**: Bandingkan metrics antar platform

## Requirements

- Python 3.8+
- Streamlit 1.28.0
- Pandas 2.1.0
- Plotly 5.17.0
- NetworkX 3.1
- WordCloud 1.9.2
- Dan dependencies lainnya (lihat requirements.txt)

## Tips Penggunaan

1. Gunakan sidebar untuk mengatur filter platform dan tanggal
2. Setiap tab memiliki analisis yang berbeda
3. Grafik interaktif - hover untuk detail informasi
4. Download hasil visualisasi dengan klik kanan pada grafik
5. Untuk dataset besar, loading mungkin memakan waktu beberapa saat

## Troubleshooting

### Error Loading Data
- Pastikan file CSV ada di folder yang sama dengan dashboard.py
- Periksa format timestamp dalam file CSV
- Pastikan kolom yang diperlukan tersedia

### Performance Issues
- Untuk dataset besar, pertimbangkan untuk menggunakan sampling
- Filter data berdasarkan tanggal untuk mengurangi beban

### Memory Issues
- Tutup aplikasi lain yang tidak perlu
- Gunakan filter untuk mengurangi data yang diproses 