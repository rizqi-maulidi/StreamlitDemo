# KeyBERT Integration - Dashboard Upgrade

## 📋 Overview
Dashboard telah diupgrade untuk menggunakan **KeyBERT** sebagai pengganti word frequency dalam ekstraksi trending topics. Ini memberikan hasil analisis topik yang lebih semantik dan meaningful.

## 🚀 Apa yang Berubah?

### 1. **Topic Extraction Algorithm**
- **Sebelum**: Frequency-based word counting
- **Sesudah**: KeyBERT dengan semantic analysis menggunakan BERT embeddings

### 2. **Keunggulan KeyBERT**
- ✅ **Semantic Understanding**: Memahami makna konteks, bukan hanya frekuensi kata
- ✅ **Phrase Detection**: Dapat mendeteksi frasa bermakna (1-3 kata)
- ✅ **Language Support**: Mendukung bahasa Indonesia dan multilingual
- ✅ **Relevance Scoring**: Setiap topic diberi skor relevance yang akurat
- ✅ **Diversity**: Memastikan topik yang beragam, tidak repetitif

### 3. **Perubahan UI/UX**
- Kolom "Frequency" → "Relevance Score"
- Loading indicator saat ekstraksi topic
- Informasi detail tentang KeyBERT dalam expander
- Error handling yang lebih baik dengan fallback

## 🔧 Technical Implementation

### Dependencies Baru
```
keybert==0.9.0
sentence-transformers>=5.1.0
torch>=2.8.0
transformers>=4.56.0
```

### Model yang Digunakan
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Alasan**: Lightweight, multilingual, cocok untuk bahasa Indonesia

### Fungsi Utama
1. `load_keybert_model()`: Load dan cache model KeyBERT
2. `extract_topics()`: Main function menggunakan KeyBERT
3. `extract_topics_fallback()`: Fallback ke frequency-based jika KeyBERT gagal

## 🎯 Fitur yang Tetap Sama

Semua functionality dashboard yang sudah ada tetap berfungsi sama:
- ✅ Klik topic untuk analisis detail
- ✅ Trend sentimen per topic  
- ✅ Analisis emotion per topic
- ✅ Platform comparison
- ✅ Time series analysis
- ✅ Sample posts display
- ✅ Cross-analysis dengan SNA

## 📊 Lokasi Perubahan

### Tab "Trending Topics"
- Extracting topics menggunakan KeyBERT
- Tampilan relevance score
- Informational expandable section

### Tab "SNA" 
- Filter isu/topic menggunakan KeyBERT topics
- Lebih akurat dalam filtering network berdasarkan topik semantik

### Platform-specific Topics
- Trending topics per platform menggunakan KeyBERT

## 🔄 Fallback Mechanism

Jika KeyBERT gagal (koneksi internet, model error, etc.), sistem otomatis menggunakan frequency-based method sebagai fallback untuk memastikan dashboard tetap berfungsi.

## 💡 Tips Penggunaan

1. **First-time loading**: Model akan didownload otomatis saat pertama kali digunakan
2. **Internet connection**: Pastikan koneksi stabil untuk download model pertama kali  
3. **Performance**: Model di-cache sehingga loading setelah pertama kali akan lebih cepat
4. **Topic quality**: Hasil topik akan lebih meaningful dan contextual

## 🐛 Troubleshooting

Jika ada masalah dengan KeyBERT:
1. Dashboard akan otomatis fallback ke method lama
2. Check koneksi internet untuk download model
3. Restart dashboard jika diperlukan

---

**Upgrade completed**: KeyBERT integration untuk trending topics analysis yang lebih powerful! 🎉 