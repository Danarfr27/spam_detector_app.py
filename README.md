# 🛡️ Spam Detector Pro

Aplikasi deteksi spam berbasis Machine Learning yang dibuat dengan Streamlit dan Scikit-Learn.

## 🎥 Demo Recording Guide (Bandicam)

Berikut langkah-langkah untuk merekam demo aplikasi:

### 1. Setup & Buka Aplikasi
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run spam_detector_app.py
```

### 2. Mulai Recording dengan Bandicam
- Buka Bandicam
- Pilih mode "Screen Recording"
- Atur area recording ke browser Streamlit
- Klik tombol REC (atau tombol F12)

### 3. Langkah Demo yang Direkam

**Scene 1: Buka Aplikasi (0:00 - 0:30)**
- Tampilkan browser membuka `http://localhost:8501`
- Tunjukkan header "Spam Detector Pro"
- Jelaskan fitur-fitur yang tersedia

**Scene 2: GitHub Repository (0:30 - 1:00)**
- Buka tab baru
- Kunjungi GitHub repository (https://github.com/[username]/spam-detector-pro)
- Tunjukkan struktur file dan kode
- Jelaskan teknologi yang digunakan

**Scene 3: Single Detection (1:00 - 2:00)**
- Tab "Single Detection"
- Masukkan contoh spam: "Congratulations! You've won a $1000 gift card!"
- Klik tombol untuk analisis
- Tunjukkan hasil "SPAM DETECTED" dengan confidence score
- Masukkan contoh normal: "Hey, are we still meeting for lunch?"
- Tunjukkan hasil "SAFE MESSAGE"

**Scene 4: Batch Upload (2:00 - 2:30)**
- Tab "Batch Upload"
- Upload file CSV sample
- Klik "Analyze All"
- Tunjukkan hasil dan download

**Scene 5: Analytics (2:30 - 3:00)**
- Tab "Analytics"
- Jelaskan model performance
- Tunjukkan confusion matrix dan metrics

### 4. Stop Recording
- Tekan F12 untuk stop recording di Bandicam
- Video akan tersimpan di folder output Bandicam

## 🚀 Deploy ke Streamlit Cloud

### Langkah 1: Buat Repository GitHub
1. Login ke GitHub
2. Buat repository baru: `spam-detector-pro`
3. Upload file:
   - `spam_detector_app.py`
   - `requirements.txt`
   - `README.md`

### Langkah 2: Deploy ke Streamlit Community Cloud
1. Kunjungi [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Login dengan akun GitHub
3. Klik "New app"
4. Pilih repository `spam-detector-pro`
5. Set main file path: `spam_detector_app.py`
6. Klik "Deploy!"

### Langkah 3: Akses Aplikasi
- Aplikasi akan live di: `https://[username]-spam-detector-pro.streamlit.app`

## 📁 Struktur File
```
spam-detector-pro/
├── spam_detector_app.py    # Main application file
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

## 🛠️ Teknologi
- **Streamlit**: Framework UI
- **Scikit-Learn**: Machine Learning (Naive Bayes)
- **TF-IDF**: Text vectorization
- **Pandas**: Data manipulation

## 📊 Model
- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF (1000 features)
- **Accuracy**: ~97%
- **Dataset**: 50 samples (25 spam, 25 ham)

## 🔗 Links
- [Streamlit Community Cloud](https://streamlit.io/cloud)
- [GitHub Repository](https://github.com)

## 📹 Recording Tips
- Gunakan Bandicam dengan setting HD (1080p)
- Aktifkan mouse click effects
- Gunakan microphone untuk narasi
- Edit dengan CapCut atau software editing lainnya
