# Analisis Ketersediaan Data Vendor

Sistem analisis ketersediaan data vendor berbasis web menggunakan Streamlit.

## Fitur Utama

- 📊 **Dashboard Analisis** - Visualisasi ringkasan dan detail ketersediaan data
- 🛠️ **Preprocessing Pipeline** - Pembersihan data otomatis dengan berbagai metode
- ⚙️ **Feature Engineering** - Pembuatan fitur untuk machine learning
- 📑 **PDF Reporting** - Export laporan profesional dalam format PDF
- 🔄 **Sorting & Filtering** - Pengurutan data berdasarkan abjad atau jumlah data

## Instalasi

### Requirements
- Python 3.8+
- pip

### Setup

1. Clone repository ini
```bash
git clone https://github.com/USERNAME/REPO-NAME.git
cd REPO-NAME
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi
```bash
streamlit run app.py
```

4. Buka browser di `http://localhost:8501`

## Cara Penggunaan

### 1. Upload Data
- Navigasi ke menu **Beranda**
- Upload file CSV SAP Anda
- Pilih rentang tahun dan Major Item

### 2. Analisis Dashboard
- Lihat ringkasan statistik
- Review status ketersediaan data
- Download laporan CSV atau PDF

### 3. Preprocessing
- Pilih Sub-Material dan Vendor
- Terapkan cleaning methods
- Tambahkan data eksogen
- Download dataset final

### 4. Feature Engineering
- Buat lag features
- Hitung rolling statistics
- Generate calendar features
- Export untuk modeling

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push ke GitHub
2. Kunjungi [share.streamlit.io](https://share.streamlit.io)
3. Login dan pilih repository
4. Deploy!

Lihat [deployment_guide.md](deployment_guide.md) untuk opsi deployment lainnya.

## Struktur File

```
.
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .gitignore            # Git ignore rules
├── Procfile              # Heroku deployment
└── setup.sh              # Heroku setup script
```

## Dependencies

- streamlit
- pandas
- plotly
- numpy
- openpyxl
- reportlab
- pillow

## Kontribusi

Silakan buat issue atau pull request untuk perbaikan dan fitur baru.

## Lisensi

MIT License

## Kontak

Untuk pertanyaan atau dukungan, silakan hubungi [email@domain.com]
