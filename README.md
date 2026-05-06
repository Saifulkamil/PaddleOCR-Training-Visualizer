

# 🚀 PaddleOCR Real-Time Training Visualizer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-Supported-green)

Aplikasi ini adalah **Real-Time Neural Network Visualizer & Training Controller** berbasis web untuk memantau dan mengontrol proses pelatihan model [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). 

Dengan antarmuka yang modern dan responsif, Anda tidak perlu lagi melihat log di terminal yang membosankan. Dashboard ini menyediakan grafik real-time, animasi neural network, serta kontrol penuh terhadap konfigurasi dan proses training Anda.

---

## ✨ Fitur Utama

- 📊 **Real-Time Metrics Visualization**: Pantau Loss, Akurasi (HMean), Precision, dan Recall secara langsung saat training berjalan.
- 🎛️ **Training Controller**: Mulai (Start) dan Hentikan (Stop) proses training langsung dari antarmuka Web.
- ⚙️ **Dynamic Configuration Editor**: Ubah parameter training (seperti epoch, learning rate, batch size, dataset path) langsung dari UI tanpa harus mengedit file YAML secara manual.
- 🌓 **Light/Dark Mode Theme**: Antarmuka modern yang mendukung mode gelap dan terang sesuai preferensi Anda.
- 🧠 **Animated Neural Network UI**: Visualisasi interaktif arsitektur jaringan yang merespon terhadap proses training.
- ⚡ **WebSocket & HTTP API**: Menggunakan arsitektur WebSocket untuk update data real-time dengan latensi rendah.

---

## 📸 Tampilan Antarmuka (Screenshots & Video)

*(Catatan: Ganti placeholder gambar dan video di bawah ini dengan tangkapan layar asli dari aplikasi Anda)*
<div align="center">

# 🖥️ ML Training Dashboard

**Real-time monitoring & configuration for machine learning training pipelines**

![Python](https://img.shields.io/badge/Python-3.10+-1D9E75?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-378ADD?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-534AB7?style=flat-square)
![Dark Mode](https://img.shields.io/badge/dark_mode-supported-0f1117?style=flat-square)

</div>

---

## 🖥️ Dashboard Utama

![Dashboard Main](./docs/dashboard.png)

> Tampilan utama — grafik loss/accuracy real-time dan log training yang mengalir langsung.

---

## ⚙️ Konfigurasi Training

![Configuration](./docs/config.png)

> Editor parameter YAML interaktif — ubah hyperparameter dan mulai training tanpa restart server.

---

## 🎥 Demo Video

[![Watch Demo](./docs/thumbnail.png)](https://link-ke-video-kamu)

> Klik thumbnail di atas untuk melihat demo lengkap aplikasi.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 📈 Real-time Monitoring | Grafik loss, accuracy, dan metrik diperbarui setiap step |
| 📝 YAML Configuration | Edit parameter training langsung dari UI |
| 🌙 Dark Mode | Tampilan gelap yang nyaman untuk sesi panjang |

---

## 🚀 Quick Start

```bash
git clone https://github.com/user/ml-dashboard
pip install -r requirements.txt
python main.py
```
---

## 🛠️ Prasyarat (Requirements)

Sebelum menjalankan aplikasi, pastikan Anda telah menginstal dependensi berikut:

- **Python 3.8+**
- Modul Python yang dibutuhkan:
  ```bash
  pip install websockets pyyaml
  ```
- **PaddleOCR**: Pastikan repositori PaddleOCR sudah ada di dalam direktori proyek ini (folder `PaddleOCR/`).

---

## 🚀 Cara Menjalankan Aplikasi

1. **Clone atau buka direktori proyek ini:**
   ```bash
   cd d:/kerja/AI/ocr
   ```

2. **Jalankan Server:**
   Anda dapat menjalankan server dengan mengeksekusi file `server.py`. Jika Anda memiliki path log file khusus, Anda bisa menyertakannya sebagai argumen (default: `./output/det_db_finetune/train.log`).
   
   ```bash
   python server.py
   ```

3. **Buka Dashboard di Browser:**
   Setelah server berjalan, buka browser web Anda dan navigasikan ke alamat berikut:
   
   ```text
   http://localhost:8080
   ```

---

## 📂 Struktur Direktori Utama

```text
📁 ocr/
├── 📄 server.py            # Backend server (WebSocket + HTTP API)
├── 📄 index.html           # Frontend dashboard (UI)
├── 📄 ocr_finetune_det.yml # File konfigurasi PaddleOCR
├── 📁 PaddleOCR/           # Direktori instalasi PaddleOCR
├── 📁 output/              # Hasil training dan logs
└── 📁 dataset/             # Kumpulan data untuk training
```

---

## 🤝 Kontribusi

Kritik, saran, dan kontribusi sangat dipersilakan! Jangan ragu untuk membuat *Pull Request* atau melaporkan masalah (*Issues*) jika Anda menemukan bug atau memiliki ide fitur baru.

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
