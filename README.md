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

### Dashboard Utama (Dark Mode)
![Dashboard Light Mode](<img width="1902" height="908" alt="image" src="https://github.com/user-attachments/assets/3e362108-a913-48e8-9e6c-6d24674fa6a6" />
)
*Tampilan utama dashboard yang menunjukkan grafik real-time dan log training.*

### Konfigurasi Training
![Configuration Settings](<img width="1904" height="908" alt="image" src="https://github.com/user-attachments/assets/93798ad2-5302-4cae-b07d-095242d97cc0" />
)
*Antarmuka untuk mengatur parameter YAML sebelum memulai training.*

### 🎥 Demo Video
[![Watch the video](./assets/video_thumbnail.png)](./assets/demo_video.mp4)
*Klik gambar di atas untuk melihat video demonstrasi aplikasi.*

*(Untuk menambahkan gambar/video, buat folder `assets` di direktori proyek dan masukkan file Anda ke dalamnya).*

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
