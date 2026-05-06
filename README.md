<div align="center">

<br>

<img src="https://img.shields.io/badge/-%F0%9F%A7%A0%20PaddleOCR%20Training%20Visualizer-0d1117?style=for-the-badge" alt="title" />

<br><br>

**Dashboard web modern untuk memantau dan mengontrol proses pelatihan model OCR secara real-time**

*Tanpa perlu buka terminal lagi.*

<br>

[![License](https://img.shields.io/badge/license-MIT-2ea043?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-Supported-0097a7?style=flat-square)](https://github.com/PaddlePaddle/PaddleOCR)
[![WebSocket](https://img.shields.io/badge/WebSocket-Real--Time-f59e0b?style=flat-square)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
[![Dark Mode](https://img.shields.io/badge/Dark_Mode-Supported-6366f1?style=flat-square)](.)
[![YAML](https://img.shields.io/badge/YAML-Config-888?style=flat-square)](.)

<br>

[**Lihat Demo →**](#-demo-video) · [**Cara Pakai →**](#-cara-menjalankan) · [**Kontribusi →**](#-kontribusi)

<br>

</div>

---

## ✨ Fitur Utama

<table>
<tr>
<td width="50%">

**📊 Real-Time Metrics**
Pantau Loss, HMean, Precision, dan Recall secara langsung saat training berjalan — diperbarui setiap step.

</td>
<td width="50%">

**🎛️ Training Controller**
Start dan Stop training langsung dari UI tanpa menyentuh terminal sama sekali.

</td>
</tr>
<tr>
<td width="50%">

**⚙️ Dynamic YAML Editor**
Edit epoch, learning rate, batch size, dan dataset path langsung dari browser.

</td>
<td width="50%">

**🧠 Neural Network UI**
Visualisasi arsitektur jaringan yang beranimasi dan merespons proses training.

</td>
</tr>
<tr>
<td width="50%">

**⚡ WebSocket API**
Arsitektur WebSocket untuk update data real-time dengan latensi sangat rendah.

</td>
<td width="50%">

**🌓 Light / Dark Mode**
Antarmuka modern yang nyaman dipakai siang maupun malam hari.

</td>
</tr>
</table>

---

## 📸 Tampilan Antarmuka

### 🖥️ Dashboard Utama

> Tampilan utama — grafik loss/accuracy real-time dan log training yang mengalir langsung.

![Dashboard Utama](https://github.com/user-attachments/assets/3e362108-a913-48e8-9e6c-6d24674fa6a6)
---

### ⚙️ Konfigurasi Training

> Editor parameter YAML interaktif — ubah hyperparameter dan mulai training tanpa restart server.

![Konfigurasi](https://github.com/user-attachments/assets/93798ad2-5302-4cae-b07d-095242d97cc0)

---

## 🎥 Demo Video

> Klik thumbnail di bawah untuk menonton demo lengkap aplikasi.

[![Tonton Demo](https://github.com/user-attachments/assets/875bb5fe-9ed4-4cdf-afa3-24090ded1ab6)

---

## 🛠️ Prasyarat

Pastikan hal-hal berikut sudah tersedia sebelum menjalankan aplikasi:

| Prasyarat | Keterangan |
|---|---|
| **Python 3.8+** | Versi Python yang kompatibel |
| **websockets** | `pip install websockets` |
| **pyyaml** | `pip install pyyaml` |
| **PaddleOCR** | Folder `PaddleOCR/` harus ada di dalam direktori proyek |

```bash
pip install websockets pyyaml
```

---

## 🚀 Cara Menjalankan

**1. Masuk ke direktori proyek**

```bash
cd [DIR]/ocr
```

**2. Jalankan server backend**

```bash
python server.py
```

> Secara default, server akan membaca log dari `./output/det_db_finetune/train.log`.
> Kamu juga bisa menentukan path log kustom sebagai argumen:
> ```bash
> python server.py ./path/ke/custom.log
> ```

**3. Buka dashboard di browser**

```
http://localhost:8080
```

---

## 📂 Struktur Direktori

```
📁 ocr/
├── 📄 server.py              # Backend server (WebSocket + HTTP API)
├── 📄 index.html             # Frontend dashboard (UI)
├── 📄 ocr_finetune_det.yml   # File konfigurasi PaddleOCR
├── 📁 PaddleOCR/             # Direktori instalasi PaddleOCR
├── 📁 output/                # Hasil training dan log
└── 📁 dataset/               # Data untuk training
```

---

## 🤝 Kontribusi

Kritik, saran, dan kontribusi sangat disambut! Ada beberapa cara untuk ikut berkontribusi:

- 🐛 **Laporkan bug** — Buka [Issues](../../issues) dan jelaskan masalah yang kamu temukan
- 💡 **Usulkan fitur** — Diskusikan ide baru di [Discussions](../../discussions) atau buka Feature Request
- 🔧 **Kirim Pull Request** — Fork repo ini, buat branch baru, lalu kirim PR

```bash
git clone https://github.com/username/repo.git
git checkout -b feat/nama-fitur
git commit -m "feat: tambahkan fitur X"
git push origin feat/nama-fitur
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **[MIT License](LICENSE)** — bebas digunakan, dimodifikasi, dan didistribusikan.

---

<div align="center">

Dibuat dengan ☕ untuk komunitas ML Indonesia

</div>
