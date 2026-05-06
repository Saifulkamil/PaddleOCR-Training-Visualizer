import os
import json
import glob
import random
import shutil

def main():
    # Folder output yang berisi JSON & Image sesuai struktur/permintaan Anda
    # Kita cek beberapa kemungkinan path jika ada salah pengetikan nama folder
    possible_dirs = [
        # r"output_test_2\detection_display\visual_and_json",
        r"data_test/image_detection/visual_and_json"
        
    ]
    
    source_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            source_dir = d
            break
            
    if not source_dir:
        print(f"Error: Folder sumber JSON tidak ditemukan! Pastikan Anda sudah menjalankan script test dan folder terbentuk.")
        return

    # Folder tujuan untuk format Dataset Kaggle (PaddleOCR format)
    dataset_dir = "dataset_finetune"
    img_dir = os.path.join(dataset_dir, "images")
    
    os.makedirs(img_dir, exist_ok=True)
    
    label_lines = []
    
    # Ambil semua file json dari folder tersebut
    json_files = glob.glob(os.path.join(source_dir, "*.json"))
    print(f"Ditemukan {len(json_files)} file JSON. Mulai memproses...")
    
    for json_path in json_files:
        filename = os.path.basename(json_path)
        
        # Mencari nama file gambar dari nama JSON (misal hasil PaddleX nama jsonnya image.jpg.json atau image_res.json)
        # Kita potong extension .json
        img_name = filename[:-5] if filename.endswith('.json') else filename
        
        # Terkadang Paddle menambahkan '_res' atau '.res' pada penamaan jsonnya
        if '_res' in img_name:
            img_name_asli = img_name.replace('_res', '')
        elif img_name.endswith('.jpg'):
             # jika jadinya penerimaan_produk_xxx.jpg maka ambil originalnya saja
             img_name_asli = img_name
        else:
            # Jika namanya bukan .jpg, asumsikan default .jpg
            img_name_asli = img_name + ".jpg"

        # Folder asli untuk gambar berada di dalam folder 'asli' satu tingkat di atas visual_and_json
        parent_dir = os.path.dirname(os.path.normpath(source_dir))
        original_img_dir = os.path.join(parent_dir, "asli")        
        img_source_path = os.path.join(original_img_dir, img_name_asli)
        
        # Fallback jika ternyata png
        if not os.path.exists(img_source_path):
            img_source_path = os.path.join(original_img_dir, img_name_asli.replace('.jpg', '.png'))
            img_name_asli = img_name_asli.replace('.jpg', '.png')
            
        if not os.path.exists(img_source_path):
            # Mencoba asumsi nama JSON itu diubah total oleh PaddleX
            # Biasanya json mendimpan "input_path" di dalamnya, kita bisa baca dari jsonnya saja langsung
            pass
            
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                continue
                
        # Jika file gambar nggak ketemu dengan tebakan nama, kita ambil dari meta json nya (input_path) kalau ada
        if not os.path.exists(img_source_path) and isinstance(data, dict) and "input_path" in data:
            asli_path = data["input_path"]
            img_name_asli = os.path.basename(asli_path)
            img_source_path = os.path.join(original_img_dir, img_name_asli)
            # Kadang _res.jpg
            if not os.path.exists(img_source_path):
                img_name_res = os.path.splitext(img_name_asli)[0] + "_res" + os.path.splitext(img_name_asli)[1]
                img_source_path = os.path.join(original_img_dir, img_name_res)
                
        if not os.path.exists(img_source_path):
             print(f"⚠️ Melewati {filename}: Gambar pasangannya tidak ditemukan di folder yang sama.")
             continue
             
        # Ambil koordinat titik deteksi (polygon)
        if isinstance(data, dict):
            polys = data.get("dt_polys", [])
        elif isinstance(data, list):
            polys = data
        else:
            polys = []
            
        # Konversi ke List of Dictionaries format PaddleOCR PPOCRLabel 
        # contoh: [{"transcription": "text", "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}, ...]
        annotations = []
        for poly in polys:
            annt = {
                "transcription": "text",  # PENTING: JANGAN gunakan "###". Di PaddleOCR, "###" dianggap sebagai teks terabaikan (ignore) dan tidak dihitung/dilatih!
                "points": poly 
            }
            annotations.append(annt)
            
        # Mengembalikan rute gambar ke dalam folder "images/"
        relative_img_path = f"images/{img_name_asli}"
        line = f"{relative_img_path}\t{json.dumps(annotations, ensure_ascii=False)}"
        
        # Copy gambar ke folder images dataset kita
        dst_img_path = os.path.join(img_dir, img_name_asli)
        try:
            shutil.copy(img_source_path, dst_img_path)
        except OSError as e:
            if e.errno == 28:
                print(f"\n❌ GAGAL BANYAK: Disk Anda PENUH (No space left on device)! Harap hapus beberapa file di Harddisk dan jalankan script ini lagi nanti.")
                return
            else:
                raise e
        
        label_lines.append(line)

    if not label_lines:
        print("\n❌ Gagal membuat dataset. Tidak ada JSON dan Gambar valid yang berhasil dipasangkan.")
        return

    # Tulis semua data ke train.txt
    train_txt_path = os.path.join(dataset_dir, "train.txt")
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(label_lines))
        
    print("\n✅ Pembuatan Dataset Format Kaggle / PaddleOCR Selesai!")
    print(f"Total Data Tersedia : {len(label_lines)} gambar")
    print(f"Data Training       : {len(label_lines)} gambar (Train.txt)")
    print(f"Directory Dataset   : {os.path.abspath(dataset_dir)}")

if __name__ == '__main__':
    main()
