"""
Merge dataset OCR Detection dari berbagai sumber untuk fine-tuning PaddleOCR.
Menghasilkan train.txt, valid.txt, dan test.txt terpisah.

Struktur input (dataset_valid_for_train/):
  1/ -> PaddleOCR format (train_label.txt, valid_label.txt, train/, valid/)
  2/ -> COCO/Roboflow (train/, valid/, test/ + _annotations.coco.json)
  3/ -> COCO/Roboflow
  4/ -> COCO/Roboflow
  5/ -> COCO/Roboflow

Output (dataset_merged/):
  images/      -> Semua gambar (nama unik per dataset+split)
  train.txt    -> Label training
  valid.txt    -> Label validation
  test.txt     -> Label testing (jika ada)
"""

import os
import sys
import json
import shutil
import hashlib
from collections import Counter


# ============================================================
# KONFIGURASI
# ============================================================
SOURCE_DIR = r"D:\kerja\AI\ocr\dataset_valid_for_train"
OUTPUT_DIR = r"D:\kerja\AI\ocr\dataset_merged"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

DATASET_PREFIXES = {"1": "ds1", "2": "ds2", "3": "ds3", "4": "ds4", "5": "ds5"}


# ============================================================
# VALIDASI & KONVERSI
# ============================================================

def validate_image_exists(img_path):
    if not os.path.isfile(img_path):
        return False, f"File tidak ditemukan: {img_path}"
    size = os.path.getsize(img_path)
    if size < 100:
        return False, f"File terlalu kecil ({size} bytes): {img_path}"
    return True, ""


def validate_paddle_annotation(ann_str, filename=""):
    try:
        ann_list = json.loads(ann_str)
    except json.JSONDecodeError as e:
        return False, f"JSON error '{filename}': {e}"
    if not isinstance(ann_list, list) or len(ann_list) == 0:
        return False, f"Annotation invalid pada '{filename}'"
    for i, ann in enumerate(ann_list):
        if "transcription" not in ann or "points" not in ann:
            return False, f"Missing key di annotation {i} pada '{filename}'"
        pts = ann["points"]
        if not isinstance(pts, list) or len(pts) < 3:
            return False, f"Points invalid di annotation {i} pada '{filename}'"
        for j, pt in enumerate(pts):
            if not isinstance(pt, list) or len(pt) != 2:
                return False, f"Point {j} invalid di annotation {i} pada '{filename}'"
            if not all(isinstance(v, (int, float)) for v in pt):
                return False, f"Point {j} bukan angka di annotation {i} pada '{filename}'"
    return True, ""


def bbox_to_quad(bbox):
    x, y, w, h = bbox
    return [
        [round(x, 1), round(y, 1)],
        [round(x + w, 1), round(y, 1)],
        [round(x + w, 1), round(y + h, 1)],
        [round(x, 1), round(y + h, 1)],
    ]


def segmentation_to_quad(segmentation):
    if not segmentation or not isinstance(segmentation, list):
        return None
    poly = segmentation[0] if isinstance(segmentation[0], list) else segmentation
    if len(poly) < 6:
        return None
    points = []
    for i in range(0, len(poly), 2):
        if i + 1 < len(poly):
            points.append([poly[i], poly[i + 1]])
    if len(points) == 4:
        return [[round(p[0], 1), round(p[1], 1)] for p in points]
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [
            [round(min(xs), 1), round(min(ys), 1)],
            [round(max(xs), 1), round(min(ys), 1)],
            [round(max(xs), 1), round(max(ys), 1)],
            [round(min(xs), 1), round(max(ys), 1)],
        ]


# ============================================================
# PROCESSORS - Mengembalikan dict per split
# ============================================================

def process_paddleocr_dataset(dataset_dir, prefix):
    """
    Proses dataset PaddleOCR format.
    Returns: dict { "train": [...], "valid": [...] }
    Setiap entry: (src_img_path, new_filename, annotation_str)
    """
    split_map = {
        "train_label.txt": "train",
        "valid_label.txt": "valid",
    }
    result = {"train": [], "valid": [], "test": []}

    for label_file, split_name in split_map.items():
        label_path = os.path.join(dataset_dir, label_file)
        if not os.path.exists(label_path):
            print(f"  [WARN] Label file tidak ada: {label_path}")
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    print(f"  [WARN] Baris {line_num} di {label_file}: no tab")
                    continue

                img_rel_path, ann_str = parts[0], parts[1]
                valid, msg = validate_paddle_annotation(ann_str, f"{label_file}:{line_num}")
                if not valid:
                    print(f"  [WARN] {msg}")
                    continue

                src_img = os.path.join(dataset_dir, img_rel_path)
                valid_img, msg = validate_image_exists(src_img)
                if not valid_img:
                    print(f"  [WARN] {msg}")
                    continue

                original_name = os.path.basename(img_rel_path)
                new_filename = f"{prefix}_{split_name}_{original_name}"
                result[split_name].append((src_img, new_filename, ann_str))

    return result


def process_coco_dataset(dataset_dir, prefix):
    """
    Proses dataset COCO/Roboflow format.
    Returns: dict { "train": [...], "valid": [...], "test": [...] }
    """
    result = {"train": [], "valid": [], "test": []}

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue

        ann_file = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(ann_file):
            print(f"  [WARN] COCO annotation tidak ditemukan: {ann_file}")
            continue

        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        id_to_image = {img['id']: img for img in coco_data['images']}
        valid_cat_ids = set(cat['id'] for cat in coco_data['categories'])

        # Group annotations by image_id
        img_annotations = {}
        for ann in coco_data['annotations']:
            if ann['category_id'] not in valid_cat_ids:
                continue
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)

        for img_id, anns in img_annotations.items():
            if img_id not in id_to_image:
                continue
            img_info = id_to_image[img_id]
            img_filename = img_info['file_name']
            src_img = os.path.join(split_dir, img_filename)

            valid_img, msg = validate_image_exists(src_img)
            if not valid_img:
                print(f"  [WARN] {msg}")
                continue

            paddle_anns = []
            for ann in anns:
                bbox = ann.get('bbox')
                segmentation = ann.get('segmentation')
                quad = None

                if segmentation and isinstance(segmentation, list) and len(segmentation) > 0:
                    if isinstance(segmentation[0], list):
                        quad = segmentation_to_quad(segmentation)

                if quad is None and bbox:
                    quad = bbox_to_quad(bbox)
                if quad is None:
                    continue

                paddle_anns.append({"transcription": "text", "points": quad})

            if not paddle_anns:
                continue

            ann_str = json.dumps(paddle_anns, ensure_ascii=False)
            valid, msg = validate_paddle_annotation(ann_str, img_filename)
            if not valid:
                print(f"  [WARN] {msg}")
                continue

            new_filename = f"{prefix}_{split}_{img_filename}"
            result[split].append((src_img, new_filename, ann_str))

    return result


# ============================================================
# RESOLVE DUPLICATES
# ============================================================

def resolve_duplicates(entries_list):
    """Resolve duplicate filenames across all entries."""
    filename_counts = Counter(e[1] for e in entries_list)
    duplicates = {k for k, v in filename_counts.items() if v > 1}

    if not duplicates:
        return entries_list

    print(f"  [WARN] {len(duplicates)} duplikat ditemukan, menyelesaikan...")
    seen = {}
    resolved = []
    for src, fname, ann in entries_list:
        if fname in seen:
            name, ext = os.path.splitext(fname)
            file_hash = hashlib.md5(src.encode()).hexdigest()[:8]
            fname = f"{name}_{file_hash}{ext}"
        seen[fname] = True
        resolved.append((src, fname, ann))
    return resolved


# ============================================================
# COPY & WRITE
# ============================================================

def copy_and_write(entries, label_file_path, split_name):
    """Copy gambar dan tulis label file. Returns (success, failed)."""
    if not entries:
        print(f"  [{split_name.upper()}] Tidak ada data")
        return 0, 0

    success = 0
    failed = 0

    with open(label_file_path, 'w', encoding='utf-8') as f:
        for i, (src_img, new_filename, ann_str) in enumerate(entries):
            dest_img = os.path.join(OUTPUT_IMAGES_DIR, new_filename)
            try:
                shutil.copy2(src_img, dest_img)
            except OSError as e:
                if e.errno == 28:
                    print(f"\n[FATAL] Disk PENUH! Berhenti.")
                    sys.exit(1)
                print(f"  [ERR] Gagal copy {src_img}: {e}")
                failed += 1
                continue

            # Verifikasi ukuran
            if os.path.getsize(src_img) != os.path.getsize(dest_img):
                print(f"  [ERR] Ukuran tidak cocok: {new_filename}")
                os.remove(dest_img)
                failed += 1
                continue

            f.write(f"images/{new_filename}\t{ann_str}\n")
            success += 1

            if (i + 1) % 1000 == 0:
                print(f"    ... {i + 1}/{len(entries)}")

    return success, failed


# ============================================================
# VALIDASI FINAL PER SPLIT
# ============================================================

def validate_label_file(label_file_path, split_name):
    """Validasi konsistensi label file vs gambar di disk."""
    if not os.path.exists(label_file_path):
        print(f"  [{split_name}] File tidak ada, skip")
        return True

    with open(label_file_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print(f"  [{split_name}] Kosong")
        return True

    errors = 0
    for line_num, line in enumerate(lines, 1):
        parts = line.split('\t', 1)
        if len(parts) < 2:
            print(f"  [ERR] {split_name} baris {line_num}: format invalid")
            errors += 1
            continue

        img_path = os.path.join(OUTPUT_DIR, parts[0])
        if not os.path.exists(img_path):
            print(f"  [ERR] {split_name} baris {line_num}: gambar tidak ada: {parts[0]}")
            errors += 1
            continue

        valid, msg = validate_paddle_annotation(parts[1], f"{split_name}:{line_num}")
        if not valid:
            print(f"  [ERR] {split_name} baris {line_num}: {msg}")
            errors += 1

    if errors == 0:
        print(f"  [{split_name.upper()}] OK - {len(lines)} baris valid")
    else:
        print(f"  [{split_name.upper()}] GAGAL - {errors} error dari {len(lines)} baris")

    return errors == 0


# ============================================================
# MAIN
# ============================================================

def merge_all_datasets():
    print("=" * 60)
    print("MERGE DATASET OCR DETECTION (train/valid/test)")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Bersihkan output
    if os.path.exists(OUTPUT_DIR):
        print("Membersihkan output lama...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    # Kumpulkan semua entries per split
    all_splits = {"train": [], "valid": [], "test": []}
    stats = {}

    # Folder 1: PaddleOCR format
    print("\n--- Dataset 1 (PaddleOCR format) ---")
    ds1 = process_paddleocr_dataset(os.path.join(SOURCE_DIR, "1"), DATASET_PREFIXES["1"])
    for split in ["train", "valid", "test"]:
        all_splits[split].extend(ds1[split])
    total_ds1 = sum(len(ds1[s]) for s in ds1)
    stats["1"] = {"train": len(ds1["train"]), "valid": len(ds1["valid"]), "test": len(ds1["test"])}
    print(f"  train={len(ds1['train'])}, valid={len(ds1['valid'])}, test={len(ds1['test'])}")

    # Folder 2-5: COCO format
    for num in ["2", "3", "4", "5"]:
        print(f"\n--- Dataset {num} (COCO/Roboflow format) ---")
        ds = process_coco_dataset(os.path.join(SOURCE_DIR, num), DATASET_PREFIXES[num])
        for split in ["train", "valid", "test"]:
            all_splits[split].extend(ds[split])
        stats[num] = {"train": len(ds["train"]), "valid": len(ds["valid"]), "test": len(ds["test"])}
        print(f"  train={len(ds['train'])}, valid={len(ds['valid'])}, test={len(ds['test'])}")

    # Resolve duplicates per split
    print("\nMengecek duplikat...")
    for split in ["train", "valid", "test"]:
        all_splits[split] = resolve_duplicates(all_splits[split])

    # Statistik total
    print("\n" + "=" * 60)
    print("STATISTIK PER DATASET")
    print("=" * 60)
    print(f"{'Dataset':<12} {'Train':>8} {'Valid':>8} {'Test':>8} {'Total':>8}")
    print("-" * 48)
    grand = {"train": 0, "valid": 0, "test": 0}
    for num in ["1", "2", "3", "4", "5"]:
        s = stats[num]
        total = s["train"] + s["valid"] + s["test"]
        print(f"  DS {num:<8} {s['train']:>8} {s['valid']:>8} {s['test']:>8} {total:>8}")
        for k in grand:
            grand[k] += s[k]
    print("-" * 48)
    print(f"  {'TOTAL':<8} {grand['train']:>8} {grand['valid']:>8} {grand['test']:>8} {sum(grand.values()):>8}")

    # Copy & write per split
    print("\n" + "=" * 60)
    print("MENYALIN GAMBAR & MENULIS LABEL")
    print("=" * 60)

    total_success = 0
    total_failed = 0

    for split in ["train", "valid", "test"]:
        entries = all_splits[split]
        if not entries:
            continue
        label_path = os.path.join(OUTPUT_DIR, f"{split}.txt")
        print(f"\n[{split.upper()}] Menyalin {len(entries)} gambar...")
        s, f = copy_and_write(entries, label_path, split)
        total_success += s
        total_failed += f
        print(f"  Berhasil: {s}, Gagal: {f}")

    # Validasi final
    print("\n" + "=" * 60)
    print("VALIDASI FINAL")
    print("=" * 60)

    all_ok = True
    for split in ["train", "valid", "test"]:
        label_path = os.path.join(OUTPUT_DIR, f"{split}.txt")
        if os.path.exists(label_path):
            ok = validate_label_file(label_path, split)
            if not ok:
                all_ok = False

    # Cek konsistensi: total baris == total gambar
    total_lines = 0
    for split in ["train", "valid", "test"]:
        lp = os.path.join(OUTPUT_DIR, f"{split}.txt")
        if os.path.exists(lp):
            with open(lp, 'r', encoding='utf-8') as f:
                total_lines += sum(1 for l in f if l.strip())

    image_files = [f for f in os.listdir(OUTPUT_IMAGES_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    print(f"\nTotal baris (train+valid+test): {total_lines}")
    print(f"Total gambar di images/:        {len(image_files)}")

    if total_lines != len(image_files):
        print(f"[ANOMALI] Jumlah baris ({total_lines}) != gambar ({len(image_files)})!")
        all_ok = False
    else:
        print(f"[OK] Konsisten: {total_lines} baris = {len(image_files)} gambar")

    if not all_ok:
        print("\n[GAGAL] Ada anomali! Cek log di atas.")
        sys.exit(1)

    # Ringkasan
    print("\n" + "=" * 60)
    print("SELESAI!")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    for split in ["train", "valid", "test"]:
        lp = os.path.join(OUTPUT_DIR, f"{split}.txt")
        if os.path.exists(lp):
            with open(lp, 'r', encoding='utf-8') as f:
                n = sum(1 for l in f if l.strip())
            print(f"  {split}.txt: {n} entries")
    print(f"  images/: {len(image_files)} gambar")
    print(f"\nSiap untuk fine-tuning PaddleOCR Detection!")


if __name__ == "__main__":
    merge_all_datasets()
