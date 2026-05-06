from paddleocr import TextDetection

model = TextDetection(model_dir="./result_model", model_name="PP-OCRv5_mobile_det")

# Memprediksi semua gambar di dalam folder data_test
output = model.predict("data_test/indicator_stasiun_AL26013010148_260130_101419.jpg", batch_size=1)

for res in output:
    res.print()
    # Menyimpan hasil visualisasi (gambar) ke output_test/
    res.save_to_img(save_path="./output_test_5/")
    # Menyimpan hasil prediksi dalam bentuk JSON
    # Kita berikan directory path agar disimpan terpisah untuk tiap file
    res.save_to_json(save_path="./output_test_5/")