from paddleocr import PaddleOCR
# recognition_model = 'model_best/export_debug'
recognition_model = 'pretrained/PP-OCRv5_mobile_rec_infer'
detection_model = 'pretrained/PP-OCRv5_mobile_det_infer'
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_dir= detection_model,
    text_recognition_model_dir= recognition_model,
    
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False)
result = ocr.predict("./data_test/indicator_stasiun_AL26013010148_260130_101419.jpg")
for res in result:
    res.print()
    res.save_to_img("output4")
    res.save_to_json("output4")
