# Oral Cavity Detection

Flask web app that detects the oral cavity in an uploaded photo and flags suspicious regions.

## Structure
- `app.py` – Flask routes (upload, predict, serve images)
- `detect.py` – detection logic (YOLO model or built-in OpenCV demo mode)
- `templates/` – `index.html` (upload), `result.html` (output)
- `static/` – `style.css`, `script.js`
- `test/` – uploaded images + `test_app.py` unit tests
- `res_img/result.png` – latest annotated output
- `assets/img/` – UI images (put trained weights at `assets/best.pt`)

## Run
```
pip install -r requirements.txt
python app.py          # open http://127.0.0.1:5000
python -m unittest discover -s test
```

## Modes
- **Demo mode** (default): HSV colour segmentation finds the mucosa region and flags white patches and dark-red lesions.
- **YOLO mode**: drop a trained YOLOv8 weights file at `assets/best.pt` (or set `ORAL_WEIGHTS`) and it is used automatically.
  Train on an oral-lesion dataset (e.g. Roboflow/Kaggle) with `yolo detect train data=data.yaml model=yolov8n.pt`.

## Disclaimer
Educational demo, not a medical device or diagnosis.
