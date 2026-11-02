"""Detection module for the Oral-Cavity-Detection project.

Two modes (chosen automatically):
  1. YOLO mode  - if trained weights exist at assets/best.pt (or $ORAL_WEIGHTS),
                  they are used via `ultralytics`.
  2. Demo mode  - otherwise a colour/HSV based OpenCV pipeline locates the oral
                  cavity and flags white patches and red lesions, so the whole
                  app runs end-to-end without a trained model.
"""
import os
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("ORAL_WEIGHTS", BASE_DIR / "assets" / "best.pt"))
RESULT_PATH = BASE_DIR / "res_img" / "result.png"
MAX_SIDE = 900

COLORS = {  # BGR
    "Oral cavity": (166, 160, 20),
    "White patch": (255, 255, 255),
    "Red lesion": (60, 60, 230),
}
_model = None


def _load_model():
    global _model
    if _model is None and WEIGHTS.exists():
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def _resize(img):
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img


def _yolo_detect(model, img):
    res = model.predict(img, conf=0.25, verbose=False)[0]
    dets = []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        dets.append({"label": res.names[int(box.cls[0])], "confidence": round(float(box.conf[0]), 3),
                     "box": [x1, y1, x2, y2]})
    return dets


def _bbox_dets(mask, label, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            out.append({"label": label, "confidence": None, "box": [x, y, x + w, y + h]})
    return out


def _demo_detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    area = img.shape[0] * img.shape[1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    red_hue = (h <= 12) | (h >= 160)
    mucosa = (red_hue & (s > 50) & (v > 50)).astype(np.uint8) * 255
    mucosa = cv2.morphologyEx(mucosa, cv2.MORPH_CLOSE, k)
    mucosa = cv2.morphologyEx(mucosa, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(mucosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.04 * area:
        return []
    x, y, w, hh = cv2.boundingRect(big)
    dets = [{"label": "Oral cavity", "confidence": None, "box": [x, y, x + w, y + hh]}]

    cavity = np.zeros(mucosa.shape, np.uint8)
    cv2.drawContours(cavity, [big], -1, 255, -1)
    cavity = cv2.dilate(cavity, k)
    small = k[:7, :7]
    white = ((s < 45) & (v > 190)).astype(np.uint8) * 255
    red = (red_hue & (s > 170) & (v < 150) & (v > 40)).astype(np.uint8) * 255
    for mask, label in ((white, "White patch"), (red, "Red lesion")):
        mask = cv2.bitwise_and(mask, cavity)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        dets += _bbox_dets(mask, label, 0.002 * area)
    return dets


def _draw(img, dets):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        col = COLORS.get(d["label"], (0, 200, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        text = d["label"] + (f" {d['confidence']:.2f}" if d["confidence"] is not None else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, text, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def detect(image_path, out_path=RESULT_PATH):
    """Run detection on an image file, save annotated image, return a result dict."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read the image. Upload a valid JPG/PNG file.")
    img = _resize(img)
    model = _load_model()
    mode = "YOLO model" if model else "Demo (colour-based)"
    dets = _yolo_detect(model, img) if model else _demo_detect(img)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), _draw(img, dets))
    findings = [d for d in dets if d["label"] != "Oral cavity"]
    return {"mode": mode, "detections": dets, "count": len(dets),
            "findings": len(findings), "result_image": Path(out_path).name}


if __name__ == "__main__":
    import sys
    print(detect(sys.argv[1]))
