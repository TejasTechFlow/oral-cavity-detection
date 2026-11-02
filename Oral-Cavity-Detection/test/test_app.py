"""Run from project root:  python -m unittest discover -s test"""
import io, sys, unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app
from detect import detect


def synthetic_mouth(path):
    img = np.full((500, 600, 3), (90, 120, 150), np.uint8)                   # skin-ish background (BGR)
    cv2.ellipse(img, (300, 260), (220, 150), 0, 0, 360, (90, 70, 190), -1)   # pink-red mucosa
    cv2.circle(img, (230, 250), 30, (245, 245, 245), -1)                      # white patch
    cv2.circle(img, (390, 290), 28, (20, 10, 120), -1)                        # dark red lesion
    cv2.imwrite(str(path), img)


class OralTests(unittest.TestCase):
    def setUp(self):
        self.sample = Path(__file__).parent / "sample_mouth.png"
        synthetic_mouth(self.sample)

    def test_detect_pipeline(self):
        r = detect(self.sample, Path(__file__).parents[1] / "res_img" / "result.png")
        labels = {d["label"] for d in r["detections"]}
        self.assertTrue({"Oral cavity", "White patch", "Red lesion"} <= labels, labels)

    def test_routes(self):
        c = app.test_client()
        self.assertEqual(c.get("/").status_code, 200)
        r = c.post("/predict", data={"image": (io.BytesIO(self.sample.read_bytes()), "m.png")},
                   content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Detection Result", r.data)
        bad = c.post("/predict", data={"image": (io.BytesIO(b"x"), "a.txt")}, content_type="multipart/form-data")
        self.assertEqual(bad.status_code, 302)


if __name__ == "__main__":
    unittest.main()
