"""Flask web app for Oral-Cavity-Detection."""
import time
import uuid
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from detect import detect

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "test"
RESULT_DIR = BASE_DIR / "res_img"
ASSET_DIR = BASE_DIR / "assets"
ALLOWED = {"jpg", "jpeg", "png", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.secret_key = "change-me-in-production"


def allowed(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please choose an image first.")
        return redirect(url_for("index"))
    if not allowed(file.filename):
        flash("Unsupported file type. Use JPG, PNG, BMP or WEBP.")
        return redirect(url_for("index"))

    name = f"{uuid.uuid4().hex[:10]}_{secure_filename(file.filename)}"
    path = UPLOAD_DIR / name
    UPLOAD_DIR.mkdir(exist_ok=True)
    file.save(path)
    try:
        result = detect(path)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("index"))
    return render_template("result.html", result=result, original=name, stamp=int(time.time()))


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/res_img/<path:filename>")
def res_img(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.route("/assets/<path:filename>")
def assets(filename):
    if filename.endswith(".pt"):
        abort(403)
    return send_from_directory(ASSET_DIR, filename)


@app.errorhandler(413)
def too_large(_):
    flash("File too large (max 10 MB).")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
