"""TimeLens AI - Flask backend.

Flow: validate upload -> (optional) ML classifier -> build year prompt ->
image-editing model (Replicate, key kept server-side) -> save result.
"""
import os
import time
import uuid
import logging
import threading
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from PIL import Image, UnidentifiedImageError

from inference.prompts import build_prompt, SUPPORTED_YEARS, DEFAULT_YEAR
from inference.transform import transform_image, TransformError
from inference import classifier

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"
UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_MB = 10
MAX_SIDE = 1536                 # uploaded image is downscaled to this longest side
FILE_TTL_SECONDS = 30 * 60      # temp files are deleted after 30 minutes
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_HOUR", "6"))  # protects your API credits

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("timelens")

_hits = defaultdict(deque)
_lock = threading.Lock()


def cleanup_old_files():
    """Delete uploads/generated files older than FILE_TTL_SECONDS."""
    now = time.time()
    for folder in (UPLOAD_DIR, GENERATED_DIR):
        for p in folder.iterdir():
            if p.name == ".gitkeep":
                continue
            try:
                if now - p.stat().st_mtime > FILE_TTL_SECONDS:
                    p.unlink()
            except OSError:
                pass


def rate_limited(ip: str) -> bool:
    now = time.time()
    with _lock:
        q = _hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        return False


@app.route("/")
def index():
    years = list(range(SUPPORTED_YEARS[0], SUPPORTED_YEARS[1] + 1))
    return render_template("index.html", years=years, default_year=DEFAULT_YEAR)


@app.route("/health")
def health():
    return jsonify(status="ok", classifier_available=classifier.is_available())


@app.route("/generate", methods=["POST"])
def generate():
    cleanup_old_files()
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr) or "").split(",")[0].strip()
    if rate_limited(ip):
        return jsonify(error="Too many requests. Please try again later."), 429

    file = request.files.get("photo")
    if not file or not file.filename:
        return jsonify(error="Please choose a photograph."), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify(error="Only JPG, JPEG, PNG and WEBP files are allowed."), 400

    try:
        year = int(request.form.get("year", DEFAULT_YEAR))
    except ValueError:
        return jsonify(error="Invalid year."), 400
    if not (SUPPORTED_YEARS[0] <= year <= SUPPORTED_YEARS[1]):
        return jsonify(error=f"Year must be between {SUPPORTED_YEARS[0]} and {SUPPORTED_YEARS[1]}."), 400

    job = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{job}.jpg"
    try:
        # Verify it is really an image, then normalise: RGB, capped size, JPEG.
        try:
            img = Image.open(file.stream)
            img.verify()
            file.stream.seek(0)
            img = Image.open(file.stream).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify(error="That file is not a valid image."), 400
        if min(img.size) < 128:
            return jsonify(error="Image is too small (minimum 128 px)."), 400
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(upload_path, "JPEG", quality=95)

        # ML classifier on the ORIGINAL photo (None if model not installed)
        orig_score = classifier.predict_1980s_probability(upload_path)

        prompt = build_prompt(year, original_1980s_prob=orig_score)
        result_bytes = transform_image(upload_path, prompt)

        out_path = GENERATED_DIR / f"{job}.jpg"
        out_path.write_bytes(result_bytes)

        gen_score = classifier.predict_1980s_probability(out_path) if (orig_score is not None and 1980 <= year <= 1989) else None

        return jsonify(
            year=year,
            result_url=f"/result/{job}.jpg",
            classifier_used=orig_score is not None,
            original_1980s_probability=orig_score,
            generated_1980s_probability=gen_score,
        )
    except TransformError as e:
        log.warning("Transform failed: %s", e)
        return jsonify(error=str(e)), 502
    except Exception:
        log.exception("Unexpected error")
        return jsonify(error="Something went wrong while generating. Please try again."), 500
    finally:
        # Uploaded original is never kept
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.route("/result/<name>")
def result(name):
    # Only 32-hex-char random names are valid; no directory traversal possible.
    stem, _, ext = name.partition(".")
    if ext != "jpg" or len(stem) != 32 or not all(c in "0123456789abcdef" for c in stem):
        abort(404)
    resp = send_from_directory(GENERATED_DIR, name, as_attachment=request.args.get("download") == "1",
                               download_name=f"timelens_{stem[:8]}.jpg")
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
