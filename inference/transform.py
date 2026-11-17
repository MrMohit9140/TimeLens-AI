"""Image transformation via Replicate (FLUX Kontext). The API token stays on the server."""
import os
import io
import time
import base64
import requests
from PIL import Image

API_BASE = "https://api.replicate.com/v1"
MODEL = os.environ.get("REPLICATE_MODEL", "black-forest-labs/flux-kontext-pro")


class TransformError(Exception):
    pass


def transform_image(image_path, prompt: str) -> bytes:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise TransformError("Server is not configured: REPLICATE_API_TOKEN is missing.")

    with open(image_path, "rb") as f:
        data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait=60"}
    body = {"input": {"prompt": prompt, "input_image": data_uri, "output_format": "jpg"}}

    try:
        r = requests.post(f"{API_BASE}/models/{MODEL}/predictions", json=body, headers=headers, timeout=90)
    except requests.RequestException:
        raise TransformError("Could not reach the image-generation service.")

    if r.status_code == 401:
        raise TransformError("Image service rejected the API token.")
    if r.status_code == 402:
        raise TransformError("Image service account has insufficient credit.")
    if r.status_code == 429:
        raise TransformError("Image service is busy. Please retry in a moment.")
    if r.status_code >= 400:
        raise TransformError(f"Image service error ({r.status_code}).")

    pred = r.json()
    # Poll if the "wait" window ended before the job finished.
    deadline = time.time() + 120
    while pred.get("status") in ("starting", "processing") and time.time() < deadline:
        time.sleep(2)
        try:
            pred = requests.get(pred["urls"]["get"], headers=headers, timeout=30).json()
        except requests.RequestException:
            continue

    if pred.get("status") != "succeeded":
        raise TransformError("Generation did not finish: " + str(pred.get("error") or pred.get("status")))

    out = pred.get("output")
    url = out[0] if isinstance(out, list) else out
    if not url:
        raise TransformError("Image service returned no image.")

    try:
        img_bytes = requests.get(url, timeout=60).content
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise TransformError("Could not download the generated image.")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()
