"""Image transformation. Two providers, chosen with the PROVIDER env variable:

  PROVIDER=hf_space   -> free Hugging Face Space via gradio_client (needs HF_TOKEN)
  PROVIDER=replicate  -> Replicate FLUX Kontext (paid, needs REPLICATE_API_TOKEN)

If PROVIDER is not set: Replicate is used when REPLICATE_API_TOKEN exists,
otherwise the Hugging Face Space. Tokens stay on the server.
"""
import os
import io
import time
import base64
import random
import requests
from PIL import Image

import logging
log = logging.getLogger("timelens")

# ---------- settings ----------
HF_SPACE = os.environ.get("HF_SPACE", "linoyts/qwen-image-2-1-realtime")
HF_RESOLUTION = os.environ.get("HF_RESOLUTION", "1024")   # '512', '1024' or '1536'
REPLICATE_BASE = "https://api.replicate.com/v1"
REPLICATE_MODEL = os.environ.get("REPLICATE_MODEL", "black-forest-labs/flux-kontext-pro")


class TransformError(Exception):
    pass


def _to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


# ---------- Hugging Face Space (free) ----------
def _transform_hf(image_path, prompt: str) -> bytes:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise TransformError("Server is not configured: HF_TOKEN is missing.")
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        raise TransformError("Server is missing the gradio_client package.")

    try:
        try:
            client = Client(HF_SPACE, token=token)          # newer gradio_client
        except TypeError:
            client = Client(HF_SPACE, hf_token=token)       # older gradio_client

        # This Space refers to uploaded photos as <image1>, <image2>, ...
        hf_prompt = "Using <image1> as the input photo: " + prompt

        job = client.submit(
            prompt=hf_prompt,
            input_images=[{"image": handle_file(str(image_path))}],
            seed=random.randint(0, 2**31 - 1),
            resolution=HF_RESOLUTION,
            aspect_ratio="Auto",
            api_name="/generate",
        )
        result = job.result(timeout=150)
    except TransformError:
        raise
    except Exception as e:
        msg = str(e).lower()
        log.error("HF Space call failed: %s: %s", type(e).__name__, e)
        if "quota" in msg or "zerogpu" in msg or "gpu" in msg:
            raise TransformError("The free GPU limit is used up for now. Please try again later.")
        if "timeout" in msg or "timed out" in msg:
            raise TransformError("The free image service took too long. Please try again.")
        raise TransformError("The free image service failed. Please try again later.")

    # result is a file path (or sometimes a tuple/list whose first item is the path)
    path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(path, dict):
        path = path.get("path") or path.get("value")
    try:
        return _to_jpeg_bytes(Image.open(path))
    except Exception:
        raise TransformError("Could not read the generated image.")


# ---------- Replicate (paid) ----------
def _transform_replicate(image_path, prompt: str) -> bytes:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise TransformError("Server is not configured: REPLICATE_API_TOKEN is missing.")

    with open(image_path, "rb") as f:
        data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait=60"}
    body = {"input": {"prompt": prompt, "input_image": data_uri, "output_format": "jpg"}}

    try:
        r = requests.post(f"{REPLICATE_BASE}/models/{REPLICATE_MODEL}/predictions",
                          json=body, headers=headers, timeout=90)
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
        img = Image.open(io.BytesIO(requests.get(url, timeout=60).content))
        return _to_jpeg_bytes(img)
    except Exception:
        raise TransformError("Could not download the generated image.")


# ---------- entry point used by app.py ----------
def transform_image(image_path, prompt: str) -> bytes:
    provider = os.environ.get("PROVIDER", "").strip().lower()
    if not provider:
        provider = "replicate" if os.environ.get("REPLICATE_API_TOKEN") else "hf_space"
    if provider == "replicate":
        return _transform_replicate(image_path, prompt)
    if provider == "hf_space":
        return _transform_hf(image_path, prompt)
    raise TransformError("Server is misconfigured: unknown PROVIDER.")
