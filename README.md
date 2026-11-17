# TimeLens AI — Your Photo. Another Era.

Flask web app that turns an uploaded photo into a historical-style photo for a chosen year,
using a pretrained image-editing model (FLUX Kontext via Replicate) with year-specific prompts.
An optional ResNet50 (1980s vs non-1980s) classifier you train yourself in Colab adjusts the prompt
and scores results.

## Run locally (VS Code)
```
python -m venv venv
venv\Scripts\activate        # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
set REPLICATE_API_TOKEN=r8_xxx   # Mac/Linux: export REPLICATE_API_TOKEN=r8_xxx
python app.py
```
Open http://127.0.0.1:5000

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render dashboard → New → Blueprint (or Web Service) → pick the repo.
3. Build: `pip install -r requirements.txt` · Start: `gunicorn app:app --workers 1 --threads 4 --timeout 180`
4. Environment → add `REPLICATE_API_TOKEN` (get it at replicate.com/account/api-tokens; billing must be enabled).
5. Deploy, then open `/health`.

## Adding the trained classifier
Commit `models/timelens_resnet50_1980s.pth` (use Git LFS if > 100 MB; ResNet50 is ~95 MB),
and change the Render build command to `pip install -r requirements-ml.txt`.
Free Render has 512 MB RAM, so PyTorch may run out of memory; use a paid instance if so.

## Honest notes
- Era descriptions in `inference/prompts.py` are hand-written historical knowledge, not learned from data.
- Identity preservation is not guaranteed; check results.
- The classifier only exists after you train it on real, verified dataset labels.
