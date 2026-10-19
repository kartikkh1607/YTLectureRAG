"""EMBED_MODEL ki ONNX files models/ me copy karo (git me commit hoti hain).

Kyun? Vercel pe cold start pe Hugging Face se download = read-only FS, xet
cache, GCS fallback — sab fail ho sakte hain (M9 ke baad hua). Model bundle
me ho toh runtime pe network chahiye hi nahi.

Model badla (YTRAG_EMBED_MODEL) toh ye dobara chalao, phir models/ commit karo:
    uv run python scripts/bundle_model.py
GitHub limit: har file < 100 MB. MiniLM ka model.onnx ~86 MB.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# .env HF_HUB_OFFLINE=1 rakhta hai; ye script hi ek jagah hai jahan download
# chahiye. huggingface_hub import ke time padhta hai -> fastembed import se pehle.
os.environ["HF_HUB_OFFLINE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastembed import TextEmbedding  # noqa: E402

from ytrag.config import EMBED_MODEL, EMBED_MODEL_DIR  # noqa: E402

GITHUB_LIMIT = 100 * 1024 * 1024

with tempfile.TemporaryDirectory() as tmp:
    model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=tmp, lazy_load=True)
    src = Path(model.model._model_dir)
    if EMBED_MODEL_DIR.exists():
        shutil.rmtree(EMBED_MODEL_DIR)
    EMBED_MODEL_DIR.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            if size >= GITHUB_LIMIT:
                sys.exit(f"{f.name} is {size / 2**20:.1f} MB — over GitHub's 100 MB limit.")
            shutil.copy2(f, EMBED_MODEL_DIR / f.name)
            print(f"{f.name:28s} {size / 2**20:7.2f} MB")
    print(f"-> {EMBED_MODEL_DIR}")
