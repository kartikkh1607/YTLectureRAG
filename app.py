"""Vercel entrypoint. Vercel root pe app.py dhoondhta hai jisme `app` variable ho.

src layout hai, toh `import ytrag` tabhi chalega jab package install ho. Vercel
install kare ya na kare — src/ ko sys.path me daal do, dono case me chalega.
Local pe isko mat chhuo: wahan `uv run ytrag serve` hi use karo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# config pehle: Vercel pe HF_HOME /tmp pe set karta hai, fastembed/huggingface_hub
# import hone se pehle (woh import ke time hi path padh lete hain).
import ytrag.config  # noqa: E402,F401
from ytrag.api import app  # noqa: E402,F401
