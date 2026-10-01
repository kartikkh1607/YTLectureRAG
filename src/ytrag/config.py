"""Saare tunables yahan. Baaki koi file os.getenv() directly nahi padhegi.

Kyun? Ek hi jagah dekh ke pata chal jaata hai ki system ke knobs kya hain.
Har value env variable se override ho sakti hai, toh code badle bina
experiment kar sakte ho (jaise chunk size 75 -> 60).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# __file__ = E:\YTLectureRAG\src\ytrag\config.py
# parents[0] = ytrag, parents[1] = src, parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# .env ko project root se load karo, chahe command kisi bhi folder se chalao.
# override=False -> agar terminal me already env var set hai, toh woh jeetega.
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = Path(os.getenv("YTRAG_DATA_DIR", PROJECT_ROOT / "data"))

# Precious vs disposable split:
#   transcripts/ -> ghanton ka GPU time, git me commit hota hai
#   audio/       -> kabhi bhi dobara download, gitignored
#   qdrant/      -> transcripts se reindex karke minutes me ban jaata hai
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
AUDIO_DIR = DATA_DIR / "audio"
QDRANT_PATH = DATA_DIR / "qdrant"
# Note: original code import hote hi folders bana deta tha (side effect).
# Hum nahi banayenge — jo function file likhega, woh khud mkdir karega.

# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
# 75 sec ~ ek explained idea (~180-220 words bolne me).
CHUNK_SECONDS = int(os.getenv("YTRAG_CHUNK_SECONDS", "75"))
# Agla chunk pichle ke last ~15 sec dobara padhega, taaki boundary pe
# aadha explanation do chunks me na bat jaaye.
CHUNK_OVERLAP_SECONDS = int(os.getenv("YTRAG_CHUNK_OVERLAP", "15"))
# Isse kam words wala chunk (intro music, "hello everyone") index nahi hoga.
MIN_CHUNK_WORDS = int(os.getenv("YTRAG_MIN_CHUNK_WORDS", "15"))

# Retrieval answer wala chunk pakadta hai, par explanation aksar thoda
# pehle shuru hota hai. Citation link 5 sec peeche se chalega.
LINK_REWIND_SECONDS = int(os.getenv("YTRAG_LINK_REWIND", "5"))