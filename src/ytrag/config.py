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


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------
# all-MiniLM-L6-v2: 87 MB, 384-dim, CPU pe tez.
# Original ka measurement (2933 chunks): bge-m3 (4.35 GB, 55 min index)
# top-1 12/12 vs MiniLM (87 MB, 1.6 min) 11/12. 50x chhota, 30x tez,
# sirf 1 sawaal ka farak — aur woh bhi rank 2 pe aa jaata hai.
EMBED_MODEL = os.getenv("YTRAG_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("YTRAG_EMBED_BATCH", "16"))
# Kuch models ko QUERY ke aage ek instruction chahiye (sirf query, documents
# nahi) — jaise bge-*-en-v1.5. MiniLM ko nahi. Model badlo toh model card padho:
# galat prefix error nahi deta, bas chupchaap results kharab kar deta hai.
EMBED_QUERY_PREFIX = os.getenv("YTRAG_EMBED_QUERY_PREFIX", "")

# ------------------------------------------------------------------
# Vector store (Qdrant)
# ------------------------------------------------------------------
# QDRANT_URL khaali -> local folder mode (data/qdrant), koi account nahi.
# Set karo -> Qdrant Cloud (deploy ke time).
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
# Final naam me embedding dim juddega: "dsa_lectures_384"
COLLECTION = os.getenv("YTRAG_COLLECTION", "dsa_lectures")
UPSERT_BATCH = int(os.getenv("YTRAG_UPSERT_BATCH", "128"))



# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
TOP_K = int(os.getenv("YTRAG_TOP_K", "6"))
# Cosine distance = 1 - similarity. Isse door wale results LLM tak nahi jaate.
#
# 0.60 original ka measured value hai (20 genuine + 10 off-topic sawaal, poore
# index pe). Dono populations OVERLAP karti hain: sabse kharab genuine sawaal
# ("number of islands") 0.568, sabse achha off-topic ("backprop") 0.409.
# 0.50 karne pe "hashmap kab use karna chahiye" jaise asli sawaal refuse hue.
# Toh ye coarse pre-filter hai, asli guard nahi — semantic judgement M6 me LLM karega.
MAX_DISTANCE = float(os.getenv("YTRAG_MAX_DISTANCE", "0.6"))
# Isse kam distance = solid match, UI "confident" dikhayega (M8).
CONFIDENT_DISTANCE = float(os.getenv("YTRAG_CONFIDENT_DISTANCE", "0.45"))
# Query ka har meaningful word jo lecture TITLE me bhi hai -> distance me se
# itna minus. Measured: top-1 accuracy 9/12 -> 12/12. 0 = band.
TITLE_BOOST = float(os.getenv("YTRAG_TITLE_BOOST", "0.06"))


# ------------------------------------------------------------------
# LLM (Groq)
# ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("YTRAG_GROQ_MODEL", "openai/gpt-oss-120b")
# gpt-oss reasoning model hai: answer se pehle "sochne" ke tokens bhi isi
# budget me se jaate hain. HireMeAI me yahi budget kam padne pe khaali
# answer aaya tha. 4096 = reasoning + 4-6 line answer ke liye kaafi.
GROQ_MAX_TOKENS = int(os.getenv("YTRAG_GROQ_MAX_TOKENS", "4096"))

# Exact string jo system bolta hai jab answer lectures me nahi hai.
# Ek jagah define — answer.py, eval aur frontend teeno isi se match karte hain.
REFUSAL = "Ye topic in lectures me cover nahi hua."