"""Golden-set eval. "Behtar hua" ko feeling se number me badalta hai.

Do tarah ki entries (eval/golden.json):

  Retrieval:  {"q": "...", "expect_video_id": "abc" | ["abc", "xyz"],
               "expect_around_sec": 230, "tolerance_sec": 120}
      HIT = expected video top-k me ho, AUR (agar around_sec diya hai) uska
      koi chunk [around - tol, around + tol] se overlap kare.

  Refusal:    {"q": "React hooks?", "expect_refusal": true}
      HIT = poori answer() pipeline EXACT refusal string lautaye.

Original se farak: refusal check strict hai. Original `not grounded` ko
refusal maanta tha — matlab model ne apni knowledge se bina citation answer
diya toh bhi "pass". Woh asli failure hai, toh hum use alag ginte hain.
"""

import json
from pathlib import Path

from ytrag.answer import answer
from ytrag.config import MAX_DISTANCE, PROJECT_ROOT, REFUSAL
from ytrag.index import search

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden.json"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    # "_" se shuru hone wale q = template/comment, skip
    return [e for e in entries if not str(e.get("q", "")).startswith("_")]


def _check_retrieval(entry: dict, k: int) -> dict:
    # max_distance=2.0 -> cutoff band. Eval ko dekhna hai retrieval ne ASAL me
    # kya diya. Cutoff ka asar alag column (`cut`) me report hota hai.
    hits = search(entry["q"], top_k=k, max_distance=2.0)

    expected = entry["expect_video_id"]
    expected = {expected} if isinstance(expected, str) else set(expected)
    around = entry.get("expect_around_sec")
    tol = int(entry.get("tolerance_sec", 120))

    def correct(chunk) -> bool:
        if chunk.video_id not in expected:
            return False
        if around is None:
            return True  # video-level entry
        # Interval overlap: [start, end] aur [around-tol, around+tol]
        return chunk.start_sec <= around + tol and chunk.end_sec >= around - tol

    # Pehla sahi result kis rank pe aaya? (1-based, nahi aaya toh None)
    rank = next((i for i, (c, _) in enumerate(hits, start=1) if correct(c)), None)
    top_dist = hits[0][1] if hits else 2.0

    return {
        "kind": "retrieval",
        "q": entry["q"],
        "hit": rank is not None,
        "rank": rank,
        "video_hit": any(c.video_id in expected for c, _ in hits),
        # Genuine sawaal jiska best match bhi cutoff ke bahar — production me refuse hoga!
        "cut": top_dist > MAX_DISTANCE,
        "top_dist": round(top_dist, 3),
        "got": [f"{c.video_title[:40]} @ {c.timestamp} ({d:.3f})" for c, d in hits[:3]],
    }


def _check_refusal(entry: dict, k: int) -> dict:
    result = answer(entry["q"], top_k=k)
    refused = result["answer"].strip() == REFUSAL
    if refused:
        how = "cutoff" if result["retrieved"] == 0 else "llm"  # guard 1 ya guard 2
    elif result["grounded"]:
        how = "ANSWERED WITH CITATIONS"  # sabse bura: fake timestamps
    else:
        how = "answered ungrounded"  # original isko pass gin leta
    return {
        "kind": "refusal",
        "q": entry["q"],
        "hit": refused,
        "how": how,
        "retrieved": result["retrieved"],
        "answer": result["answer"][:120],
    }


def run_eval(k: int = 5, path: Path = GOLDEN_PATH) -> dict:
    entries = load_golden(path)
    results = [
        _check_refusal(e, k) if e.get("expect_refusal") else _check_retrieval(e, k)
        for e in entries
    ]
    ret = [r for r in results if r["kind"] == "retrieval"]
    ref = [r for r in results if r["kind"] == "refusal"]

    summary: dict = {"k": k, "max_distance": MAX_DISTANCE, "results": results}
    if ret:
        n = len(ret)
        summary["retrieval"] = {
            "n": n,
            f"hit@{k}": sum(r["hit"] for r in ret) / n,
            "top1": sum(r["rank"] == 1 for r in ret) / n,
            # MRR: har sawaal ka 1/rank (miss = 0), average. Rank 1 -> 1.0,
            # rank 2 -> 0.5, rank 5 -> 0.2. Sirf "mila/nahi mila" se zyada batata hai.
            "mrr": sum(1 / r["rank"] for r in ret if r["rank"]) / n,
            "video_hit": sum(r["video_hit"] for r in ret) / n,
            "cut_by_cutoff": sum(r["cut"] for r in ret),
        }
    if ref:
        summary["refusal"] = {
            "n": len(ref),
            "rate": sum(r["hit"] for r in ref) / len(ref),
        }
    return summary