# YT Lecture RAG

Ask a DSA question in Hinglish or English and jump straight to the second of the
lecture where it is explained. Timestamp-level retrieval over 126 YouTube DSA
lectures (~68 hours, 2933 chunks), with an optional grounded answer that refuses
when the lectures don't cover the topic.

## How it works

1. **Query rewrite**: a small Groq model turns a Hinglish question into a short
   English DSA search query ("paas paas wale ghar nahi loot sakte" →
   "cannot rob houses that are adjacent to each other"). Rewrites are cached on disk.
2. **Retrieve**: the rewritten query is embedded with all-MiniLM-L6-v2 via
   [fastembed](https://github.com/qdrant/fastembed) (ONNX, no torch) and searched in
   a local Qdrant index. Transcripts are chunked in 75 s time windows with 15 s
   overlap, so every hit maps to an exact timestamp. A distance cutoff drops weak
   matches, and a title-word boost re-ranks the rest.
3. **Answer (optional)**: an LLM answers only from the retrieved excerpts, with
   three refusal guards: (1) nothing survives the cutoff → refuse without calling
   the LLM; (2) the model says the excerpts don't cover it → refuse; (3) no valid
   citation in the answer → shown as ungrounded, no links. Citations are
   renumbered to only the ones actually used.
4. **Serve**: FastAPI + a single-page UI that plays results in an embedded
   YouTube IFrame player at the matched second.

## Evaluation

Golden set: 20 retrieval questions (mostly Hinglish, many phrased without the
problem's name) + 8 off-topic questions that must be refused. `k = 5`.

| Setup | hit@5 | top-1 | MRR | Refusal |
|---|---|---|---|---|
| Baseline (no query rewrite) | 65% | ~55% | ~0.59 | 100% |
| + Query rewrite (sentence-transformers) | 95% | 65% | 0.77 | 100% |
| + fastembed ONNX runtime (current) | 95% | 65% | 0.77 | 100% |

**Root cause of the baseline misses:** the embedder is English-only, so Hinglish
queries matched other Hinglish chunks by *language* rather than by *topic*.
Rewriting the query into English DSA terms fixed that. Caching the rewrites made
the eval deterministic (zero run-to-run noise), so later changes, like switching
the embedding runtime, can be measured exactly. The switch to fastembed changed
no metric.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and a [Groq](https://console.groq.com/) API key.

```bash
uv sync
echo GROQ_API_KEY=your_key_here > .env
uv run ytrag reindex     # embeds the committed transcripts with the bundled model in models/
uv run ytrag serve       # http://127.0.0.1:8000
```

Other commands: `uv run ytrag search "..."`, `uv run ytrag ask "..."`,
`uv run ytrag eval -v`, `uv run ytrag stats`.

## Deployment

Runs on Vercel (Python, zero-config FastAPI via the root `app.py`). The Qdrant
index is committed; on Vercel's read-only filesystem it is copied to `/tmp` on
cold start. The ONNX model is committed in `models/` and loaded read-only, so
nothing is downloaded from Hugging Face at runtime (`scripts/bundle_model.py`
regenerates it if the model changes). Set `GROQ_API_KEY` in the project's
environment variables. `/health` returns 503 and names the failed component if
startup goes wrong.

### Incident: demo down on Vercel (Oct 2026)

**Symptom:** every route returned 500.
**Root cause:** on startup, fastembed downloaded the model through hf-xet, which
writes its cache under `HF_HOME` (`~/.cache`, read-only on Vercel). The GCS
fallback also failed, so startup crashed.
**Fix:** first, `HF_HOME`/`HF_XET_CACHE` under `/tmp` and a non-fatal startup with
an honest `/health`; then the ONNX model shipped in the bundle, so there is no
Hugging Face call at runtime.
**Proof:** offline boot with `VERCEL=1` and Hugging Face + GCS blocked was healthy
with zero attempts to those hosts; a control run without the bundled model hit
the block. Eval unchanged: hit@5 95%, MRR 0.77, refusal 100%.
**Remaining dependency:** the bundle (313 MB) exceeds Vercel's standard size, so
Python packages are installed from PyPI on each cold start.
