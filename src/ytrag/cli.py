"""`ytrag` command line. Har command ek function, typer type hints se args banata hai."""

import time

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True, help="YT Lecture RAG")
console = Console()

# Heavy imports (torch, qdrant) commands ke ANDAR hain, top pe nahi.
# Isliye `ytrag --help` turant khulta hai, 10 sec model load ka wait nahi.


@app.command()
def reindex(
    replace: bool = typer.Option(
        False, "--replace", help="Har video ke purane chunks pehle delete karo (chunk size badla ho tab)."
    ),
):
    """Cached transcripts se re-chunk + re-embed. Kabhi re-transcribe nahi karta."""
    from ytrag.chunk import chunk_segments
    from ytrag.index import collection_name, count_points, delete_video, upsert_chunks
    from ytrag.transcribe import (
        cached_video_ids,
        load_transcript,
        segments_from_transcript,
        video_from_transcript,
    )

    ids = cached_video_ids()
    if not ids:
        console.print("[red]Koi transcript nahi mila data/transcripts me.[/red]")
        raise typer.Exit(1)

    console.print(f"Reindexing {len(ids)} transcripts -> [bold]{collection_name()}[/bold]")
    t0 = time.perf_counter()
    total = skipped = 0
    with console.status("embedding...") as status:
        for n, vid in enumerate(ids, start=1):
            data = load_transcript(vid)
            if data is None:
                skipped += 1  # corrupt file poore run ko nahi rokegi
                continue
            if replace:
                delete_video(vid)
            chunks = chunk_segments(video_from_transcript(data), segments_from_transcript(data))
            total += upsert_chunks(chunks)
            status.update(f"embedding... {n}/{len(ids)} videos, {total} chunks")

    console.print(
        f"[green]Done[/green]: {total} chunks in {time.perf_counter() - t0:.0f}s "
        f"| collection now {count_points()} points | skipped {skipped}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Sawaal, Hinglish ya English"),
    top_k: int = typer.Option(6, "--top-k", "-k"),
    raw: bool = typer.Option(False, "--raw", help="MAX_DISTANCE cutoff band — sab dikhao."),
):
    """Sirf retrieval, LLM nahi (jab tak rewrite OFF hai). Cutoff tune karne ke liye."""
    from ytrag.config import MAX_DISTANCE, QUERY_REWRITE
    from ytrag.index import search as do_search
    from ytrag.index import title_overlap

    # Jo query ASAL me embed hogi. Rewrite cached hai, toh search() ke andar
    # dobara call pe LLM call nahi hogi.
    effective = query
    if QUERY_REWRITE:
        from ytrag.rewrite import rewrite_query

        effective = rewrite_query(query)
        console.print(f"[dim]rewritten: {effective}[/dim]")

    hits = do_search(query, top_k=top_k, max_distance=2.0 if raw else None)
    if not hits:
        console.print(f"[yellow]Kuch nahi mila (cutoff {MAX_DISTANCE}). --raw try karo.[/yellow]")
        return

    table = Table(title=f"{query!r}  (cutoff: {'off' if raw else MAX_DISTANCE})")
    table.add_column("#", justify="right")
    table.add_column("dist", justify="right")
    table.add_column("title+", justify="right")  # kitne words title se match hue
    table.add_column("lecture")
    table.add_column("at (click)", justify="right")
    for i, (chunk, dist) in enumerate(hits, start=1):
        table.add_row(
            str(i),
            f"{dist:.3f}",
            str(title_overlap(effective, chunk.video_title)),  # wahi query jo boost me gayi
            chunk.video_title[:60],
            # [link=URL]text[/link] -> terminal hyperlink. VS Code terminal me Ctrl+Click.
            f"[link={chunk.url}]{chunk.timestamp}[/link]",
        )
    console.print(table)
    console.print(f"Top hit: {hits[0][0].url}")


@app.command()
def stats():
    """Index me abhi kya hai — aur kaunse mode me (local/cloud) chal rahe ho."""
    from ytrag.config import EMBED_MODEL, QDRANT_PATH, QDRANT_URL
    from ytrag.index import collection_name, count_points, indexed_video_ids
    from ytrag.transcribe import cached_video_ids

    # Lesson from M4: jo config decide karta hai, woh PRINT karo.
    mode = f"cloud {QDRANT_URL}" if QDRANT_URL else f"local {QDRANT_PATH}"
    console.print(f"Qdrant       : {mode}")
    console.print(f"Embed model  : {EMBED_MODEL}")
    console.print(f"Collection   : {collection_name()}")
    console.print(f"Points       : {count_points()}")
    console.print(f"Videos       : {len(indexed_video_ids())} indexed / {len(cached_video_ids())} transcripts")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Sawaal, Hinglish ya English"),
    top_k: int = typer.Option(6, "--top-k", "-k"),
):
    """Grounded answer + clickable timestamps (Groq LLM)."""
    from rich.markup import escape

    from ytrag.answer import answer

    result = answer(question, top_k=top_k)

    color = "green" if result["grounded"] else "yellow"
    # escape(): LLM ke text me "[bold]" jaisa kuch aaya toh rich use style samjhega.
    # User/LLM text ko hamesha escape karke markup me daalo.
    console.print(f"\n[{color}]{escape(result['answer'])}[/{color}]\n")

    for i, c in enumerate(result["citations"], start=1):
        console.print(f"  [{i}] [link={c['url']}]{c['timestamp']}[/link]  {c['title'][:60]}  (dist {c['distance']})")

    # Debug line: retrieval ne kitna diya, model ne kitna use kiya
    console.print(
        f"\n[dim]retrieved {result['retrieved']} | cited {len(result['citations'])} "
        f"| grounded {result['grounded']}[/dim]"
    )


@app.command()
def find(
    keyword: str = typer.Argument(..., help="Exact word/phrase, transcripts me grep"),
    limit: int = typer.Option(15, "--limit", "-n"),
):
    """Transcripts me plain keyword search — golden set likhne ke liye.

    Embeddings use NAHI karta, jaan-boojh ke. Golden set ka "sahi jawab" system
    ke apne search se nikaloge toh eval khud ko hi grade karega (circular).
    """
    from ytrag.models import format_timestamp
    from ytrag.transcribe import cached_video_ids, load_transcript

    kw = keyword.lower()
    shown = 0
    for vid in cached_video_ids():
        data = load_transcript(vid)
        if not data:
            continue
        for s in data["segments"]:
            if kw in s["text"].lower():
                start = int(s["start"])
                console.print(
                    f"[cyan]{vid}[/cyan] {start:>5}s ({format_timestamp(start)})  "
                    f"{data['title'][:50]}"
                )
                shown += 1
                if shown >= limit:
                    return
    if shown == 0:
        console.print(f"[yellow]'{keyword}' kisi transcript me nahi mila.[/yellow]")


@app.command(name="eval")  # "eval" Python ka built-in hai, isliye function ka naam alag
def eval_cmd(
    k: int = typer.Option(5, "-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Har entry dikhao, sirf misses nahi."),
):
    """Golden set pe retrieval + refusal score."""
    from ytrag.config import QUERY_REWRITE
    from ytrag.evaluate import run_eval

    console.print(f"[bold]query rewrite: {'ON' if QUERY_REWRITE else 'OFF'}[/bold]")
    s = run_eval(k=k)
    for r in s["results"]:
        if not verbose and r["hit"]:
            continue
        mark = "[green]PASS[/green]" if r["hit"] else "[red]MISS[/red]"
        if r["kind"] == "retrieval":
            extra = f"rank={r['rank']} top_dist={r['top_dist']}" + (" [red]CUT[/red]" if r["cut"] else "")
            console.print(f"{mark} {r['q']}  ({extra})")
            if not r["hit"]:
                for g in r["got"]:
                    console.print(f"       got: {g}")
        else:
            console.print(f"{mark} {r['q']}  (refusal via {r['how']}, retrieved {r['retrieved']})")

    console.print(f"\n[bold]k={s['k']}  MAX_DISTANCE={s['max_distance']}[/bold]")
    if "retrieval" in s:
        m = s["retrieval"]
        console.print(
            f"Retrieval ({m['n']}): hit@{k} {m[f'hit@{k}']:.0%} | top1 {m['top1']:.0%} | "
            f"MRR {m['mrr']:.2f} | video_hit {m['video_hit']:.0%} | cut by cutoff: {m['cut_by_cutoff']}"
        )
    if "refusal" in s:
        m = s["refusal"]
        console.print(f"Refusal   ({m['n']}): {m['rate']:.0%}")