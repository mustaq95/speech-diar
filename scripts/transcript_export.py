"""Export the stored read-aloud transcripts as one self-contained HTML document.

For embedding in a PDF evaluation report, so: light palette, no external assets, no
JavaScript, and page breaks placed so a recording is never split across pages.

Everything here is read from the API and the stored rows. The one thing computed at
export time is the word-level alignment used to mark each error, and it is computed
with `packages.metrics.score` — the same function that produced the stored rates — so
the marks and the totals can never disagree.

    uv run python scripts/transcript_export.py            # every recorded, scored row
    uv run python scripts/transcript_export.py 147 148    # specific recordings
    uv run python scripts/transcript_export.py -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.metrics.wer import score  # noqa: E402

API = "http://localhost:8010"

LANGUAGE_LABELS = {
    "ar": "Arabic",
    "en": "English",
    "mixed-50-50": "Mixed Arabic / English",
}

# Transport is part of the measurement, not a footnote: a chunked engine is cut at a
# fixed interval and a word split across a boundary counts against it, while a stream
# has no boundaries. The two rates describe two pipelines.
TRANSPORT_LABELS = {
    "stream": "streaming (server-side VAD)",
    "chunks": "fixed-interval chunks",
}


def fetch(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=20) as response:
        return json.load(response)


def collect(ids: list[int] | None) -> list[dict]:
    rows = fetch("/recordings?surface=transcript")
    by_id = {row["audioFileId"]: row for row in rows}
    if ids:
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"No transcript recording with id(s): {missing}")
        wanted = [by_id[i] for i in ids]
    else:
        # Only rows that actually have something to show: audio recorded AND scored.
        # A generated-but-unread script has no transcript to export.
        wanted = [row for row in rows if row["hasAudio"] and row["scored"]]

    out = []
    for row in wanted:
        try:
            reference = fetch(f"/evaluations/{row['audioFileId']}/reference")
        except urllib.error.HTTPError:
            continue  # no ground truth, so nothing to compare against
        runs = fetch(f"/evaluations/{row['audioFileId']}/transcript")
        runs = [run for run in runs if run.get("text")]
        if runs:
            out.append({"row": row, "reference": reference, "runs": runs})
    return out


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def clock(seconds) -> str:
    if not seconds:
        return "—"
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def marked_hypothesis(reference: str, hypothesis: str) -> str:
    """The engine's transcript with substituted and inserted words marked.

    Marks are placed from the SAME alignment the score comes from, so a reader
    counting the highlights gets the substitution and insertion counts in the table.
    Deletions are shown separately below, since a dropped word has no position in the
    output to highlight.

    The original (un-normalized) tokens are rendered: normalization exists to make the
    comparison fair, not to change what the engine actually said.
    """
    result = score(reference, hypothesis)
    raw_tokens = hypothesis.split()
    marks: dict[int, str] = {}
    for step in result.alignment:
        if step.hyp_index is None:
            continue
        if step.op == "sub":
            marks[step.hyp_index] = "sub"
        elif step.op == "ins":
            marks[step.hyp_index] = "ins"

    pieces = []
    for index, token in enumerate(raw_tokens):
        mark = marks.get(index)
        pieces.append(f'<em class="{mark}">{esc(token)}</em>' if mark else esc(token))
    dropped = [step.ref for step in result.alignment if step.op == "del" and step.ref]
    return " ".join(pieces), dropped


def summary_rows(entries: list[dict]) -> tuple[list[str], dict[str, dict]]:
    """Per-engine totals over the whole set.

    Aggregate rates are total errors over total reference words, never the mean of
    per-recording rates: one short clip would otherwise swing the headline as hard as
    a long one.
    """
    engines: dict[str, dict] = {}
    order: list[str] = []
    for entry in entries:
        for run in entry["runs"]:
            metrics = run.get("metrics") or {}
            if not metrics:
                continue
            asr_id = run["asrId"]
            if asr_id not in engines:
                order.append(asr_id)
                engines[asr_id] = {
                    "name": run.get("asrName") or asr_id,
                    "transport": run.get("transport") or "",
                    "sub": 0, "del": 0, "ins": 0, "ref": 0, "hyp": 0,
                    "cer_weighted": 0.0, "latencies": [],
                }
            bucket = engines[asr_id]
            bucket["sub"] += metrics.get("subCount") or 0
            bucket["del"] += metrics.get("delCount") or 0
            bucket["ins"] += metrics.get("insCount") or 0
            bucket["ref"] += metrics.get("refWordCount") or 0
            bucket["hyp"] += metrics.get("hypWordCount") or 0
            bucket["cer_weighted"] += (metrics.get("cer") or 0.0) * (metrics.get("refWordCount") or 0)
            if run.get("avgLatencyMs") is not None:
                bucket["latencies"].append(run["avgLatencyMs"])
    return order, engines


def build(entries: list[dict]) -> str:
    order, engines = summary_rows(entries)
    total_ref = max((bucket["ref"] for bucket in engines.values()), default=0)
    total_audio = sum(entry["row"]["durationSec"] for entry in entries)
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    parts: list[str] = []
    add = parts.append

    add(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Read-aloud transcript evaluation</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#6b6b6b; --line:#e4e1d9; --paper:#fff; --bg:#f7f6f3;
           --sub:#b4342a; --ins:#8a5a00; --del:#6b6b6b; --good:#0a7c42; }}
  * {{ box-sizing:border-box; }}
  body {{ font:13.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--ink); margin:0; padding:32px; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:23px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:28px 0 8px; }}
  h3 {{ font-size:14px; margin:0; }}
  .sub {{ color:var(--muted); font-size:12px; }}
  .cards {{ display:flex; gap:10px; margin:18px 0; flex-wrap:wrap; }}
  .card {{ flex:1 1 150px; background:var(--paper); border:1px solid var(--line); padding:10px 12px; }}
  .card b {{ display:block; font-size:19px; margin-top:2px; }}
  table {{ width:100%; border-collapse:collapse; margin:8px 0 4px; background:var(--paper);
           border:1px solid var(--line); }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); }}
  td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.win {{ font-weight:700; color:var(--good); }}
  .note {{ background:var(--paper); border:1px solid var(--line); padding:12px 14px;
           margin:14px 0; font-size:12.5px; }}
  .note b {{ display:block; margin-bottom:3px; }}
  .rec {{ background:var(--paper); border:1px solid var(--line); margin:14px 0; padding:14px 16px;
          break-inside:avoid; page-break-inside:avoid; }}
  .rec-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:14px;
               border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:10px; }}
  .tag {{ font-size:10.5px; border:1px solid var(--line); padding:2px 7px; color:var(--muted); }}
  .block {{ margin:12px 0 0; }}
  .label {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted);
            display:flex; justify-content:space-between; gap:12px; margin-bottom:5px; }}
  /* Justified: a script that starts in Arabic resolves to RTL, and a ragged edge on
     mixed-language lines reads as broken layout rather than as text. */
  .text {{ font-size:13.5px; line-height:1.8; text-align:justify; margin:0;
           padding:9px 11px; background:#fbfaf7; border:1px solid var(--line); }}
  .text.ref {{ background:#f4f7f4; }}
  em {{ font-style:normal; padding:0 1px; }}
  em.sub {{ color:var(--sub); border-bottom:2px solid var(--sub); }}
  em.ins {{ color:var(--ins); border-bottom:2px dotted var(--ins); }}
  .dropped {{ font-size:12px; color:var(--muted); margin:5px 0 0; }}
  .dropped s {{ color:var(--del); }}
  .key {{ font-size:11.5px; color:var(--muted); margin:6px 0 0; }}
  .key em {{ margin-right:2px; }}
  footer {{ margin-top:28px; padding-top:10px; border-top:1px solid var(--line);
            color:var(--muted); font-size:11px; }}
  @media print {{
    body {{ background:#fff; padding:0; font-size:11pt; }}
    .rec, .card, table, .note {{ border-color:#ccc; }}
    h2 {{ page-break-after:avoid; }}
  }}
</style>
</head><body><main class="wrap">

<h1>Read-aloud transcript evaluation</h1>
<p class="sub">{len(entries)} recordings &middot; {clock(total_audio)} of audio &middot;
{total_ref} reference words &middot; generated {esc(generated)}</p>

<div class="cards">
  <div class="card"><span class="sub">Recordings</span><b>{len(entries)}</b></div>
  <div class="card"><span class="sub">Audio scored</span><b>{clock(total_audio)}</b></div>
  <div class="card"><span class="sub">Reference words</span><b>{total_ref}</b></div>
  <div class="card"><span class="sub">Engines</span><b>{len(order)}</b></div>
</div>

<div class="note"><b>How to read this.</b>
Every recording was read aloud from a script that was written first, so these are true
error rates against known ground truth rather than agreement between models. Word Error
Rate is substitutions plus deletions plus insertions over the number of words in the
script; lower is better, and it can exceed 100% when an engine emits more wrong words
than the script contains. Set-level rates are total errors over total reference words,
not an average of per-recording rates. Each engine ran on its own native transport and
every rate includes the cost of that transport, so these are measurements of two
<em style="font-style:italic">pipelines</em>, not of two models in the abstract.
</div>
""")

    # --- per-engine summary --------------------------------------------------
    add("<h2>Overall, by engine</h2>\n<table><thead><tr>"
        "<th>Engine</th><th>Transport</th><th class='num'>WER</th><th class='num'>CER</th>"
        "<th class='num'>Sub / Del / Ins</th><th class='num'>Words ref &rarr; out</th>"
        "<th class='num'>Avg latency</th></tr></thead><tbody>")
    best = min(
        ((asr_id, (b["sub"] + b["del"] + b["ins"]) / b["ref"]) for asr_id, b in engines.items() if b["ref"]),
        key=lambda pair: pair[1], default=(None, None),
    )[0]
    for asr_id in order:
        bucket = engines[asr_id]
        wer = (bucket["sub"] + bucket["del"] + bucket["ins"]) / bucket["ref"] if bucket["ref"] else None
        cer = bucket["cer_weighted"] / bucket["ref"] if bucket["ref"] else None
        latency = (
            f"{round(sum(bucket['latencies']) / len(bucket['latencies']))} ms"
            if bucket["latencies"] else "—"
        )
        add(f"<tr><td><strong>{esc(bucket['name'])}</strong></td>"
            f"<td class='sub'>{esc(TRANSPORT_LABELS.get(bucket['transport'], bucket['transport']))}</td>"
            f"<td class='num{' win' if asr_id == best else ''}'>{pct(wer)}</td>"
            f"<td class='num'>{pct(cer)}</td>"
            f"<td class='num'>{bucket['sub']} / {bucket['del']} / {bucket['ins']}</td>"
            f"<td class='num'>{bucket['ref']} &rarr; {bucket['hyp']}</td>"
            f"<td class='num'>{latency}</td></tr>")
    add("</tbody></table>")

    # --- by language of the script ------------------------------------------
    # Grouped on the mix that was REQUESTED. The generator follows a mix instruction
    # loosely, so each recording also carries the share actually measured in its own
    # section below; the two genuinely differ and both are worth having.
    by_language: dict[str, dict[str, dict]] = {}
    language_order: list[str] = []
    for entry in entries:
        mix = (entry["reference"].get("params") or {}).get("languageMix") or "unspecified"
        if mix not in by_language:
            by_language[mix] = {}
            language_order.append(mix)
        for run in entry["runs"]:
            metrics = run.get("metrics") or {}
            if not metrics:
                continue
            bucket = by_language[mix].setdefault(run["asrId"], {"errors": 0, "ref": 0})
            bucket["errors"] += (
                (metrics.get("subCount") or 0) + (metrics.get("delCount") or 0)
                + (metrics.get("insCount") or 0)
            )
            bucket["ref"] += metrics.get("refWordCount") or 0

    add("<h2>By language of script</h2>\n<table><thead><tr><th>Language</th>"
        "<th class='num'>Recordings</th><th class='num'>Reference words</th>")
    for asr_id in order:
        add(f"<th class='num'>{esc(engines[asr_id]['name'])}</th>")
    add("</tr></thead><tbody>")
    for mix in language_order:
        buckets = by_language[mix]
        count = sum(
            1 for entry in entries
            if ((entry["reference"].get("params") or {}).get("languageMix") or "unspecified") == mix
        )
        words = max((b["ref"] for b in buckets.values()), default=0)
        rates = {
            asr_id: (b["errors"] / b["ref"] if b["ref"] else None) for asr_id, b in buckets.items()
        }
        lowest = min((r for r in rates.values() if r is not None), default=None)
        add(f"<tr><td>{esc(LANGUAGE_LABELS.get(mix, mix))}</td>"
            f"<td class='num'>{count}</td><td class='num'>{words}</td>")
        for asr_id in order:
            value = rates.get(asr_id)
            klass = "num win" if value is not None and value == lowest else "num"
            add(f"<td class='{klass}'>{pct(value)}</td>")
        add("</tr>")
    add("</tbody></table>")

    # --- per-recording summary ----------------------------------------------
    add("<h2>By recording</h2>\n<table><thead><tr><th>Recording</th><th>Language</th>"
        "<th class='num'>Length</th><th class='num'>Words</th>")
    for asr_id in order:
        add(f"<th class='num'>{esc(engines[asr_id]['name'])} WER</th>")
    add("</tr></thead><tbody>")
    for entry in entries:
        params = entry["reference"].get("params") or {}
        wers = {}
        for run in entry["runs"]:
            metrics = run.get("metrics") or {}
            wers[run["asrId"]] = metrics.get("wer")
        lowest = min((w for w in wers.values() if w is not None), default=None)
        ref_words = max(
            ((run.get("metrics") or {}).get("refWordCount") or 0 for run in entry["runs"]),
            default=entry["reference"]["wordCount"],
        )
        add(f"<tr><td>{esc(entry['row']['filename'])}</td>"
            f"<td>{esc(LANGUAGE_LABELS.get(params.get('languageMix'), params.get('languageMix') or '—'))}</td>"
            f"<td class='num'>{clock(entry['row']['durationSec'])}</td>"
            f"<td class='num'>{ref_words}</td>")
        for asr_id in order:
            value = wers.get(asr_id)
            klass = "num win" if value is not None and value == lowest else "num"
            add(f"<td class='{klass}'>{pct(value)}</td>")
        add("</tr>")
    add("</tbody></table>")

    # --- the transcripts themselves -----------------------------------------
    add("<h2>Transcripts</h2>")
    add('<p class="key">In each engine\'s output: '
        '<em class="sub">substituted</em> words were heard as something else, '
        '<em class="ins">inserted</em> words were not in the script, and dropped words are '
        'listed below the transcript. Marks come from the same alignment the rates are '
        'computed from.</p>')

    for entry in entries:
        row, reference = entry["row"], entry["reference"]
        params = reference.get("params") or {}
        split = params.get("languageSplit") or {}
        measured = (
            f" &middot; {round((split.get('arabic') or 0) * 100)}% AR / "
            f"{round((split.get('english') or 0) * 100)}% EN measured" if split else ""
        )
        add(f'<section class="rec">'
            f'<div class="rec-head"><h3>{esc(row["filename"])}</h3>'
            f'<span class="sub">{clock(row["durationSec"])} &middot; '
            f'{esc(LANGUAGE_LABELS.get(params.get("languageMix"), params.get("languageMix") or ""))}'
            f'{measured}</span></div>')

        hard = params.get("hardCases") or []
        meta = []
        if params.get("minutes"):
            meta.append(f'<span class="tag">target {esc(params["minutes"])} min</span>')
        for case in hard:
            meta.append(f'<span class="tag">{esc(case)}</span>')
        if params.get("generatorModel"):
            meta.append(f'<span class="tag">{esc(params["generatorModel"])}</span>')
        if meta:
            add(f'<div class="block">{" ".join(meta)}</div>')

        add(f'<div class="block"><div class="label"><span>Reference script '
            f'({esc(reference["source"])})</span><span>{reference["wordCount"]} words</span></div>'
            f'<p class="text ref" dir="auto">{esc(reference["text"])}</p></div>')

        for run in entry["runs"]:
            metrics = run.get("metrics") or {}
            marked, dropped = marked_hypothesis(reference["text"], run["text"])
            timing = []
            if run.get("firstLatencyMs") is not None:
                timing.append(f'first {run["firstLatencyMs"]} ms')
            if run.get("avgLatencyMs") is not None:
                timing.append(f'avg {run["avgLatencyMs"]} ms')
            if run.get("chunkCount"):
                unit = "chunks" if run.get("transport") == "chunks" else "segments"
                timing.append(f'{run["chunkCount"]} {unit}')
            add(f'<div class="block"><div class="label">'
                f'<span>{esc(run.get("asrName") or run["asrId"])} &middot; '
                f'{esc(TRANSPORT_LABELS.get(run.get("transport"), run.get("transport") or ""))}</span>'
                f'<span>WER {pct(metrics.get("wer"))} &middot; CER {pct(metrics.get("cer"))} &middot; '
                f'{metrics.get("subCount", 0)}S / {metrics.get("delCount", 0)}D / '
                f'{metrics.get("insCount", 0)}I'
                f'{" &middot; " + " &middot; ".join(timing) if timing else ""}</span></div>'
                f'<p class="text" dir="auto">{marked}</p>')
            if dropped:
                shown = ", ".join(esc(word) for word in dropped[:24])
                more = f" and {len(dropped) - 24} more" if len(dropped) > 24 else ""
                add(f'<p class="dropped">Dropped from the script: <s>{shown}</s>{more}</p>')
            add("</div>")
        add("</section>")

    add(f"""<footer>SPEECHDYN &middot; read-aloud transcript evaluation &middot;
every figure on this page is measured from stored output; nothing is estimated or
reconstructed. Error marks are computed with the same alignment that produced the
rates. Real-Time Factor is not reported for a streaming engine, which consumes audio
at 1&times; by definition.</footer>
</main></body></html>""")
    return "".join(parts)


def build_scripts_only(entries: list[dict]) -> str:
    """Just the scripts that were read aloud, with nothing measured attached.

    The ground truth on its own: what the speaker was given to read. No engine
    output, no error marks, no rates. Same paper-friendly styling as the full
    comparison so the two can sit in one PDF.
    """
    total_words = sum(entry["reference"]["wordCount"] for entry in entries)
    total_audio = sum(entry["row"]["durationSec"] for entry in entries)
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    parts: list[str] = []
    add = parts.append
    add(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Read-aloud scripts</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#6b6b6b; --line:#e4e1d9; --paper:#fff; --bg:#f7f6f3; }}
  * {{ box-sizing:border-box; }}
  body {{ font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--ink); margin:0; padding:36px; }}
  .wrap {{ max-width:860px; margin:0 auto; }}
  h1 {{ font-size:23px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:12px; }}
  .rec {{ background:var(--paper); border:1px solid var(--line); margin:16px 0;
          padding:18px 20px; break-inside:avoid; page-break-inside:avoid; }}
  .rec h2 {{ font-size:15px; margin:0 0 3px; }}
  .meta {{ color:var(--muted); font-size:11.5px; margin:0 0 12px;
           padding-bottom:10px; border-bottom:1px solid var(--line); }}
  /* Justified: a script that opens in Arabic resolves to RTL, and a ragged edge on
     mixed-language lines reads as broken layout rather than as text. */
  .text {{ margin:0; font-size:14.5px; line-height:1.9; text-align:justify; }}
  footer {{ margin-top:26px; padding-top:10px; border-top:1px solid var(--line);
            color:var(--muted); font-size:11px; }}
  @media print {{ body {{ background:#fff; padding:0; font-size:11.5pt; }}
                  .rec {{ border-color:#ccc; }} }}
</style>
</head><body><main class="wrap">
<h1>Read-aloud scripts</h1>
<p class="sub">{len(entries)} scripts &middot; {total_words} words &middot;
{clock(total_audio)} of audio recorded &middot; generated {esc(generated)}</p>
""")

    for entry in entries:
        row, reference = entry["row"], entry["reference"]
        params = reference.get("params") or {}
        split = params.get("languageSplit") or {}
        bits = [row.get("createdAt", "")[:10], clock(row["durationSec"])]
        if params.get("languageMix"):
            label = LANGUAGE_LABELS.get(params["languageMix"], params["languageMix"])
            if split:
                label += (f" ({round((split.get('arabic') or 0) * 100)}% AR / "
                          f"{round((split.get('english') or 0) * 100)}% EN measured)")
            bits.append(label)
        bits.append(f'{reference["wordCount"]} words')
        add(f'<section class="rec"><h2>{esc(row["filename"])}</h2>'
            f'<p class="meta">{" &middot; ".join(esc(bit) for bit in bits)}</p>'
            f'<p class="text" dir="auto">{esc(reference["text"])}</p></section>')

    add("""<footer>SPEECHDYN &middot; the reference scripts as they were read aloud,
reproduced verbatim from storage.</footer></main></body></html>""")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", type=int, help="recording ids; default is all scored ones")
    parser.add_argument("-o", "--out", default="transcript-evaluation.html")
    parser.add_argument(
        "--scripts-only", action="store_true",
        help="just the scripts that were read aloud, without the engine comparison",
    )
    args = parser.parse_args()

    entries = collect(args.ids or None)
    if not entries:
        raise SystemExit("No scored read-aloud recordings to export.")
    page = build_scripts_only(entries) if args.scripts_only else build(entries)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"{len(entries)} recordings -> {args.out}")


if __name__ == "__main__":
    main()
