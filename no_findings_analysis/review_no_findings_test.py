#!/usr/bin/env python3
"""
Manual review UI for no-findings (NF) ground-truth labels on the test split.

Builds a stratified, randomly shuffled queue, then serves a local browser UI with
keyboard shortcuts (0 / 1 / x). Progress autosaves to no_findings_test_set_human_review.csv.

Usage
-----
  conda run -n PTCT3 python review_no_findings_test.py
  conda run -n PTCT3 python review_no_findings_test.py --port 8765 --n-random-neg 500
  conda run -n PTCT3 python review_no_findings_test.py --rebuild-queue --seed 123

Open http://127.0.0.1:8765/ in your browser. Keys: 0 = not NF, 1 = is NF, x = indeterminate,
u = undo last label.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime, timezone
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW_CSV = _DIR / "no_findings_review.csv"
DEFAULT_QUEUE_CSV = _DIR / "no_findings_test_review_queue.csv"
DEFAULT_OUTPUT_CSV = _DIR / "no_findings_test_set_human_review.csv"

OUTPUT_FIELDS = [
    "patient_id",
    "accession_number",
    "study_date",
    "source_row",
    "split",
    "conclusion",
    "word_count",
    "bert_sim",
    "best_ref",
    "nf_label",
    "human_nf",
    "bert_error",
    "review_note",
    "reviewed_at",
    "queue_index",
]

_ROW_KEY_FIELDS = ("source_row",)


def _row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    # Normalize so CSV resume keys always match queue keys.
    return (str(row.get("source_row", "")).strip(),)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_stratified_queue(
    test_rows: List[Dict[str, str]],
    *,
    n_random_neg: int,
    borderline_low: float,
    borderline_high: float,
    seed: int,
) -> List[Dict[str, str]]:
    """All BERT positives, all borderline sim band, random negatives; dedupe; shuffle."""
    positives = [r for r in test_rows if str(r.get("nf_label", "")).strip() == "1"]
    borderline = [
        r
        for r in test_rows
        if borderline_low <= _parse_float(str(r.get("bert_sim", ""))) < borderline_high
    ]
    negatives = [r for r in test_rows if str(r.get("nf_label", "")).strip() != "1"]

    rng = random.Random(seed)
    n_sample = min(n_random_neg, len(negatives))
    random_neg = rng.sample(negatives, n_sample) if n_sample else []

    seen: Set[Tuple[Any, ...]] = set()
    merged: List[Dict[str, str]] = []
    for bucket_name, bucket in (
        ("nf_label_1", positives),
        ("borderline_sim", borderline),
        ("random_neg", random_neg),
    ):
        for row in bucket:
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["_queue_bucket"] = bucket_name
            merged.append(out)

    rng.shuffle(merged)
    for i, row in enumerate(merged):
        row["queue_index"] = str(i)
    return merged


def load_or_build_queue(
    review_csv: Path,
    queue_csv: Path,
    *,
    rebuild: bool,
    n_random_neg: int,
    borderline_low: float,
    borderline_high: float,
    seed: int,
) -> List[Dict[str, str]]:
    if queue_csv.is_file() and not rebuild:
        rows = _read_csv(queue_csv)
        if rows:
            rows.sort(key=lambda r: int(r.get("queue_index", 0)))
            return rows

    all_rows = _read_csv(review_csv)
    test_rows = [r for r in all_rows if r.get("split") == "test"]
    if not test_rows:
        raise SystemExit(f"No test split rows in {review_csv}")

    queue = build_stratified_queue(
        test_rows,
        n_random_neg=n_random_neg,
        borderline_low=borderline_low,
        borderline_high=borderline_high,
        seed=seed,
    )
    _write_csv(queue_csv, queue, list(queue[0].keys()) if queue else OUTPUT_FIELDS)
    print(f"Built queue: {len(queue)} rows -> {queue_csv}")
    n_pos = sum(1 for r in queue if str(r.get("nf_label")) == "1")
    n_b = sum(1 for r in queue if _parse_float(str(r.get("bert_sim"))) >= borderline_low
              and _parse_float(str(r.get("bert_sim"))) < borderline_high)
    print(f"  nf_label=1: {n_pos}  borderline [{borderline_low},{borderline_high}): {n_b}")
    return queue


def _compute_bert_error(human_nf: str, nf_label: str) -> str:
    if human_nf not in ("0", "1"):
        return ""
    try:
        return "1" if int(human_nf) != int(nf_label) else "0"
    except ValueError:
        return ""


class ReviewSession:
    def __init__(self, queue: List[Dict[str, str]], output_csv: Path) -> None:
        self.queue = queue
        self.output_csv = output_csv
        self.reviews: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.history: List[Tuple[Any, ...]] = []
        self.position = 0
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.output_csv.is_file():
            self.position = 0
            return
        labeled: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        for row in _read_csv(self.output_csv):
            hn = str(row.get("human_nf", "")).strip().lower()
            if hn in ("", "nan"):
                continue
            key = _row_key(row)
            record = dict(row)
            self.reviews[key] = record
            labeled.append((str(record.get("reviewed_at", "")), key, record))
        labeled.sort(key=lambda item: item[0])
        self.history = [key for _, key, _ in labeled]
        self.position = self._first_unreviewed_index()

    @property
    def total(self) -> int:
        return len(self.queue)

    @property
    def reviewed_count(self) -> int:
        return len(self.reviews)

    def _index_for_key(self, key: Tuple[Any, ...]) -> int:
        for i, row in enumerate(self.queue):
            if _row_key(row) == key:
                return i
        return 0

    def _first_unreviewed_index(self) -> int:
        for i, row in enumerate(self.queue):
            if _row_key(row) not in self.reviews:
                return i
        return len(self.queue)

    def current_row(self) -> Optional[Dict[str, str]]:
        if self.position >= len(self.queue):
            return None
        return self.queue[self.position]

    def _public_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        key = _row_key(row)
        existing = self.reviews.get(key, {})
        return {
            "patient_id": row.get("patient_id", ""),
            "accession_number": row.get("accession_number", ""),
            "study_date": row.get("study_date", ""),
            "source_row": row.get("source_row", ""),
            "split": row.get("split", ""),
            "conclusion": row.get("conclusion", ""),
            "word_count": row.get("word_count", ""),
            "bert_sim": row.get("bert_sim", ""),
            "best_ref": row.get("best_ref", ""),
            "nf_label": row.get("nf_label", ""),
            "queue_index": row.get("queue_index", ""),
            "queue_bucket": row.get("_queue_bucket", row.get("queue_bucket", "")),
            "human_nf": existing.get("human_nf", ""),
            "review_note": existing.get("review_note", ""),
        }

    def state(self) -> Dict[str, Any]:
        idx = self.position
        cur = self.current_row()
        return {
            "queue_index": idx,
            "total": self.total,
            "reviewed": self.reviewed_count,
            "remaining": self.total - self.reviewed_count,
            "undo_available": len(self.history) > 0,
            "done": cur is None,
            "current": self._public_row(cur) if cur else None,
        }

    def label(self, human_nf: str, note: str = "") -> Dict[str, Any]:
        human_nf = str(human_nf).strip().lower()
        if human_nf not in ("0", "1", "x"):
            raise ValueError("human_nf must be 0, 1, or x")

        row = self.current_row()
        if row is None:
            return self.state()

        key = _row_key(row)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nf_label = str(row.get("nf_label", "")).strip()
        record: Dict[str, Any] = {
            "patient_id": row.get("patient_id", ""),
            "accession_number": row.get("accession_number", ""),
            "study_date": row.get("study_date", ""),
            "source_row": row.get("source_row", ""),
            "split": row.get("split", ""),
            "conclusion": row.get("conclusion", ""),
            "word_count": row.get("word_count", ""),
            "bert_sim": row.get("bert_sim", ""),
            "best_ref": row.get("best_ref", ""),
            "nf_label": nf_label,
            "human_nf": human_nf,
            "bert_error": _compute_bert_error(human_nf, nf_label),
            "review_note": note.strip(),
            "reviewed_at": now,
            "queue_index": row.get("queue_index", ""),
        }
        self.reviews[key] = record
        if not self.history or self.history[-1] != key:
            self.history.append(key)
        self.position = self._first_unreviewed_index()
        self._flush()
        return self.state()

    def undo(self) -> Dict[str, Any]:
        if not self.history:
            return self.state()
        key = self.history.pop()
        self.reviews.pop(key, None)
        self.position = self._index_for_key(key)
        self._flush()
        return self.state()

    def _flush(self) -> None:
        ordered: List[Dict[str, Any]] = []
        for row in self.queue:
            key = _row_key(row)
            if key in self.reviews:
                ordered.append(self.reviews[key])
        _write_csv(self.output_csv, ordered, OUTPUT_FIELDS)


def _highlight_conclusion(text: str) -> str:
    """Highlight no / fdg / avid for quick scanning (HTML-safe)."""
    import html

    escaped = html.escape(text)
    patterns = [
        (r"\bno\b", "hl-no"),
        (r"\bfdg\b", "hl-fdg"),
        (r"\bavid\b", "hl-avid"),
    ]
    for pat, cls in patterns:
        escaped = re.sub(
            pat,
            lambda m: f'<mark class="{cls}">{m.group(0)}</mark>',
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>No-findings test review</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #5b9fd4;
      --nf-yes: #3d8f6e;
      --nf-no: #c45c5c;
      --indet: #c9a227;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    .wrap { max-width: 960px; margin: 0 auto; padding: 1.25rem 1.5rem 3rem; }
    .top { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; }
    .progress { color: var(--muted); font-size: 0.95rem; }
    .progress strong { color: var(--text); }
    .meta { font-size: 0.85rem; color: var(--muted); margin-top: 0.25rem; }
    .bert-banner {
      margin: 1rem 0;
      padding: 0.75rem 1rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 1.05rem;
    }
    .bert-banner.nf1 { background: rgba(61, 143, 110, 0.2); border-left: 4px solid var(--nf-yes); }
    .bert-banner.nf0 { background: rgba(196, 92, 92, 0.15); border-left: 4px solid var(--nf-no); }
    .ref-box {
      background: var(--panel);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      margin-bottom: 1rem;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .ref-box span { color: var(--accent); }
    .conclusion {
      background: var(--panel);
      border-radius: 10px;
      padding: 1.25rem 1.35rem;
      font-size: 1.2rem;
      line-height: 1.55;
      min-height: 6rem;
    }
    mark.hl-no { background: #5b8def44; color: #a8c4ff; }
    mark.hl-fdg { background: #e8a83844; color: #ffd89a; }
    mark.hl-avid { background: #3d8f6e44; color: #9fdfc4; }
    .actions {
      display: flex;
      gap: 0.75rem;
      margin-top: 1.5rem;
      flex-wrap: wrap;
    }
    button {
      font-size: 1rem;
      padding: 0.65rem 1.25rem;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    button.k0 { background: #4a5568; color: #fff; }
    button.k1 { background: var(--nf-yes); color: #fff; }
    button.kx { background: #6b5a2a; color: #fff; }
    button.ku { background: transparent; color: var(--muted); border: 1px solid #444; }
    button:disabled { opacity: 0.35; cursor: not-allowed; }
    button:not(:disabled):hover { filter: brightness(1.08); }
    .hint {
      margin-top: 1rem;
      font-size: 0.8rem;
      color: var(--muted);
    }
    kbd {
      background: #2a3544;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      font-family: inherit;
    }
    .done {
      text-align: center;
      padding: 3rem 1rem;
      font-size: 1.25rem;
    }
    .done a { color: var(--accent); }
    .flash {
      position: fixed;
      bottom: 1rem;
      right: 1rem;
      background: #2a3544;
      padding: 0.5rem 0.85rem;
      border-radius: 6px;
      opacity: 0;
      transition: opacity 0.2s;
      pointer-events: none;
    }
    .flash.show { opacity: 1; }
    #focus-trap { position: absolute; width: 0; height: 0; opacity: 0; }
  </style>
</head>
<body>
  <input id="focus-trap" tabindex="0" aria-hidden="true" />
  <div id="app" class="wrap">
    <p>Loading…</p>
  </div>
  <div id="flash" class="flash"></div>
  <script>
    const app = document.getElementById('app');
    const flash = document.getElementById('flash');
    let state = null;
    let busy = false;

    function showFlash(msg) {
      flash.textContent = msg;
      flash.classList.add('show');
      setTimeout(() => flash.classList.remove('show'), 600);
    }

    async function api(path, body) {
      const usePost = path === '/api/label' || path === '/api/undo';
      const opts = usePost ? {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body ?? {}),
      } : {};
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }

    function nfLabelText(nf) {
      return nf === '1'
        ? 'BERT: NO FINDINGS (nf_label=1)'
        : 'BERT: Findings (nf_label=0)';
    }

    function render() {
      if (!state) return;
      if (state.done) {
        app.innerHTML = `
          <div class="done">
            <p>Queue complete — ${state.reviewed} / ${state.total} labeled.</p>
            <p>Saved to <code>no_findings_test_set_human_review.csv</code></p>
            <p><a href="/">Restart</a> after adding more rows to the queue.</p>
          </div>`;
        return;
      }
      const c = state.current;
      const bannerClass = c.nf_label === '1' ? 'nf1' : 'nf0';
      const sim = parseFloat(c.bert_sim).toFixed(4);
      app.innerHTML = `
        <div class="top">
          <div class="progress">
            <strong>${state.reviewed + 1}</strong> of ${state.total}
            · ${state.remaining} left
            · queue #${c.queue_index}
          </div>
          <div class="meta">${escapeHtml(c.accession_number)} · ${escapeHtml(c.study_date)} · row ${escapeHtml(String(c.source_row))}</div>
        </div>
        <div class="bert-banner ${bannerClass}">${nfLabelText(c.nf_label)} · sim ${sim}</div>
        <div class="ref-box">Nearest ref: <span>${escapeHtml(c.best_ref)}</span>
          ${c.queue_bucket ? ' · bucket: ' + escapeHtml(c.queue_bucket) : ''}</div>
        <div class="conclusion" id="conclusion-html"></div>
        <div class="actions">
          <button type="button" class="k0" data-nf="0">0 — Not NF</button>
          <button type="button" class="k1" data-nf="1">1 — Is NF</button>
          <button type="button" class="kx" data-nf="x">x — Indeterminate</button>
          <button type="button" class="ku" id="btn-undo" ${state.undo_available ? '' : 'disabled'}>Undo (u)</button>
        </div>
        <div class="hint">
          Keys: <kbd>0</kbd> not NF · <kbd>1</kbd> is NF · <kbd>x</kbd> indeterminate · <kbd>u</kbd> undo
        </div>`;
      const conclusionEl = app.querySelector('#conclusion-html');
      if (conclusionEl) {
        conclusionEl.innerHTML = c.conclusion_html || escapeHtml(c.conclusion);
      }
      app.querySelectorAll('button[data-nf]').forEach(btn => {
        btn.onclick = () => submit(btn.dataset.nf);
      });
      const undoBtn = app.querySelector('#btn-undo');
      if (undoBtn) undoBtn.onclick = () => undo();
      document.getElementById('focus-trap').focus();
    }

    function escapeHtml(s) {
      const el = document.createElement('div');
      el.textContent = s;
      return el.innerHTML;
    }

    async function refresh() {
      state = await api('/api/state');
      render();
    }

    async function submit(human_nf) {
      if (busy || !state || state.done) return;
      busy = true;
      try {
        state = await api('/api/label', { human_nf });
        showFlash('Saved: ' + human_nf);
        render();
      } catch (err) {
        showFlash('Error: ' + err.message);
      } finally {
        busy = false;
      }
    }

    async function undo() {
      if (busy || !state || !state.undo_available) return;
      busy = true;
      try {
        state = await api('/api/undo', {});
        const acc = state.current ? state.current.accession_number : '';
        showFlash('Undone' + (acc ? ' — ' + acc : ''));
        render();
      } catch (err) {
        showFlash('Undo failed: ' + err.message);
      } finally {
        busy = false;
      }
    }

    document.addEventListener('keydown', (e) => {
      if (busy || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      const k = e.key.toLowerCase();
      if (k === '0') { e.preventDefault(); submit('0'); }
      else if (k === '1') { e.preventDefault(); submit('1'); }
      else if (k === 'x') { e.preventDefault(); submit('x'); }
      else if (k === 'u') { e.preventDefault(); undo(); }
    });

    refresh().catch(err => { app.innerHTML = '<p>Error: ' + err + '</p>'; });
  </script>
</body>
</html>

"""


def make_handler(session: ReviewSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _no_cache(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")

        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._no_cache()
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._no_cache()
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_html(HTML_PAGE)
                return
            if path == "/api/state":
                st = session.state()
                if st.get("current"):
                    st["current"]["conclusion_html"] = _highlight_conclusion(
                        st["current"]["conclusion"]
                    )
                self._send_json(200, st)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return

            try:
                if path == "/api/label":
                    st = session.label(
                        human_nf=str(data.get("human_nf", "")),
                        note=str(data.get("note", "")),
                    )
                elif path == "/api/undo":
                    st = session.undo()
                else:
                    self.send_error(404)
                    return
                if st.get("current"):
                    st["current"]["conclusion_html"] = _highlight_conclusion(
                        st["current"]["conclusion"]
                    )
                self._send_json(200, st)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="No-findings test set manual review UI")
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--queue-csv", type=Path, default=DEFAULT_QUEUE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--rebuild-queue", action="store_true", help="Regenerate stratified queue CSV")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for queue")
    parser.add_argument("--n-random-neg", type=int, default=400, help="Random nf_label=0 test samples")
    parser.add_argument("--borderline-low", type=float, default=0.75)
    parser.add_argument("--borderline-high", type=float, default=0.90)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    queue = load_or_build_queue(
        args.review_csv,
        args.queue_csv,
        rebuild=args.rebuild_queue,
        n_random_neg=args.n_random_neg,
        borderline_low=args.borderline_low,
        borderline_high=args.borderline_high,
        seed=args.seed,
    )
    session = ReviewSession(queue, args.output_csv)
    done = session.reviewed_count
    resume_idx = session.position
    print(f"Queue: {len(queue)} rows | Already reviewed: {done}")
    if done < len(queue):
        print(f"Resuming at queue position {resume_idx + 1} (skip {done} labeled)")
    else:
        print("All queue rows already labeled.")
    print(f"Open http://{args.host}:{args.port}/  (hard-refresh if the page looks wrong)")
    print(f"Output: {args.output_csv}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if probe.connect_ex((args.host, args.port)) == 0:
            raise SystemExit(
                f"Port {args.port} is already in use — stop the other review server first "
                f"(e.g. pkill -f review_no_findings_test) or use --port {args.port + 1}"
            )

    server = HTTPServer((args.host, args.port), make_handler(session))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
