"""Score each 10-K section with the OpenAI API.

Commands:
  pilot           direct calls on --n random sections (--dry-run prints the first request)
  direct          direct calls on all pending sections
  batch-prepare   write JSONL files for the Batch API
  batch-submit    upload and create the prepared batches
  batch-collect   download completed batches and store the scores
  batch-run       submit and collect until nothing is left

A section is never scored twice with the same model and prompt version, and identical texts
(same hash) are sent once. Requires OPENAI_API_KEY.
"""
import argparse
import json
import logging
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import tiktoken

import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("score")
enc = tiktoken.get_encoding(C.ENCODING)
MAX_REQUESTS = 50_000
MAX_BYTES = 150 * 1024 * 1024
MAX_ERRORS = 3
ACTIVE = ("validating", "in_progress", "finalizing", "cancelling", "submitted")
FINAL = {"completed": "completed", "failed": "failed", "expired": "expired", "cancelled": "cancelled"}

EXTRA = """
CREATE TABLE IF NOT EXISTS queue (
    accession_number TEXT, item TEXT, model TEXT, prompt_version TEXT, batch TEXT, custom_id TEXT,
    PRIMARY KEY (accession_number, item, model, prompt_version));
CREATE INDEX IF NOT EXISTS idx_queue_batch ON queue(batch);
CREATE TABLE IF NOT EXISTS errors (
    accession_number TEXT, item TEXT, model TEXT, prompt_version TEXT, batch TEXT, error TEXT, at TEXT);
"""

# cp1252 punctuation mis-decoded in older filings
CP1252 = {0x91: "'", 0x92: "'", 0x93: '"', 0x94: '"', 0x95: "-", 0x96: "-", 0x97: "-", 0x85: "...", 0xA0: " "}


def now():
    return datetime.now().isoformat(timespec="seconds")


def connect(db):
    if not os.path.exists(db):
        sys.exit(f"{db} not found; run build_db.py first")
    con = sqlite3.connect(db, timeout=120)
    con.row_factory = sqlite3.Row
    con.executescript(EXTRA)
    return con


def pending(con, a):
    """Sections without a score or queued request; one row per (accession, item), smallest CIK."""
    items = [i.strip() for i in a.items.split(",") if i.strip()]
    sql = f"""
    SELECT t.accession_number, t.item, t.n_tokens, t.text_hash, f.cik, f.company_name,
           f.sic, f.sic_description, f.sector, f.filing_date, f.year
    FROM texts t
    JOIN (SELECT accession_number, MIN(cik) AS cik FROM filings GROUP BY accession_number) fa
         ON fa.accession_number = t.accession_number
    JOIN filings f ON f.cik = fa.cik AND f.accession_number = t.accession_number
    LEFT JOIN scores s ON s.accession_number = t.accession_number AND s.item = t.item
                      AND s.model = :model AND s.prompt_version = :pv
    LEFT JOIN queue q ON q.accession_number = t.accession_number AND q.item = t.item
                     AND q.model = :model AND q.prompt_version = :pv
    WHERE s.accession_number IS NULL AND q.accession_number IS NULL
      AND f.year BETWEEN :y0 AND :y1
      AND t.item IN ({",".join(f"'{i}'" for i in items)})
      AND (SELECT COUNT(*) FROM errors e WHERE e.accession_number = t.accession_number
           AND e.item = t.item AND e.model = :model AND e.prompt_version = :pv) < :max_err
    ORDER BY f.year, t.accession_number, t.item
    """
    rows = [dict(r) for r in con.execute(sql, {"model": a.model, "pv": C.PROMPT_VERSION, "y0": a.year_from,
                                                  "y1": a.year_to, "max_err": MAX_ERRORS})]
    return rows[: a.limit] if a.limit else rows


def section_text(con, acc, item):
    return con.execute("SELECT text FROM texts WHERE accession_number=? AND item=?", (acc, item)).fetchone()[0]


def save_score(con, r, d, tok_in, tok_out, truncated, response_id, batch, snapshot, model):
    con.execute("""INSERT OR IGNORE INTO scores (accession_number, item, model, prompt_version, E, S, G,
                   E_reason, S_reason, G_reason, n_tokens_in, n_tokens_out, truncated, text_hash,
                   response_id, batch, scored_at, model_snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["accession_number"], r["item"], model, C.PROMPT_VERSION, d["E"], d["S"], d["G"],
                 d["E_reason"], d["S_reason"], d["G_reason"], tok_in, tok_out, int(bool(truncated)),
                 r["text_hash"], response_id, batch, now(), snapshot))


def save_error(con, r, batch, err, model):
    con.execute("INSERT INTO errors VALUES (?,?,?,?,?,?,?)",
                (r["accession_number"], r["item"], model, C.PROMPT_VERSION, batch, str(err)[:500], now()))


def copy_identical(con, rows, model):
    """Copy the score of an identical text (same hash) that is already scored."""
    left = []
    for r in rows:
        s = con.execute("SELECT * FROM scores WHERE text_hash=? AND model=? AND prompt_version=? LIMIT 1",
                        (r["text_hash"], model, C.PROMPT_VERSION)).fetchone()
        if s is None:
            left.append(r)
            continue
        save_score(con, r, dict(s), 0, 0, s["truncated"], s["response_id"], f"copy:{s['accession_number']}",
                   s["model_snapshot"], model)
    con.commit()
    return left


def prepare_text(text, n_tokens, max_tokens):
    text = text.translate(CP1252)
    if n_tokens <= max_tokens:
        return text, False
    return enc.decode(enc.encode_ordinary(text)[:max_tokens]), True


def build_messages(r, text, max_tokens):
    text, truncated = prepare_text(text, r["n_tokens"], max_tokens)
    user = C.user_prompt(r["company_name"], r["sic"], r["sic_description"], r["sector"],
                         r["filing_date"], r["item"], text, truncated)
    return [{"role": "system", "content": C.SYSTEM_PROMPT}, {"role": "user", "content": user}], truncated


def request_body(messages, model):
    return {"model": model, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": C.JSON_SCHEMA},
            "max_completion_tokens": C.MAX_COMPLETION_TOKENS,
            "reasoning_effort": C.REASONING_EFFORT}


def parse_response(body):
    choice = body["choices"][0]
    msg = choice["message"]
    if msg.get("refusal"):
        raise ValueError("refusal: " + msg["refusal"][:200])
    if choice.get("finish_reason") == "length":
        raise ValueError("output cut by max_completion_tokens")
    d = json.loads(msg["content"])
    for k in ("E", "S", "G"):
        d[k] = max(0, min(100, int(d[k])))
    u = body.get("usage", {})
    return d, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def by_hash(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["text_hash"]].append(r)
    return list(groups.values())


def client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    from openai import OpenAI
    return OpenAI(max_retries=6, timeout=180)


def cmd_direct(con, a, pilot=False):
    groups = by_hash(copy_identical(con, pending(con, a), a.model))
    if pilot:
        random.Random(a.seed).shuffle(groups)
        groups = groups[: a.n]
    if not groups:
        log.info("nothing pending")
        return
    if a.dry_run:
        r = groups[0][0]
        msgs, _ = build_messages(r, section_text(con, r["accession_number"], r["item"]), a.max_tokens)
        body = request_body(msgs, a.model)
        print(msgs[1]["content"][:1500] + "\n[...]")
        body["messages"][1]["content"] = "<section text>"
        print(json.dumps(body, indent=1, ensure_ascii=False)[:2500])
        return
    cl = client()
    done = failed = 0
    for i in range(0, len(groups), a.workers * 4):
        jobs = []
        for g in groups[i: i + a.workers * 4]:
            r = g[0]
            msgs, truncated = build_messages(r, section_text(con, r["accession_number"], r["item"]), a.max_tokens)
            jobs.append((g, truncated, request_body(msgs, a.model)))
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(lambda b: cl.chat.completions.create(**b).model_dump(), body): (g, tr) for g, tr, body in jobs}
            for fut in as_completed(futs):
                g, truncated = futs[fut]
                try:
                    body = fut.result()
                    d, tin, tout = parse_response(body)
                except Exception as e:
                    failed += 1
                    for r in g:
                        save_error(con, r, "direct", e, a.model)
                    continue
                for k, r in enumerate(g):
                    save_score(con, r, d, tin if k == 0 else 0, tout if k == 0 else 0, truncated, body.get("id"),
                               "direct" if k == 0 else f"copy:{g[0]['accession_number']}", body.get("model"), a.model)
                done += 1
        con.commit()
        log.info(f"{done + failed}/{len(groups)} | errors {failed}")


def next_batch_name(con):
    last = con.execute("SELECT batch FROM batches ORDER BY batch DESC LIMIT 1").fetchone()
    return f"batch_{int(last['batch'].split('_')[1]) + 1 if last else 1:04d}"


def cmd_batch_prepare(con, a):
    os.makedirs(a.batch_dir, exist_ok=True)
    groups = by_hash(copy_identical(con, pending(con, a), a.model))
    if not groups:
        log.info("nothing pending")
        return
    fh, name, path, n_req, n_bytes = None, None, None, 0, 0

    def close():
        fh.close()
        con.execute("UPDATE batches SET status='prepared', n_requests=?, jsonl_path=?, created=? WHERE batch=?",
                    (n_req, path, now(), name))
        con.commit()
        log.info(f"{name}: {n_req} requests")

    for g in groups:
        if fh is not None and (n_req >= MAX_REQUESTS or n_bytes >= MAX_BYTES):
            close()
            fh = None
        if fh is None:
            name = next_batch_name(con)
            con.execute("INSERT INTO batches (batch, status) VALUES (?, 'writing')", (name,))
            path = os.path.join(a.batch_dir, name + ".jsonl")
            fh = open(path, "w", encoding="utf-8")
            n_req = n_bytes = 0
        r = g[0]
        msgs, truncated = build_messages(r, section_text(con, r["accession_number"], r["item"]), a.max_tokens)
        custom_id = f"{r['accession_number']}|{r['item']}|{int(truncated)}"
        line = json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                           "body": request_body(msgs, a.model)}, ensure_ascii=False) + "\n"
        fh.write(line)
        n_bytes += len(line.encode("utf-8"))
        n_req += 1
        con.executemany("INSERT OR IGNORE INTO queue VALUES (?,?,?,?,?,?)",
                        [(x["accession_number"], x["item"], a.model, C.PROMPT_VERSION, name, custom_id) for x in g])
    if fh is not None:
        close()


def cmd_batch_submit(con, a, cl=None):
    cl = cl or client()
    active = con.execute(f"SELECT COUNT(*) FROM batches WHERE status IN ({','.join('?' * len(ACTIVE))})", ACTIVE).fetchone()[0]
    for b in con.execute("SELECT * FROM batches WHERE status='prepared' ORDER BY batch").fetchall():
        if active >= a.max_active:
            break
        file_id = b["input_file_id"]
        if not file_id:
            with open(b["jsonl_path"], "rb") as fh:
                file_id = cl.files.create(file=fh, purpose="batch").id
            con.execute("UPDATE batches SET input_file_id=? WHERE batch=?", (file_id, b["batch"]))
            con.commit()
        try:
            job = cl.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h")
        except Exception as e:
            if any(w in str(e).lower() for w in ("enqueued", "token limit", "rate")):
                log.info(f"queue limit reached; {b['batch']} waits")
                break
            raise
        con.execute("UPDATE batches SET batch_id=?, status=?, updated=? WHERE batch=?",
                    (job.id, job.status or "submitted", now(), b["batch"]))
        con.commit()
        active += 1
        log.info(f"{b['batch']} submitted as {job.id}")


def ingest(con, cl, batch, file_id, model):
    ok = err = 0
    for line in cl.files.content(file_id).text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id", "")
        truncated = int((cid.split("|") + ["0"])[2])
        rows = [dict(r) for r in con.execute(
            "SELECT q.accession_number, q.item, t.text_hash FROM queue q JOIN texts t "
            "ON t.accession_number=q.accession_number AND t.item=q.item WHERE q.custom_id=? AND q.batch=?", (cid, batch))]
        if not rows:
            continue
        resp = rec.get("response") or {}
        try:
            if rec.get("error") or resp.get("status_code") != 200:
                raise ValueError(json.dumps(rec.get("error") or resp.get("body", {}).get("error"))[:300])
            d, tin, tout = parse_response(resp["body"])
        except Exception as e:
            err += 1
            for r in rows:
                save_error(con, r, batch, e, model)
            continue
        body = resp.get("body", {})
        for k, r in enumerate(rows):
            save_score(con, r, d, tin if k == 0 else 0, tout if k == 0 else 0, truncated, body.get("id"),
                       batch if k == 0 else f"copy:{rows[0]['accession_number']}", body.get("model"), model)
        ok += 1
    con.commit()
    return ok, err


def cmd_batch_collect(con, a, cl=None):
    cl = cl or client()
    for b in con.execute("SELECT * FROM batches WHERE batch_id IS NOT NULL AND status NOT IN "
                         "('completed','failed','expired','cancelled')").fetchall():
        job = cl.batches.retrieve(b["batch_id"])
        status = job.status
        if job.status in FINAL:
            ok = err = 0
            if job.output_file_id:
                ok, err = ingest(con, cl, b["batch"], job.output_file_id, a.model)
            if job.error_file_id:
                err += ingest(con, cl, b["batch"], job.error_file_id, a.model)[1]
            con.execute("DELETE FROM queue WHERE batch=?", (b["batch"],))   # unscored rows become pending again
            status = FINAL[job.status]
            log.info(f"{b['batch']} {status}: {ok} scored, {err} errors")
        con.execute("UPDATE batches SET status=?, output_file_id=?, error_file_id=?, updated=? WHERE batch=?",
                    (status, job.output_file_id, job.error_file_id, now(), b["batch"]))
        con.commit()


def cmd_batch_run(con, a):
    cl = client()
    while True:
        try:
            cmd_batch_submit(con, a, cl)
            cmd_batch_collect(con, a, cl)
        except Exception as e:
            log.info(f"transient error ({type(e).__name__}); retrying")
            time.sleep(60)
            continue
        n_active = con.execute(f"SELECT COUNT(*) FROM batches WHERE status IN ({','.join('?' * len(ACTIVE))})",
                               ACTIVE).fetchone()[0]
        n_prep = con.execute("SELECT COUNT(*) FROM batches WHERE status='prepared'").fetchone()[0]
        if n_active == 0 and n_prep == 0:
            break
        time.sleep(a.interval)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["pilot", "direct", "batch-prepare", "batch-submit", "batch-collect", "batch-run"])
    p.add_argument("--db", default=C.DB_PATH)
    p.add_argument("--batch-dir", default=C.BATCH_DIR)
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--year-from", type=int, default=2011)
    p.add_argument("--year-to", type=int, default=2025)
    p.add_argument("--items", default=",".join(C.ITEMS))
    p.add_argument("--max-tokens", type=int, default=C.MAX_INPUT_TOKENS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-active", type=int, default=4)
    p.add_argument("--interval", type=int, default=600)
    a = p.parse_args()
    con = connect(a.db)
    {"pilot": lambda: cmd_direct(con, a, pilot=True), "direct": lambda: cmd_direct(con, a),
     "batch-prepare": lambda: cmd_batch_prepare(con, a), "batch-submit": lambda: cmd_batch_submit(con, a),
     "batch-collect": lambda: cmd_batch_collect(con, a), "batch-run": lambda: cmd_batch_run(con, a)}[a.cmd]()
    con.close()


if __name__ == "__main__":
    main()
