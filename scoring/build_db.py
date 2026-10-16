"""Build the SQLite database with the Item 1 and Item 1A text of each 10-K.

Input: a CSV with one row per 10-K file and the columns cik, accession_number, filing_date, form,
company_name, sic, sic_description, sector, file, item_1 and item_1a (section text), plus a
manifest with one row per file (cik, accession_number) used to detect duplicated files.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool

import tiktoken

import config as C

csv.field_size_limit(sys.maxsize)
CHUNK = 1500

SCHEMA = """
PRAGMA journal_mode=OFF;
PRAGMA synchronous=OFF;
PRAGMA page_size=8192;
CREATE TABLE filings (
    cik TEXT NOT NULL, accession_number TEXT NOT NULL, filing_date TEXT, year INTEGER,
    fiscal_year_approx INTEGER, form TEXT, company_name TEXT, sic TEXT, sic_description TEXT,
    sector TEXT, file TEXT,
    PRIMARY KEY (cik, accession_number));
CREATE TABLE texts (
    accession_number TEXT NOT NULL, item TEXT NOT NULL, text TEXT, n_chars INTEGER,
    n_tokens INTEGER, text_hash TEXT,
    PRIMARY KEY (accession_number, item));
CREATE TABLE duplicates (cik TEXT, accession_number TEXT, file_dropped TEXT, file_kept TEXT);
CREATE TABLE scores (
    accession_number TEXT NOT NULL, item TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
    E INTEGER, S INTEGER, G INTEGER, E_reason TEXT, S_reason TEXT, G_reason TEXT,
    n_tokens_in INTEGER, n_tokens_out INTEGER, truncated INTEGER, text_hash TEXT,
    response_id TEXT, batch TEXT, scored_at TEXT, model_snapshot TEXT,
    PRIMARY KEY (accession_number, item, model, prompt_version));
CREATE TABLE batches (
    batch TEXT PRIMARY KEY, batch_id TEXT, input_file_id TEXT, output_file_id TEXT, error_file_id TEXT,
    status TEXT, n_requests INTEGER, jsonl_path TEXT, created TEXT, updated TEXT);
"""

_enc = None


def _init():
    global _enc
    _enc = tiktoken.get_encoding(C.ENCODING)


def _tokenize(args):
    acc, item, text = args
    return acc, item, len(_enc.encode_ordinary(text)), hashlib.sha1(text.encode("utf-8")).hexdigest()


def choose_copy(copies):
    """Same (cik, accession): keep the copy with more complete items, then the non-'cleaned' file."""
    def key(r):
        return (sum(len(r[it]) >= C.MIN_CHARS for it in C.ITEMS), 0 if r["file"].endswith("-cleaned.txt") else 1)
    order = sorted(copies, key=key, reverse=True)
    return order[0], order[1:]


def fiscal_year_approx(filing_date):
    # most 10-Ks are filed 60-90 days after the fiscal year end
    try:
        return (datetime.strptime(filing_date, "%Y-%m-%d") - timedelta(days=75)).year
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items-csv", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--db", default=C.DB_PATH)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max-rows", type=int, default=0)
    a = p.parse_args()
    if os.path.exists(a.db):
        sys.exit(f"{a.db} already exists")
    t0 = time.time()
    tmp = os.path.join(tempfile.gettempdir(), os.path.basename(a.db))
    if os.path.exists(tmp):
        os.remove(tmp)

    copies = Counter()
    with open(a.manifest, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            copies[(r["cik"], r["accession_number"])] += 1
    dup_keys = {k: n for k, n in copies.items() if n > 1}

    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    stats = Counter()
    waiting = defaultdict(list)
    hashes = {}
    pool = Pool(a.workers, initializer=_init)

    def process(chunk):
        rows, jobs, texts = [], [], {}
        for r in chunk:
            fd = r["filing_date"]
            rows.append((r["cik"], r["accession_number"], fd, int(fd[:4]), fiscal_year_approx(fd), r["form"],
                         r["company_name"], r["sic"] or None, r["sic_description"] or None, r["sector"] or None, r["file"]))
            for it in C.ITEMS:
                t = r[it]
                if len(t) < C.MIN_CHARS:
                    stats[f"{it}_missing_or_short"] += 1
                    continue
                key = (r["accession_number"], it)
                if key in hashes or key in texts:   # same accession under another CIK: text stored once
                    stats["shared_accession"] += 1
                    continue
                texts[key] = t
                jobs.append((r["accession_number"], it, t))
        res = pool.map(_tokenize, jobs, chunksize=4)
        con.executemany("INSERT OR IGNORE INTO filings VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.executemany("INSERT INTO texts VALUES (?,?,?,?,?,?)",
                        [(acc, it, texts[(acc, it)], len(texts[(acc, it)]), n, h) for acc, it, n, h in res])
        con.commit()
        for acc, it, n, h in res:
            hashes[(acc, it)] = h
            stats[f"{it}_stored"] += 1

    def resolve(k, cps):
        kept, dropped = choose_copy(cps)
        con.executemany("INSERT INTO duplicates VALUES (?,?,?,?)", [(k[0], k[1], d["file"], kept["file"]) for d in dropped])
        return kept

    chunk = []
    with open(a.items_csv, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if a.max_rows and i >= a.max_rows:
                break
            if not row["filing_date"]:
                continue
            k = (row["cik"], row["accession_number"])
            if k in dup_keys:
                waiting[k].append(row)
                if len(waiting[k]) < dup_keys[k]:
                    continue
                row = resolve(k, waiting.pop(k))
            chunk.append(row)
            if len(chunk) >= CHUNK:
                process(chunk)
                chunk = []
    for k, cps in list(waiting.items()):
        chunk.append(resolve(k, cps) if len(cps) > 1 else cps[0])
    if chunk:
        process(chunk)
    pool.close()
    pool.join()

    con.executescript("""
        CREATE INDEX idx_filings_acc ON filings(accession_number);
        CREATE INDEX idx_filings_year ON filings(year);
        CREATE INDEX idx_texts_hash ON texts(text_hash);
        CREATE INDEX idx_scores_hash ON scores(text_hash);
    """)
    con.close()
    shutil.move(tmp, a.db)
    print(f"{a.db} built in {time.time() - t0:.0f}s", dict(stats))


if __name__ == "__main__":
    main()
