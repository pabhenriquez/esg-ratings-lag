"""Download the period of report (fiscal year end) of each 10-K from the SEC submissions API.

Writes cik, accession_number, filing_date, report_date and form for every 10-K filed since --since
by the firms in data/ai_scores_*.csv.gz. The SEC asks for a User-Agent with a contact email.

    SEC_USER_AGENT="Name email@domain" python3 report_dates.py --out report_dates.csv
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("SEC_CACHE", os.path.expanduser("~/sec_submissions_cache")))
FORMS = {"10-K", "10-K405", "10-KT"}
PAUSE = 0.16   # below the SEC limit of 10 requests per second


def firms(data_dir):
    ciks = set()
    for path in sorted(Path(data_dir).glob("ai_scores_*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            ciks |= {int(r["cik"]) for r in csv.DictReader(fh)}
    return sorted(ciks)


def fetch(url, ua):
    cached = CACHE / url.rsplit("/", 1)[-1]
    if cached.exists():
        return json.loads(cached.read_text())
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            cached.write_bytes(data)
            time.sleep(PAUSE)
            return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(5 * (attempt + 1) * (3 if e.code == 429 else 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def rows(cik, block, since):
    for acc, fd, rd, form in zip(block["accessionNumber"], block["filingDate"], block["reportDate"], block["form"]):
        if form in FORMS and fd >= since:
            yield [cik, acc, fd, rd, form]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=ROOT / "data")
    p.add_argument("--since", default="2011-01-01")
    p.add_argument("--out", default="report_dates.csv")
    a = p.parse_args()
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit("set SEC_USER_AGENT to a name and contact email")
    CACHE.mkdir(exist_ok=True)
    ciks = firms(a.data)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cik", "accession_number", "filing_date", "report_date", "form"])
        for i, cik in enumerate(ciks, 1):
            j = fetch(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", ua)
            if j is None:
                continue
            recent = j["filings"]["recent"]
            w.writerows(rows(cik, recent, a.since))
            oldest = recent["filingDate"][-1] if recent["filingDate"] else "9999"
            if oldest > a.since:
                for extra in j["filings"].get("files", []):
                    if extra["filingTo"] >= a.since:
                        k = fetch("https://data.sec.gov/submissions/" + extra["name"], ua)
                        if k:
                            w.writerows(rows(cik, k, a.since))
            if i % 250 == 0:
                print(f"{i}/{len(ciks)} firms", flush=True)


if __name__ == "__main__":
    main()
