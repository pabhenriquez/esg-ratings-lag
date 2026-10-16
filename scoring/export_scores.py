"""Export one row per scored 10-K section, without text, to data/ai_scores_<item>.csv.gz.

The fiscal year comes from the EDGAR period of report when available (report_dates.py);
otherwise it is the year of the filing date minus 75 days.
"""
import argparse
import csv
import os
import sqlite3

import pandas as pd

import config as C

SQL = """
SELECT f.cik, t.accession_number, f.company_name, f.form, f.filing_date, f.fiscal_year_approx,
       f.sic, f.sic_description AS industry, f.sector, t.item, t.n_chars, t.n_tokens,
       s.truncated, s.E, s.S, s.G, s.E_reason, s.S_reason, s.G_reason,
       s.model_snapshot AS model, s.prompt_version
FROM texts t
JOIN (SELECT accession_number, MIN(cik) AS cik FROM filings GROUP BY accession_number) fa
     ON fa.accession_number = t.accession_number
JOIN filings f ON f.cik = fa.cik AND f.accession_number = t.accession_number
JOIN scores s ON s.accession_number = t.accession_number AND s.item = t.item
             AND s.model = ? AND s.prompt_version = ?
WHERE f.year BETWEEN ? AND ?
ORDER BY f.cik, f.filing_date, t.item
"""
COLS = ["cik", "accession_number", "company_name", "form", "filing_date", "report_date", "fiscal_year_approx",
        "fiscal_year", "sic", "industry", "sector", "item", "n_chars", "n_tokens", "truncated", "E", "S", "G",
        "E_reason", "S_reason", "G_reason", "model", "prompt_version"]


def fiscal_year(report_date, approx):
    rd = pd.to_datetime(report_date, errors="coerce")
    fy = rd.dt.year - ((rd.dt.month == 1) & (rd.dt.day <= 7)).astype(int)   # 52/53-week years ending in early January
    return fy.where(rd.notna(), approx).astype(int)


def export(con, report_dates, out_dir, model=C.MODEL, year_from=2011, year_to=2025):
    d = pd.read_sql(SQL, con, params=(model, C.PROMPT_VERSION, year_from, year_to))
    rd = pd.read_csv(report_dates, dtype={"accession_number": str}).drop_duplicates("accession_number")
    d = d.merge(rd[["accession_number", "report_date"]], on="accession_number", how="left")
    d["fiscal_year"] = fiscal_year(d["report_date"], d["fiscal_year_approx"])
    for c in ("E_reason", "S_reason", "G_reason"):
        d[c] = d[c].str.replace("\x00", "", regex=False)   # NUL bytes cannot be written to CSV
    for item, x in d[COLS].groupby("item"):   # quoting keeps the line breaks of some reasons
        x.to_csv(os.path.join(out_dir, f"ai_scores_{item.replace('_', '')}.csv.gz"), index=False, compression="gzip",
                 quoting=csv.QUOTE_NONNUMERIC)
    return d[COLS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=C.DB_PATH)
    p.add_argument("--report-dates", required=True)
    p.add_argument("--out-dir", default=os.path.join(C.ROOT, "data"))
    a = p.parse_args()
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    d = export(con, a.report_dates, a.out_dir)
    print(f"{len(d):,} sections of {d.accession_number.nunique():,} filings -> {a.out_dir}")


if __name__ == "__main__":
    main()
