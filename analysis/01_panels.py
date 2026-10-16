"""Build the firm-year panels from the AI scores, the sample filings and the LSEG panel."""
import pandas as pd

from config import *
from utils import Log, read_scores, save

log = Log("01_panels.txt")


def load_scores():
    return read_scores(dtype={"cik": str, "accession_number": str, "sic": str}, parse_dates=["filing_date"], low_memory=False)


def load_lseg():
    if not LSEG_PANEL.exists():
        raise SystemExit(f"LSEG panel not found at {LSEG_PANEL}; set LSEG_PANEL (see README)")
    return pd.read_csv(LSEG_PANEL, dtype={"cik": str}).rename(columns={"fiscal_year": "year"})


def filings(d, year_col):
    """One row per 10-K with the scores of Items 1 and 1A and their average."""
    meta = ["cik", "accession_number", "filing_date", year_col, "company_name", "sector", "industry", "sic"]
    f = d.drop_duplicates(["cik", "accession_number"])[meta].rename(columns={year_col: "year"})
    for item, suf in [("item_1", "1"), ("item_1a", "1a")]:
        x = d[d["item"] == item][["cik", "accession_number", "E", "S", "G", "n_tokens"]]
        x = x.rename(columns={"E": f"E{suf}", "S": f"S{suf}", "G": f"G{suf}", "n_tokens": f"tok{suf}"})
        f = f.merge(x, on=["cik", "accession_number"], how="left")
    for X in PILLARS:
        f[f"{X}ai"] = f[[f"{X}1", f"{X}1a"]].mean(axis=1)
    return f


def panel(f, lseg):
    """Annual grid by firm: AI scores by fiscal year of the 10-K, LSEG scores by LSEG fiscal year."""
    ai = f.sort_values("filing_date").drop_duplicates(["cik", "year"], keep="last")
    cols = ["filing_date", "accession_number"] + [f"{X}{s}" for X in PILLARS for s in ("1", "1a", "ai")] + ["tok1", "tok1a"]
    ai = ai.set_index(["cik", "year"])[cols]
    ls = lseg.set_index(["cik", "year"])[["E_LSEG", "S_LSEG", "G_LSEG", "ESG_LSEG"]]
    both = ai.join(ls, how="outer")
    rng = both.reset_index().groupby("cik")["year"].agg(["min", "max"])
    idx = pd.MultiIndex.from_tuples([(c, y) for c, r in rng.iterrows() for y in range(int(r["min"]), int(r["max"]) + 1)],
                                    names=["cik", "year"])
    P = both.reindex(idx)
    meta = f.sort_values("filing_date").groupby("cik")[["company_name", "sector", "industry", "sic"]].last()
    P = P.join(meta, on="cik")
    P = P.join(ls.reset_index().groupby("cik")["year"].min().rename("first_lseg_year"), on="cik")
    P["covered_since_2011"] = (P["first_lseg_year"] == 2011).astype(int)
    # fiscal years ending January to July: fiscal year equal to the filing year and 10-K filed before August
    nonstd = f[(f["year"] == f["filing_date"].dt.year) & (f["filing_date"].dt.month <= 7)]["cik"].unique()
    P["nonstandard_fye"] = P.index.get_level_values(0).isin(nonstd).astype(int)
    return P


def universe(d):
    """One row per scored 10-K, for the event study."""
    keys = ["cik", "accession_number", "filing_date", "company_name", "sector"]
    out = None
    for item, suf in [("item_1", "1"), ("item_1a", "1a")]:
        x = d[d["item"] == item][keys + ["E", "S", "G"]].rename(columns={"E": f"E{suf}", "S": f"S{suf}", "G": f"G{suf}"})
        out = x if out is None else out.merge(x, on=keys, how="outer")
    return out


def main():
    d = load_scores()
    sample = pd.read_csv(SAMPLE, dtype=str)   # 10-K sections matched to LSEG
    s = d.merge(sample, on=["cik", "accession_number", "item"]).sort_values(["company_name", "filing_date", "item"])
    lseg = load_lseg()
    f = filings(s, "fiscal_year")
    P = panel(f, lseg)
    U = universe(d)
    f.to_pickle(WORK / "filings_sample.pkl")
    P.to_pickle(WORK / "panel_firm_year.pkl")
    U.to_pickle(WORK / "filings_universe.pkl")
    f0 = filings(s, "fiscal_year_approx")   # approximate fiscal year, for a robustness check
    f0.to_pickle(WORK / "filings_sample_approx.pkl")
    panel(f0, lseg).to_pickle(WORK / "panel_firm_year_approx.pkl")

    fil = s.drop_duplicates(["cik", "accession_number"])
    corr = fil["fiscal_year"] != fil["fiscal_year_approx"]
    rd = pd.to_datetime(fil.loc[corr, "report_date"], errors="coerce")
    dec = ((rd.dt.month == 12) | ((rd.dt.month == 1) & (rd.dt.day <= 7))).mean()
    log(f"10-Ks with a corrected fiscal year: {corr.sum():,} of {len(fil):,} ({corr.mean():.2%}); "
        f"December or early-January year ends among them: {dec:.1%}")
    both = P.dropna(subset=["Eai", "E_LSEG"])
    log(f"Scored universe: {len(U):,} 10-Ks of {U.cik.nunique():,} firms | sample: {len(f):,} 10-Ks of "
        f"{f.cik.nunique():,} firms | firm-years with AI and LSEG scores: {len(both):,} "
        f"({both.index.get_level_values(0).nunique():,} firms)")
    save(n_fy_corrected=int(corr.sum()), pct_fy_corrected=round(100 * float(corr.mean()), 1),
         pct_corrected_dec_fye=round(100 * float(dec), 1),
         n_sections_universe=int(U["E1"].notna().sum() + U["E1a"].notna().sum()), n_filings_universe=int(len(U)),
         n_firms_universe=int(U.cik.nunique()), n_filings_sample=int(len(f)), n_firms_sample=int(f.cik.nunique()),
         n_firm_years=int(len(both)), n_firms=int(both.index.get_level_values(0).nunique()),
         n_firms_covered_2011=int(P.groupby(level=0)["covered_since_2011"].first().sum()))
    log.close()


if __name__ == "__main__":
    main()
