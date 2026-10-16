"""The 2020 human capital mandate as a disclosure shock: event study, difference-in-differences,
dose-response, cross-sectional correlations and cohort means (Table 3 and Fig. 1 data).

Cohorts: g = fiscal year of the first 10-K filed on or after 9 November 2020 (2020 for fiscal years
ending September-December 2020, 2021 for fiscal years ending January-August 2021).
"""
import math
import warnings

import pandas as pd
import statsmodels.formula.api as smf

from config import *
from utils import Log, coef, final_sample, save, se

warnings.filterwarnings("ignore")
log = Log("04_mandate.txt")
AI_VARS = ["S1", "S1a", "E1", "G1", "E1a", "G1a"]


def event_study(U, date=MANDATE, sample=None):
    """Within-firm change in AI scores around the first 10-K filed after the mandate (k = 0)."""
    U = U.sort_values(["cik", "filing_date"]).copy()
    U["seq"] = U.groupby("cik").cumcount()
    s0 = U[U.filing_date >= date].groupby("cik")["seq"].min().rename("seq0")
    E = U.merge(s0, on="cik")
    E["k"] = E["seq"] - E["seq0"]
    if sample is not None:   # k follows the firm's actual filing sequence; keep sample filings only
        E = E[E.accession_number.isin(sample)]
    pair = E[E.k.isin([-1, 0])].pivot(index="cik", columns="k", values="filing_date").dropna()
    ok = pair[((pair[0] - pair[-1]).dt.days.between(300, 430)) & (pair[0] < pd.Timestamp("2022-01-01"))].index
    E = E[E.cik.isin(ok) & E.k.between(-4, 3)]
    base = E[E.k == -1].set_index("cik")[AI_VARS]
    rows, diff = [], {}
    for k in range(-4, 4):
        x = E[E.k == k].set_index("cik")[AI_VARS]
        d = x - base.reindex(x.index)
        diff[k] = d
        for v in AI_VARS:
            s = d[v].dropna()
            rows.append({"k": k, "variable": v, "mean": s.mean(), "se": s.std() / math.sqrt(len(s)) if len(s) > 1 else 0, "n": len(s)})
    return pd.DataFrame(rows), diff, len(ok)


def cohorts(F, date=MANDATE):
    first = F[F.filing_date >= date].sort_values("filing_date").groupby("cik")["year"].first()
    return first.rename("g")


def wide(P, g):
    Q = P.reset_index()
    Q = Q[Q.cik.isin(g.index)].dropna(subset=["Eai", "E_LSEG"])   # firm-years of the final sample
    W = Q.pivot(index="cik", columns="year", values=["S1", "S1a", "E1", "G1", "S_LSEG", "E_LSEG", "G_LSEG", "ESG_LSEG"])
    meta = Q.groupby("cik")[["sector", "covered_since_2011", "nonstandard_fye"]].first()
    return W, meta.join(g)


def did_2x2(W, meta, v, y0, y1):
    d = pd.DataFrame({"dy": W[(v, y1)] - W[(v, y0)]}).join(meta).dropna(subset=["dy", "g", "sector"])
    d = d[d.g.isin([2020, 2021])]
    d["treated"] = (d.g == 2020).astype(int)
    m = smf.ols("dy ~ treated + C(sector)", data=d).fit(cov_type="HC1")
    return m.params["treated"], m.bse["treated"], m.pvalues["treated"], int(m.nobs), int(d.treated.sum())


def dose(W, meta, X, h):
    """Change in rating X from g-1 to g+h on the jump S1[g] - S1[g-1]; h = 'pre' uses g-2 to g-1."""
    rows = []
    for c, r in meta[meta.g.isin([2020, 2021])].iterrows():
        g = int(r.g)
        try:
            ds1 = W.loc[c, ("S1", g)] - W.loc[c, ("S1", g - 1)]
            dy = W.loc[c, (X, g - 1)] - W.loc[c, (X, g - 2)] if h == "pre" else W.loc[c, (X, g + h)] - W.loc[c, (X, g - 1)]
        except KeyError:
            continue
        rows.append({"cik": c, "ds1": ds1, "dy": dy, "g": g, "sector": r.sector, "early": r.covered_since_2011})
    d = pd.DataFrame(rows).dropna()
    m = smf.ols("dy ~ ds1 + C(g) + C(sector)", data=d).fit(cov_type="HC1")
    return m.params["ds1"], m.bse["ds1"], m.pvalues["ds1"], int(m.nobs), d


def block(rows, labels):
    c1 = " & ".join(coef(b, p, 2) for _, b, s, p, *_ in rows)
    s1 = " & ".join(se(s, 2) for _, b, s, *_ in rows)
    c2 = " & ".join(coef(bp, pp, 2) for *_, bp, sp, pp in rows)
    s2 = " & ".join(se(sp, 2) for *_, bp, sp, pp in rows)
    n = " & ".join(f"{x[4]:,}" for x in rows)
    return (" & " + " & ".join(labels) + r" \\" + "\n\\cmidrule(lr){2-5}\n"
            + r"Treated $\times$ (FY2020 $-$ FY2019) & " + c1 + r" \\" + "\n & " + s1 + r" \\" + "\n"
            + r"Placebo: treated $\times$ (FY2019 $-$ FY2018) & " + c2 + r" \\" + "\n & " + s2 + r" \\" + "\n"
            + r"Firms & " + n + r" \\")


def main():
    U = pd.read_pickle(WORK / "filings_universe.pkl")
    F = pd.read_pickle(WORK / "filings_sample.pkl")
    P = pd.read_pickle(WORK / "panel_firm_year.pkl")

    # A. event study on the filings of the final sample
    ev, diff, n_firms = event_study(U, sample=set(final_sample().accession_number))
    ev.to_csv(RES / "figure1a_data.csv", index=False)
    j = diff[0]["S1"].dropna()
    pl = (-diff[-2]["S1"]).dropna()   # placebo: change from k = -2 to k = -1
    log(f"Event study: {n_firms:,} firms")
    for v in AI_VARS:
        s = diff[0][v].dropna()
        log(f"   jump at k=0 {v}: mean {s.mean():.2f}, median {s.median():.1f}, N {len(s)}")
    log(f"   S1 placebo: mean {pl.mean():.2f}, sd {pl.std():.2f}; noise bound per 10-K {pl.std() / math.sqrt(2):.2f}")
    save(ev_firms=n_firms, ev_S1_k0=round(float(j.mean()), 2), ev_S1_k0_median=float(j.median()),
         ev_S1_k0_share10=round(float((j >= 10).mean()), 3), ev_S1_k0_n=int(len(j)),
         ev_S1a_k0=round(float(diff[0]["S1a"].mean()), 2), ev_E1_k0=round(float(diff[0]["E1"].mean()), 2),
         ev_G1_k0=round(float(diff[0]["G1"].mean()), 2), ev_G1a_k0=round(float(diff[0]["G1a"].mean()), 2),
         ev_E1a_k0=round(float(diff[0]["E1a"].mean()), 2),
         ev_S1_k1=round(float(diff[1]["S1"].mean()), 2), ev_S1_k2=round(float(diff[2]["S1"].mean()), 2),
         ev_S1_k3=round(float(diff[3]["S1"].mean()), 2),
         ev_placebo_S1_mean=round(float(pl.mean()), 2), ev_placebo_S1_sd=round(float(pl.std()), 2),
         ev_noise_bound=round(float(pl.std() / math.sqrt(2)), 2))

    # B. cohorts among firms with LSEG scores
    g = cohorts(F)
    W, meta = wide(P, g)
    log(f"Cohorts (firms): {meta.g.value_counts().to_dict()}")
    save(cohort2020=int((meta.g == 2020).sum()), cohort2021=int((meta.g == 2021).sum()))

    rowsA, rowsB, out = [], [], []
    for v in ["S1", "S1a", "E1", "G1", "S_LSEG", "E_LSEG", "G_LSEG", "ESG_LSEG"]:
        b, s, p, n, nt = did_2x2(W, meta, v, 2019, 2020)
        bp, sp, pp, n_p, _ = did_2x2(W, meta, v, 2018, 2019)
        out.append([v, "did_2019_2020", b, s, p, n, nt])
        out.append([v, "placebo_2018_2019", bp, sp, pp, n_p, None])
        log(f"   DiD {v}: {b:+.2f} (se {s:.2f}, p {p:.3f}) N={n} treated={nt} | placebo {bp:+.2f} (se {sp:.2f}, p {pp:.3f})")
        save(**{f"did_{v}": round(float(b), 2), f"did_{v}_se": round(float(s), 2), f"did_{v}_p": round(float(p), 3),
                f"did_{v}_n": n, f"did_{v}_treated": nt, f"did_{v}_placebo": round(float(bp), 2),
                f"did_{v}_placebo_se": round(float(sp), 2), f"did_{v}_placebo_p": round(float(pp), 3)})
        (rowsA if v in ("S1", "S1a", "E1", "G1") else rowsB).append((v, b, s, p, n, bp, sp, pp))
    b, s = [x for x in out if x[0] == "S_LSEG" and x[1] == "did_2019_2020"][0][2:4]
    save(did_S_LSEG_ci_low=round(float(b - 1.96 * s), 2), did_S_LSEG_ci_high=round(float(b + 1.96 * s), 2))

    # mean change in the Item 1 social score, 2019 to 2020, for treated and control firms
    d_s1 = pd.DataFrame({"dy": W[("S1", 2020)] - W[("S1", 2019)]}).join(meta).dropna(subset=["dy", "g", "sector"])
    d_s1 = d_s1[d_s1.g.isin([2020, 2021])]
    save(did_S1_change_control=round(float(d_s1[d_s1.g == 2021].dy.mean()), 2),
         did_S1_change_treated=round(float(d_s1[d_s1.g == 2020].dy.mean()), 2))

    # dose-response
    rowsC = []
    for X in ["S_LSEG", "E_LSEG", "G_LSEG"]:
        r = [X]
        for h in [0, 1, 2, "pre"]:
            bb, ss, pp, nn, d = dose(W, meta, X, h)
            r.append((bb, ss, pp, nn))
            out.append([X, f"dose_h{h}", bb, ss, pp, nn, None])
            save(**{f"dose_{X}_h{h}": round(float(bb), 4), f"dose_{X}_h{h}_se": round(float(ss), 4),
                    f"dose_{X}_h{h}_p": round(float(pp), 3), f"dose_{X}_h{h}_n": nn})
            if X == "S_LSEG" and h == 0:
                save(dose_ds1_mean=round(float(d.ds1.mean()), 2), dose_ds1_sd=round(float(d.ds1.std()), 2))
            log(f"   dose {X} h={h}: {bb:+.4f} (se {ss:.4f}, p {pp:.3f}) N={nn}")
        rowsC.append(r)
    pd.DataFrame(out, columns=["variable", "test", "coef", "se", "p", "n", "treated"]).to_csv(RES / "table3_mandate.csv", index=False)

    # cross-sectional correlation of S1 and LSEG S by year relative to adoption (balanced, g-2 to g+1)
    Q = P.reset_index().dropna(subset=["Eai", "E_LSEG"]).merge(g[g.isin([2020, 2021])].reset_index(), on="cik")
    Q["k"] = Q["year"] - Q["g"]
    bal = Q[Q.k.between(-2, 1)].dropna(subset=["S1", "S_LSEG"]).groupby("cik").filter(lambda x: len(x) == 4)
    cor = {int(k): float(x.S1.corr(x.S_LSEG)) for k, x in bal.groupby("k")}
    cor1a = {int(k): float(x.S1a.corr(x.S_LSEG)) for k, x in bal.dropna(subset=["S1a"]).groupby("k")}
    log(f"   correlation S1-LSEG S by k ({bal.cik.nunique()} firms): {cor}; S1a: {cor1a}")
    save(xcorr_firms=int(bal.cik.nunique()), **{f"xcorr_S1_k{k}": round(v, 3) for k, v in cor.items()},
         **{f"xcorr_S1a_k{k}": round(v, 3) for k, v in cor1a.items()})

    # cohort means, balanced panel 2018-2022 (Fig. 1b)
    req = [("S1", y) for y in range(2018, 2023)] + [("S_LSEG", y) for y in range(2018, 2023)]
    B = W.dropna(subset=req)
    gB = meta.loc[B.index, "g"]
    B, gB = B[gB.isin([2020, 2021])], gB[gB.isin([2020, 2021])]
    rows = []
    for c in (2020, 2021):
        x = B[gB == c]
        for v in ["S1", "S_LSEG"]:
            for y in range(2018, 2023):
                rows.append({"cohort": c, "variable": v, "year": y, "mean": x[(v, y)].mean(), "n": len(x)})
    pd.DataFrame(rows).to_csv(RES / "figure1b_data.csv", index=False)
    save(fig1b_firms=int(len(B)), fig1b_cohort2020=int((gB == 2020).sum()), fig1b_cohort2021=int((gB == 2021).sum()))

    a, bB = block(rowsA, ["S, Item 1", "S, Item 1A", "E, Item 1", "G, Item 1"]), block(rowsB, ["LSEG S", "LSEG E", "LSEG G", "LSEG ESG"])
    names = {"S_LSEG": "LSEG S", "E_LSEG": "LSEG E", "G_LSEG": "LSEG G"}
    r = rowsC[0]   # the table shows the S pillar; E and G are in table3_mandate.csv
    rows_c = [names[r[0]] + " & " + " & ".join(coef(b, p, 3) for b, s, p, n in r[1:]) + r" \\",
              " & " + " & ".join(se(s, 3) for b, s, p, n in r[1:]) + r" \\"]
    nC = " & ".join(f"{n:,}" for b, s, p, n in r[1:])
    t = rf"""\begin{{table}}[!htbp]
\centering
\caption{{Disclosure and rating responses to the 2020 human capital mandate.}}
\label{{tab:mandate}}
\small
\begin{{tabular}}{{lcccc}}
\toprule
\multicolumn{{5}}{{l}}{{\textit{{Panel A. AI scores (change in score)}}}} \\
{a}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel B. LSEG scores (change in score)}}}} \\
{bB}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{Panel C. Dose--response of the LSEG S score to the mandated jump in the AI S score of Item 1}}}} \\
 & $h=0$ & $h=1$ & $h=2$ & Placebo \\
\cmidrule(lr){{2-5}}
{chr(10).join(rows_c)}
Firms & {nC} \\
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.97\linewidth}}
\vspace{{4pt}}\footnotesize
\textit{{Notes.}} E, S, and G denote the environmental, social, and governance pillars, and FY denotes fiscal year. Treated firms file their first 10-K under the amended Item 101(c) for fiscal year 2020 (fiscal years ending
September to December 2020); control firms first comply in fiscal year 2021. Panels A and B regress the FY2019 to FY2020 change
(placebo, FY2018 to FY2019) on a treatment indicator and sector fixed effects. Panel C regresses the change in the rating
from the last pre-mandate fiscal year $g-1$ to $g+h$ on the change in the AI social score of Item 1 between $g-1$ and the first
compliant fiscal year $g$, with cohort and sector fixed effects; the placebo column uses the rating change from $g-2$ to $g-1$.
Heteroskedasticity-robust standard errors in parentheses. *, **, and *** denote significance at the 10\%, 5\%, and 1\% levels.
\end{{minipage}}
\end{{table}}
"""
    (RES / "table3_mandate.tex").write_text(t, encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
