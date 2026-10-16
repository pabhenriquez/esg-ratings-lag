"""Table B.1: correlations between AI scores and LSEG scores."""
import importlib
import warnings

import pandas as pd
from scipy import stats

from config import *
from utils import Log, num, save, stars

warnings.filterwarnings("ignore")
log = Log("08_correlations.txt")
L = importlib.import_module("03_lagged_scores")
M = importlib.import_module("04_mandate")
NAMES = {"E": "Environmental", "S": "Social", "G": "Governance"}


def two_way(s, tol=1e-10, iters=5000):
    """Residual of firm and year fixed effects in an unbalanced panel (alternating projections)."""
    x = s.astype(float).copy()
    f, y = x.index.get_level_values(0), x.index.get_level_values(1)
    for _ in range(iters):
        x0 = x
        x = x - x.groupby(f).transform("mean")
        x = x - x.groupby(y).transform("mean")
        if (x - x0).abs().max() < tol:
            break
    return x


def c2(r, p=None):
    return num(r, 2) + (stars(p) if p is not None else "")


def panel_a(P):
    """Sample of Table 2: pooled, between-firm and within-firm correlations."""
    q = L.in_sample(L.add_lags(P))
    rows, n_fy, n_f = [], None, None
    for X in PILLARS:
        y, lags = f"{X}_LSEG", [f"{X}ai_l{k}" for k in (1, 2, 3)]
        d = q.dropna(subset=[y] + lags + [f"{X}_LSEG_l1"])
        n_fy, n_f = len(d), d.index.get_level_values(0).nunique()
        res = {"pooled": [], "between": [], "within": []}
        for k, v in enumerate([f"{X}ai"] + lags):
            m = d[[y, v]].groupby(level=0).mean()
            r = {"pooled": d[y].corr(d[v]), "between": m[y].corr(m[v]), "within": two_way(d[y]).corr(two_way(d[v]))}
            for t in res:
                res[t].append(r[t])
                save(**{f"corrA_{X}_k{k}_{t}": round(float(r[t]), 3)})
        log(f"A {X}: N={n_fy} firms={n_f} | " + " | ".join(f"{t}: " + ", ".join(f"{x:.3f}" for x in rs) for t, rs in res.items()))
        rows.append(rf"{NAMES[X]}, pooled & " + " & ".join(c2(x) for x in res["pooled"]) + r" \\")
        rows.append(r"\quad between firms & " + " & ".join(c2(x) for x in res["between"]) + r" \\")
        rows.append(r"\quad within firms & " + " & ".join(c2(x) for x in res["within"]) + r" \\")
    save(corrA_n=n_fy, corrA_firms=n_f)
    rows.append("Firm-years & " + " & ".join([f"{n_fy:,}"] * 4) + r" \\")
    rows.append("Firms & " + " & ".join([f"{n_f:,}"] * 4) + r" \\")
    return rows


def panel_b(P, g):
    """Cross-sectional correlation by fiscal year relative to adoption, balanced panel."""
    Q = P.reset_index().dropna(subset=["Eai", "E_LSEG"]).merge(g[g.isin([2020, 2021])].reset_index(), on="cik")
    Q["k"] = Q["year"] - Q["g"]
    bal = Q[Q.k.between(-2, 1)].dropna(subset=["S1", "S_LSEG"]).groupby("cik").filter(lambda x: len(x) == 4)
    rows, n_rows = [], []
    for v, label in (("S1", "Item 1"), ("S1a", "Item 1A")):
        cs, ns = [], []
        for k in (-2, -1, 0, 1):
            x = bal[bal.k == k].dropna(subset=[v])
            r, p = stats.pearsonr(x[v], x["S_LSEG"])
            cs.append(c2(r, p))
            ns.append(f"{len(x):,}")
            save(**{f"corrB_{v}_k{k}": round(float(r), 3), f"corrB_{v}_k{k}_p": round(float(p), 4), f"corrB_{v}_k{k}_n": len(x)})
        log(f"B {v}: " + ", ".join(f"{c} (N {n})" for c, n in zip(cs, ns)))
        rows.append(f"{label} & " + " & ".join(cs) + r" \\")
        n_rows.append(f"Firms, {label} & " + " & ".join(ns) + r" \\")
    return rows + n_rows


def panel_c(P, g):
    """Mandated jump in the Item 1 social score and change in the LSEG social score."""
    W, meta = M.wide(P, g)
    cs, ns, rs = [], [], []
    for h in (0, 1, 2, "pre"):
        d = M.dose(W, meta, "S_LSEG", h)[4]
        r, p = stats.pearsonr(d["ds1"], d["dy"])
        cs.append(c2(r, p))
        ns.append(f"{len(d):,}")
        rs.append(r)
        save(**{f"corrC_h{h}": round(float(r), 3), f"corrC_h{h}_p": round(float(p), 4), f"corrC_h{h}_n": len(d)})
    log("C: " + ", ".join(f"{c} (N {n})" for c, n in zip(cs, ns)))
    save(corrC_max=round(float(max(rs)), 3))
    return ["Correlation & " + " & ".join(cs) + r" \\", "Firms & " + " & ".join(ns) + r" \\"]


def main():
    P = pd.read_pickle(WORK / "panel_firm_year.pkl")
    g = M.cohorts(pd.read_pickle(WORK / "filings_sample.pkl"))
    A, B, C = panel_a(P), panel_b(P, g), panel_c(P, g)
    nl = "\n"
    t = rf"""\begin{{table}}[H]
\caption{{Correlations between AI scores and LSEG scores.}}
\label{{tab:corr}}
\small
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}lcccc}}
\toprule
\multicolumn{{5}}{{@{{}}l}}{{\rlap{{\textit{{Panel A. LSEG pillar score of year $t$ and AI score of the same pillar of year $t-k$}}}}}} \\
 & $k=0$ & $k=1$ & $k=2$ & $k=3$ \\
\cmidrule(lr){{2-5}}
{nl.join(A)}
\midrule
\multicolumn{{5}}{{@{{}}l}}{{\rlap{{\textit{{Panel B. AI social score of each item and LSEG social score, by fiscal year relative to adoption $g$}}}}}} \\
 & $g-2$ & $g-1$ & $g$ & $g+1$ \\
\cmidrule(lr){{2-5}}
{nl.join(B)}
\midrule
\multicolumn{{5}}{{@{{}}l}}{{\rlap{{\textit{{Panel C. Mandated jump in the AI social score of Item 1 and change in the LSEG social score}}}}}} \\
 & $h=0$ & $h=1$ & $h=2$ & Placebo \\
\cmidrule(lr){{2-5}}
{nl.join(C)}
\bottomrule
\end{{tabular*}}
\par\vspace{{4pt}}
\noindent\begin{{minipage}}{{\linewidth}}\footnotesize
\textit{{Notes.}} Pearson correlations. In Panel A, the AI score is the average of Items 1 and 1A and the sample is that of
Table~\ref{{tab:lags}}. Pooled correlations use all firm-years, between-firm correlations use firm means, and within-firm
correlations use the residuals of firm and year fixed effects, which is the variation used in column (3) of
Table~\ref{{tab:lags}}. Panel A reports no tests, since its pooled and within-firm correlations use repeated observations of the same
firm, and Table~\ref{{tab:lags}} reports the corresponding inference. Panel B uses the balanced panel of Section 3.3, with firms that
first comply in fiscal year 2020 or 2021; Item 1A, which the mandate did not change, is missing for some firm-years. Panel C
uses the samples of Panel C of Table~\ref{{tab:mandate}} and correlates the change in the AI social score of Item 1 from
$g-1$ to $g$ with the change in the LSEG social score from $g-1$ to $g+h$ (placebo, $g-2$ to $g-1$). In Panels B and C each
firm contributes one observation, and *, **, and *** denote significance at the 10\%, 5\%, and 1\% levels.
\end{{minipage}}
\end{{table}}
"""
    (RES / "tableB1_correlations.tex").write_text(t, encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
