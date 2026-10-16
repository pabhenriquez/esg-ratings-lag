"""Table 2: lagged AI scores and LSEG pillar scores, and the reverse regression."""
import warnings

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

from config import *
from utils import Log, coef, save, se

warnings.filterwarnings("ignore")
log = Log("03_lagged_scores.txt")
NAMES = {"E": "Environmental", "S": "Social", "G": "Governance"}


def add_lags(P, kmax=5):
    q = P.copy()
    g = q.groupby(level=0)
    for X in PILLARS:
        for k in range(1, kmax + 1):
            q[f"{X}ai_l{k}"] = g[f"{X}ai"].shift(k)
            for s in ("1", "1a"):
                q[f"{X}{s}_l{k}"] = g[f"{X}{s}"].shift(k)
        q[f"{X}_LSEG_l1"] = g[f"{X}_LSEG"].shift(1)
    return q


def in_sample(q):
    """Firm-years with AI and LSEG scores in year t; lagged AI scores may come from earlier years."""
    return q[q["Eai"].notna() & q["E_LSEG"].notna()]


def estimate(d, y, xs, kind):
    d = d[[y] + xs].dropna()
    if kind == "re":
        m = RandomEffects(d[y], sm.add_constant(d[xs])).fit(cov_type="clustered", cluster_entity=True)
    elif kind == "re_year":
        yd = pd.get_dummies(d.index.get_level_values(1), prefix="y", drop_first=True).astype(float)
        yd.index = d.index
        m = RandomEffects(d[y], sm.add_constant(pd.concat([d[xs], yd], axis=1))).fit(cov_type="clustered", cluster_entity=True)
    else:
        m = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True, drop_absorbed=True).fit(
            cov_type="clustered", cluster_entity=True)
    return m, len(d), d.index.get_level_values(0).nunique()


def table(q):
    rows_tex, rows_csv = [], []
    for i, X in enumerate(PILLARS):
        y, lags, ylag = f"{X}_LSEG", [f"{X}ai_l{k}" for k in (1, 2, 3)], f"{X}_LSEG_l1"
        d = in_sample(q).dropna(subset=[y] + lags + [ylag])
        specs = [("re", lags), ("re_year", lags), ("fe", lags), ("fe", [ylag] + lags)]
        res = [estimate(d, y, xs, kind) for kind, xs in specs]
        rows_tex.append(rf"\multicolumn{{5}}{{l}}{{\textit{{Panel {'ABC'[i]}. Dependent variable: LSEG {NAMES[X]} pillar score$_t$}}}} \\")
        for v, label in [(lags[0], r"AI score$_{t-1}$"), (lags[1], r"AI score$_{t-2}$"), (lags[2], r"AI score$_{t-3}$"),
                         (ylag, r"LSEG score$_{t-1}$")]:
            c, s = [], []
            for (m, n, nf), (kind, xs) in zip(res, specs):
                if v in xs:
                    c.append(coef(m.params[v], m.pvalues[v]))
                    s.append(se(m.std_errors[v]))
                    rows_csv.append([X, v, kind, len(xs), m.params[v], m.std_errors[v], m.pvalues[v], n, nf])
                    key = f"lags_{X}_{kind}{'_ldv' if ylag in xs else ''}_{v.split('_')[-1] if v != ylag else 'lsegl1'}"
                    save(**{key: round(float(m.params[v]), 4), key + "_p": round(float(m.pvalues[v]), 4)})
                else:
                    c.append("")
                    s.append("")
            rows_tex.append(label + " & " + " & ".join(c) + r" \\")
            rows_tex.append(" & " + " & ".join(s) + r" \\")
        rows_tex.append(r"Observations & " + " & ".join(f"{n:,}" for _, n, _ in res) + r" \\")
        rows_tex.append(r"Firms & " + " & ".join(f"{nf:,}" for _, _, nf in res) + r" \\")
        if i < 2:
            rows_tex.append(r"\midrule")
        log(f"{X}: " + " | ".join(f"{kind}: " + ", ".join(f"{m.params[x]:.3f}" for x in xs) for (m, _, _), (kind, xs) in zip(res, specs)))
        save(**{f"lags_{X}_n": res[0][1], f"lags_{X}_firms": res[0][2]})
    return rows_tex, rows_csv


def main():
    q = add_lags(pd.read_pickle(WORK / "panel_firm_year.pkl"))
    rows_tex, rows_csv = table(q)
    pd.DataFrame(rows_csv, columns=["pillar", "variable", "model", "k", "coef", "se", "p", "n", "firms"]).to_csv(
        RES / "table2_lags.csv", index=False)
    body = "\n".join(rows_tex)
    t = rf"""\begin{{table}}[!htbp]
\centering
\caption{{Lagged AI scores and LSEG pillar scores.}}
\label{{tab:lags}}
\small
\begin{{tabular}}{{lcccc}}
\toprule
 & (1) & (2) & (3) & (4) \\
 & Random & Random effects & Firm and year & Firm and year FE \\
 & effects & + year FE & fixed effects & + lagged rating \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.97\linewidth}}
\vspace{{4pt}}\footnotesize
\textit{{Notes.}} The AI score is the gpt-5-nano score of the same pillar, averaged over Items 1 and 1A of the 10-K for fiscal
year $t-k$. Column (1) is a random-effects specification without time effects, as in \citet{{wang2025}}. All columns use the same
sample, the firm-years of Table~\ref{{tab:desc}} with three lags of the AI score and the lagged rating available. FE denotes fixed effects. Standard errors clustered by firm
in parentheses. *, **, and *** denote significance at the 10\%, 5\%, and 1\% levels.
\end{{minipage}}
\end{{table}}
"""
    (RES / "table2_lags.tex").write_text(t, encoding="utf-8")

    # reverse regression: does the lagged rating predict the AI score?
    for X in PILLARS:
        m, n, nf = estimate(in_sample(q), f"{X}ai", [f"{X}_LSEG_l1", f"{X}ai_l1"], "fe")
        v = f"{X}_LSEG_l1"
        log(f"Reverse {X}: LSEG(t-1) -> AI(t): {m.params[v]:.4f} (se {m.std_errors[v]:.4f}, p {m.pvalues[v]:.3f}), N={n}")
        save(**{f"reverse_{X}": round(float(m.params[v]), 4), f"reverse_{X}_se": round(float(m.std_errors[v]), 4),
                f"reverse_{X}_p": round(float(m.pvalues[v]), 4), f"reverse_{X}_n": n})
    log.close()


if __name__ == "__main__":
    main()
