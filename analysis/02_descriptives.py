"""Table 1: summary statistics of the final sample."""
import pandas as pd

from config import *
from utils import Log, final_sample, read_scores, save

log = Log("02_descriptives.txt")
NAMES = {"E": "Environmental", "S": "Social", "G": "Governance"}


def row(name, s):
    s = s.dropna()
    return [name, f"{len(s):,}", f"{s.mean():.2f}", f"{s.std():.2f}", f"{s.quantile(.25):.1f}", f"{s.median():.1f}", f"{s.quantile(.75):.1f}"]


def main():
    A = final_sample()
    blocks = []
    for letter, X in zip("ABC", PILLARS):
        blocks.append((f"Panel {letter}. {NAMES[X]} pillar",
                       [row("AI score, Item 1", A[f"{X}1"]), row("AI score, Item 1A", A[f"{X}1a"]),
                        row("AI score, average of Items 1 and 1A", A[f"{X}ai"]), row("LSEG pillar score", A[f"{X}_LSEG"])]))
    blocks.append(("Panel D. Overall score", [row("LSEG ESG score", A["ESG_LSEG"])]))
    pd.DataFrame([[title] + r for title, rows in blocks for r in rows],
                 columns=["Panel", "Variable", "N", "Mean", "SD", "P25", "Median", "P75"]).to_csv(RES / "table1_descriptives.csv", index=False)

    n_fy, n_f = len(A), A.index.get_level_values(0).nunique()
    body = "\n\\midrule\n".join(
        f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\\n" + "\n".join(" & ".join(r) + r" \\" for r in rows)
        for title, rows in blocks)
    t = rf"""\begin{{table}}[!htbp]
\centering
\caption{{Summary statistics by pillar.}}
\label{{tab:desc}}
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
 & N & Mean & SD & P25 & Median & P75 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.95\linewidth}}
\vspace{{4pt}}\footnotesize
\textit{{Notes.}} The sample is the firm-year panel used in all regressions, with {n_fy:,} firm-years of {n_f:,} firms with both AI
and LSEG scores and fiscal years taken from the EDGAR period of report. AI scores are the gpt-5-nano scores of Items 1 and 1A of
the 10-K for that fiscal year, on a 0--100 scale, and their average, which is the AI score used in Table~\ref{{tab:lags}}. N is lower
for each item than for the average, because an item is missing when the section is absent or shorter than 500 characters and the
average uses the available item. LSEG scores are the pillar scores and the overall ESG score.
\end{{minipage}}
\end{{table}}
"""
    (RES / "table1_descriptives.tex").write_text(t, encoding="utf-8")

    zero = (A["E_LSEG"] == 0).mean()
    n_sec = int(A["E1"].notna().sum() + A["E1a"].notna().sum())
    sec = pd.concat([A.loc[A[f"E{s}"].notna(), ["accession_number"]].reset_index().assign(item=item)
                     for s, item in (("1", "item_1"), ("1a", "item_1a"))])[["cik", "accession_number", "item"]]
    sec = sec.merge(read_scores(usecols=["cik", "accession_number", "filing_date", "item", "truncated"], dtype=str),
                    on=["cik", "accession_number", "item"])
    m = A.groupby(level="year")[["S1", "S1a"]].mean().round(1)
    log(f"Final sample: {n_fy} firm-years, {n_f} firms, {n_sec:,} scored sections ({len(sec):,} in the scores file), "
        f"{(sec.truncated == '1').mean():.2%} truncated; LSEG E equal to zero in {zero:.1%}")
    save(share_E_LSEG_zero=round(float(zero), 4), n_sections_sample=n_sec,
         pct_truncated_sample=round(100 * float((sec.truncated == "1").mean()), 2),
         mean_S1_2020=float(m.loc[2020, "S1"]), mean_S1_2021=float(m.loc[2021, "S1"]),
         mean_S1a_2020=float(m.loc[2020, "S1a"]), mean_S1a_2021=float(m.loc[2021, "S1a"]))
    log.close()


if __name__ == "__main__":
    main()
