"""Topics cited in the Item 1 rationales in the last 10-K before the mandate (k = -1) and the first after it (k = 0)."""
import importlib
import re

import pandas as pd

from config import *
from utils import Log, final_sample, read_scores, save

log = Log("07_rationales.txt")
M = importlib.import_module("04_mandate")
TOPICS = {"diversity": r"divers|inclusi|equit|women|gender|minorit",
          "safety": r"safety|injur|TRIR|TCIR|health",
          "training": r"train|development|learning|career",
          "turnover_engagement": r"turnover|retention|engagement|attrition",
          "human_capital": r"human capital|workforce|employee|personnel|talent",
          "ethics": r"code of conduct|ethic",
          "board": r"board|committee|oversight",
          "absence": r"\b(?:no|not|none|lack\w*|limited|minimal|without|absen\w*|little|generic|boilerplate)\b"}


def main():
    U = pd.read_pickle(WORK / "filings_universe.pkl")
    sample = set(final_sample().accession_number)
    _, diff, _ = M.event_study(U, sample=sample)
    U = U.sort_values(["cik", "filing_date"]).copy()
    U["seq"] = U.groupby("cik").cumcount()
    s0 = U[U.filing_date >= MANDATE].groupby("cik")["seq"].min().rename("seq0")
    E = U.merge(s0, on="cik")
    E["k"] = E["seq"] - E["seq0"]
    E = E[E.cik.isin(diff[0].index) & E.k.isin([-1, 0]) & E.accession_number.isin(sample)][["cik", "accession_number", "k"]]
    R = read_scores(usecols=["cik", "accession_number", "filing_date", "item", "S", "G", "S_reason", "G_reason"],
                    dtype={"cik": str, "accession_number": str})
    R = R[(R.item == "item_1") & R.accession_number.isin(E.accession_number)].drop(columns=["cik", "filing_date", "item"])
    J = E.merge(R, on="accession_number")
    share = {}
    for k in (-1, 0):
        x = J[J.k == k]
        for t, rx in TOPICS.items():
            share[(k, "S", t)] = x.S_reason.str.contains(rx, flags=re.I, regex=True).mean()
            share[(k, "G", t)] = x.G_reason.str.contains(rx, flags=re.I, regex=True).mean()
    for t in TOPICS:
        log(f"{t:20s} S: k=-1 {share[(-1, 'S', t)]:.3f} -> k=0 {share[(0, 'S', t)]:.3f} | "
            f"G: k=-1 {share[(-1, 'G', t)]:.3f} -> k=0 {share[(0, 'G', t)]:.3f}")
        save(**{f"rationale_S_{t}_km1": round(float(share[(-1, 'S', t)]), 3), f"rationale_S_{t}_k0": round(float(share[(0, 'S', t)]), 3),
                f"rationale_G_{t}_km1": round(float(share[(-1, 'G', t)]), 3), f"rationale_G_{t}_k0": round(float(share[(0, 'G', t)]), 3)})
    # firms whose G score jumps by 10 or more at k = 0: does the G rationale cite human capital topics?
    w = J.pivot_table(index="cik", columns="k", values="G")
    jump = w[(w[0] - w[-1]) >= 10].index
    x = J[(J.k == 0) & J.cik.isin(jump)]
    hc = x.G_reason.str.contains(TOPICS["human_capital"] + "|" + TOPICS["diversity"], flags=re.I, regex=True).mean()
    log(f"Firms with a G jump of 10 or more at k=0: {len(jump)}; G rationale citing human capital or diversity: {hc:.3f}")
    save(rationale_G_jump10_n=int(len(jump)), rationale_G_jump10_cites_hc=round(float(hc), 3), rationale_firms=int(J.cik.nunique()))
    log.close()


if __name__ == "__main__":
    main()
