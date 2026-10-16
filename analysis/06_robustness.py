"""Robustness checks for Tables 2 and 3."""
import importlib
import warnings

import pandas as pd
import statsmodels.formula.api as smf

from config import *
from utils import Log, save

warnings.filterwarnings("ignore")
log = Log("06_robustness.txt")
L = importlib.import_module("03_lagged_scores")
M = importlib.import_module("04_mandate")


def lag_model(q, X, suffix, name, kmax=3):
    y = f"{X}_LSEG"
    xs = [f"{X}{suffix}_l{k}" for k in range(1, kmax + 1)]
    d = L.in_sample(q).dropna(subset=[y] + xs)
    re, n, nf = L.estimate(d, y, xs, "re")
    fe, _, _ = L.estimate(d, y, xs, "fe")
    sig = [x for x in xs if fe.pvalues[x] < 0.05]
    log(f"{name} {X}: N={n} firms={nf} | RE " + ", ".join(f"{re.params[x]:.3f}" for x in xs)
        + " | FE " + ", ".join(f"{fe.params[x]:.3f}{'*' if fe.pvalues[x] < .05 else ''}" for x in xs))
    return re, fe, n, nf, sig


def main():
    P = pd.read_pickle(WORK / "panel_firm_year.pkl")
    q = L.add_lags(P)

    for X in PILLARS:   # five lags
        re, fe, n, nf, sig = lag_model(q, X, "ai", "five lags", kmax=5)
        save(**{f"rob_5lags_{X}_re_l1": round(float(re.params[f"{X}ai_l1"]), 3),
                f"rob_5lags_{X}_fe_l1": round(float(fe.params[f"{X}ai_l1"]), 3),
                f"rob_5lags_{X}_fe_sig": len(sig), f"rob_5lags_{X}_n": n})
    qe = q[(q["covered_since_2011"] == 1) & (q.index.get_level_values(1) <= 2022)]
    for X in PILLARS:   # firms covered by LSEG since 2011, 2011-2022
        re, fe, n, nf, sig = lag_model(qe, X, "ai", "covered since 2011")
        save(**{f"rob_early_{X}_re_l1": round(float(re.params[f"{X}ai_l1"]), 3),
                f"rob_early_{X}_fe_l1": round(float(fe.params[f"{X}ai_l1"]), 3),
                f"rob_early_{X}_fe_l1_p": round(float(fe.pvalues[f"{X}ai_l1"]), 3), "rob_early_firms": nf})
    for X in PILLARS:   # each item separately
        for s in ("1", "1a"):
            re, fe, n, nf, sig = lag_model(q, X, s, f"Item {s}")
            save(**{f"rob_item{s}_{X}_re_l1": round(float(re.params[f"{X}{s}_l1"]), 3),
                    f"rob_item{s}_{X}_fe_l1": round(float(fe.params[f"{X}{s}_l1"]), 3)})
    re, fe, n, nf, sig = lag_model(q[q["E_LSEG"] != 0], "E", "ai", "E without zeros")
    save(rob_nozero_E_re_l1=round(float(re.params["Eai_l1"]), 3), rob_nozero_E_fe_l1=round(float(fe.params["Eai_l1"]), 3))
    q0 = L.add_lags(pd.read_pickle(WORK / "panel_firm_year_approx.pkl"))
    for X in PILLARS:   # approximate fiscal year
        re, fe, n, nf, sig = lag_model(q0, X, "ai", "approximate fiscal year")
        save(**{f"rob_approxfy_{X}_re_l1": round(float(re.params[f"{X}ai_l1"]), 3),
                f"rob_approxfy_{X}_fe_l1": round(float(fe.params[f"{X}ai_l1"]), 3)})

    F = pd.read_pickle(WORK / "filings_sample.pkl")
    Wd, meta = M.wide(P, M.cohorts(F, ADOPTION))   # cohorts from the adoption date
    for v in ["S1", "S_LSEG"]:
        b, s, p, n, nt = M.did_2x2(Wd, meta, v, 2019, 2020)
        log(f"adoption date {v}: {b:+.2f} (se {s:.2f}) N={n}")
        save(**{f"rob_adoption_{v}": round(float(b), 2), f"rob_adoption_{v}_se": round(float(s), 2)})
    Wd, meta = M.wide(P, M.cohorts(F))
    ms = meta[meta.nonstandard_fye == 0]   # without non-standard fiscal year ends
    for v in ["S1", "S_LSEG"]:
        b, s, p, n, nt = M.did_2x2(Wd, ms, v, 2019, 2020)
        log(f"standard fiscal year ends {v}: {b:+.2f} (se {s:.2f}) N={n}")
        save(**{f"rob_nonstd_{v}": round(float(b), 2), f"rob_nonstd_{v}_se": round(float(s), 2)})
    P0 = pd.read_pickle(WORK / "panel_firm_year_approx.pkl")
    F0 = pd.read_pickle(WORK / "filings_sample_approx.pkl")
    W0, m0 = M.wide(P0, M.cohorts(F0))   # approximate fiscal year
    for v in ["S1", "S_LSEG"]:
        b, s, p, n, nt = M.did_2x2(W0, m0, v, 2019, 2020)
        log(f"approximate fiscal year {v}: {b:+.2f} (se {s:.2f}) N={n}")
        save(**{f"rob_approxfy_did_{v}": round(float(b), 2), f"rob_approxfy_did_{v}_se": round(float(s), 2)})
    for h in [0, 1, 2]:   # early (large) versus late LSEG coverage
        _, _, _, _, d = M.dose(Wd, meta, "S_LSEG", h)
        m = smf.ols("dy ~ ds1 * early + C(g) + C(sector)", data=d).fit(cov_type="HC1")
        log(f"dose h={h}: late {m.params['ds1']:+.4f} (p {m.pvalues['ds1']:.3f}); "
            f"early minus late {m.params['ds1:early']:+.4f} (p {m.pvalues['ds1:early']:.3f}) N={int(m.nobs)}")
        save(**{f"rob_het_h{h}_late": round(float(m.params["ds1"]), 4), f"rob_het_h{h}_late_p": round(float(m.pvalues["ds1"]), 3),
                f"rob_het_h{h}_diff": round(float(m.params["ds1:early"]), 4), f"rob_het_h{h}_diff_p": round(float(m.pvalues["ds1:early"]), 3)})
    log.close()


if __name__ == "__main__":
    main()
