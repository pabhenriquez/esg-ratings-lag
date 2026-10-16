"""Formatting helpers, the numbers file and the final sample."""
import json
import math

import pandas as pd

from config import AI_SCORES, LOGS, RES, WORK

NUMBERS = RES / "numbers.json"


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def num(x, dec=3):
    t = f"{abs(x):.{dec}f}"
    return ("$-$" + t) if x < 0 and float(t) != 0 else t


def coef(b, p, dec=3):
    return f"{num(b, dec)}{stars(p)}"


def se(s, dec=3):
    return f"({s:.{dec}f})"


def save(**kv):
    """Store numbers cited in the paper in results/numbers.json."""
    d = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
    for k, v in kv.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        d[k] = v
    NUMBERS.write_text(json.dumps(d, indent=1, sort_keys=True))


class Log:
    def __init__(self, name):
        self.path = LOGS / name
        self.lines = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)

    def close(self):
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def read_scores(**kw):
    """AI scores of both items, one row per section."""
    d = pd.concat([pd.read_csv(p, **kw) for p in AI_SCORES], ignore_index=True)
    return d.sort_values(["cik", "filing_date", "item"], kind="mergesort", ignore_index=True)


def final_sample():
    """Firm-years with both AI and LSEG scores."""
    return pd.read_pickle(WORK / "panel_firm_year.pkl").dropna(subset=["Eai", "E_LSEG"])
