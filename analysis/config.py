"""Paths and constants."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AI_SCORES = [ROOT / "data" / "ai_scores_item1.csv.gz", ROOT / "data" / "ai_scores_item1a.csv.gz"]
SAMPLE = ROOT / "data" / "sample_sections.csv"
LSEG_PANEL = Path(os.environ.get("LSEG_PANEL", ROOT.parent / "lseg_panel.csv"))   # not distributed
OUT = Path(os.environ.get("OUT_DIR", ROOT.parent / "esg-ratings-lag-output"))
WORK = OUT / "work"
RES = OUT / "results"
LOGS = RES / "logs"

MANDATE = pd.Timestamp("2020-11-09")    # effective date of the amendment to Regulation S-K
ADOPTION = pd.Timestamp("2020-08-26")   # adoption date (Release 33-10825)
PILLARS = ["E", "S", "G"]

for d in (WORK, RES, LOGS):
    d.mkdir(parents=True, exist_ok=True)
