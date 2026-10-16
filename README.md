# Do ESG ratings lag corporate disclosure?

Code and AI-generated ESG scores for *Do ESG ratings lag corporate disclosure? Evidence from generative AI and the SEC human capital mandate*.

## Contents

- `data/ai_scores_item1.csv.gz` and `data/ai_scores_item1a.csv.gz`: environmental, social and governance scores (0 to 100) for 139,526 sections (72,295 of Item 1 and 67,231 of Item 1A) of 76,582 10-K filings by 10,210 firms, filed between 2011 and 2025, scored with gpt-5-nano (snapshot gpt-5-nano-2025-08-07).
- `data/sample_sections.csv`: the 50,714 sections (26,701 10-K filings of 3,507 firms) matched to LSEG ESG data and used in the paper.
- `scoring/`: builds the database of sections, scores each section with the OpenAI API and exports the scores.
- `analysis/`: reproduces the tables and the figure of the paper.

The repository does not include the LSEG (formerly Refinitiv) ESG scores, which are licensed, or the text of the 10-K filings, which is available from SEC EDGAR.

## AI scores

One row per 10-K section.

| Column | Description |
|---|---|
| `cik` | SEC Central Index Key |
| `accession_number` | EDGAR accession number |
| `company_name`, `form`, `filing_date`, `sic`, `industry` | from EDGAR (`industry` is the SIC description) |
| `sector` | sector assigned from the SIC code |
| `report_date` | EDGAR period of report, retrieved for the sample firms when it changes the fiscal year or ends in the first week of January |
| `fiscal_year_approx` | year of the filing date minus 75 days |
| `fiscal_year` | year of `report_date` (fiscal years ending in the first week of January belong to the previous year), or `fiscal_year_approx` when `report_date` is missing |
| `item` | `item_1` (Business) or `item_1a` (Risk Factors) |
| `n_chars`, `n_tokens` | length of the section (tokens with o200k_base) |
| `truncated` | 1 if the section was cut to its first 60,000 tokens |
| `E`, `S`, `G` | scores, integers from 0 to 100 |
| `E_reason`, `S_reason`, `G_reason` | one-sentence justification returned by the model before each score |
| `model`, `prompt_version` | model snapshot and prompt version |

Sections shorter than 500 characters were not scored. A 10-K filed under several CIKs appears once, under the smallest CIK. The prompt and the request parameters are in `scoring/config.py`.

## Reproducing the paper

The analysis needs a CSV with LSEG ESG scores by firm and fiscal year, which is not distributed here. Scores were obtained from LSEG Workspace, matching firms through the tickers reported to the SEC.

| Column | Description |
|---|---|
| `cik` | SEC Central Index Key |
| `fiscal_year` | LSEG fiscal year |
| `E_LSEG`, `S_LSEG`, `G_LSEG` | environmental, social and governance pillar scores |
| `ESG_LSEG` | ESG score |

```
pip install -r requirements.txt
LSEG_PANEL=/path/to/lseg_panel.csv OUT_DIR=/path/to/output bash analysis/run_all.sh
```

Results are written to `OUT_DIR/results`: Tables 1, 2, 3 and B.1 in LaTeX, Figure 1, and `numbers.json` with the numbers cited in the text. By default `LSEG_PANEL` is `../lseg_panel.csv` and `OUT_DIR` is `../esg-ratings-lag-output`, both outside the repository.

## Scoring

1. `scoring/build_db.py` builds an SQLite database with the Item 1 and Item 1A text of each 10-K from a CSV of extracted sections.
2. `scoring/score.py batch-prepare` and `scoring/score.py batch-run` score each section with the OpenAI Batch API (requires `OPENAI_API_KEY`); `scoring/score.py pilot --dry-run` prints a request without calling the API.
3. `scoring/report_dates.py` downloads the period of report of each 10-K from the SEC submissions API.
4. `scoring/export_scores.py` writes the two score files in `data/`.

## License

Code: MIT (see `LICENSE`). Data in `data/`: CC BY 4.0.
