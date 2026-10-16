"""Model, prompt and request settings used to score each 10-K section."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("ESG_DB", os.path.join(os.path.dirname(ROOT), "esg_10k.sqlite"))
BATCH_DIR = os.environ.get("ESG_BATCH_DIR", os.path.expanduser("~/esg_batches"))

ITEMS = ("item_1", "item_1a")
SECTION_NAME = {"item_1": "Item 1. Business", "item_1a": "Item 1A. Risk Factors"}
MIN_CHARS = 500
ENCODING = "o200k_base"

MODEL = "gpt-5-nano"            # snapshot gpt-5-nano-2025-08-07
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 4000    # includes reasoning tokens
MAX_INPUT_TOKENS = 60_000       # longer sections are truncated
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an expert ESG (Environmental, Social and Governance) analyst.

You will receive ONE section of a company's annual report on Form 10-K filed with the U.S. Securities and Exchange Commission (SEC): either "Item 1. Business" or "Item 1A. Risk Factors". Your task is the following question: "Can you help me assess its ESG development for that year based on the annual report and give a score on a 100-point scale (separately for Environmental, Social, Governance)?"

Rules:
- Base your assessment ONLY on the text provided, plus general knowledge of what is material for the company's industry. Do not use outside information about the company.
- Score each pillar with an integer from 0 to 100:
  0-20   : no meaningful disclosure on this pillar, or evidence of poor practices.
  21-40  : minimal or generic boilerplate mentions; no concrete policies, programs or metrics.
  41-60  : moderate: some concrete policies, programs, targets or metrics are described.
  61-80  : well developed: several concrete initiatives with targets, metrics, oversight or accountability.
  81-100 : leading practice: comprehensive, quantified, integrated into strategy, with clear governance and accountability.
- Environmental covers: emissions and energy, climate strategy and climate-risk management, water, waste, pollution, biodiversity, environmental compliance, environmental impact of products.
- Social covers: employees (health and safety, diversity, training, labor relations), customers and product safety, data privacy, supply chain and human rights, community relations.
- Governance covers: board structure and independence, executive compensation, shareholder rights, ethics and anti-corruption, internal controls, risk oversight, transparency.
- If the section is "Item 1A. Risk Factors", assess how thoroughly the company identifies, discloses and manages E, S and G related risks: generic boilerplate risk lists score low; specific, quantified risks with mitigation measures score high.
- Judge the substance of the disclosure, not its length. Be consistent across companies so that scores are comparable.
- For each pillar give first a one-sentence justification (max 30 words, citing the concrete evidence), then the score.
- Reply ONLY with the requested JSON object."""


def user_prompt(company_name, sic, sic_description, sector, filing_date, item, text, truncated):
    note = (f"\n[Note: the section text was truncated to its first {MAX_INPUT_TOKENS:,} tokens.]"
            if truncated else "")
    return (f"Company: {company_name}\n"
            f"Industry (SIC {sic or 'n/a'}): {sic_description or 'n/a'}\n"
            f"Sector: {sector or 'n/a'}\n"
            f"Filing date of the 10-K: {filing_date}\n"
            f"Section: {SECTION_NAME[item]}\n\n"
            f"<section_text>\n{text}\n</section_text>{note}")


# Structured output; the reason comes before each score
JSON_SCHEMA = {
    "name": "esg_scores",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "E_reason": {"type": "string", "description": "One sentence (max 30 words) with the concrete evidence for the Environmental score."},
            "E": {"type": "integer", "description": "Environmental score, integer 0-100."},
            "S_reason": {"type": "string", "description": "One sentence (max 30 words) with the concrete evidence for the Social score."},
            "S": {"type": "integer", "description": "Social score, integer 0-100."},
            "G_reason": {"type": "string", "description": "One sentence (max 30 words) with the concrete evidence for the Governance score."},
            "G": {"type": "integer", "description": "Governance score, integer 0-100."},
        },
        "required": ["E_reason", "E", "S_reason", "S", "G_reason", "G"],
        "additionalProperties": False,
    },
}
