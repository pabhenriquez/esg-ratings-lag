"""Figure 1 from figure1a_data.csv and figure1b_data.csv."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RES

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def main():
    a = pd.read_csv(RES / "figure1a_data.csv")
    b = pd.read_csv(RES / "figure1b_data.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    styles = {"S1": ("Social, Item 1", BLUE, "o", "-"), "S1a": ("Social, Item 1A", ORANGE, "s", "--"),
              "G1": ("Governance, Item 1", GREEN, "^", ":")}
    for v, (lab, col, mk, ls) in styles.items():
        x = a[a.variable == v].sort_values("k")
        ax1.errorbar(x.k, x["mean"], yerr=1.96 * x["se"], color=col, marker=mk, ms=4, lw=1.4, ls=ls, capsize=0,
                     elinewidth=0.8, label=lab, mfc=col, mec="white", mew=0.6)
        ax1.annotate(lab, (x.k.iloc[-1], x["mean"].iloc[-1]), xytext=(5, 0), textcoords="offset points",
                     va="center", fontsize=7.5, color=INK)
    ax1.axvline(-0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax1.axhline(0, color=INK2, lw=0.6)
    ax1.text(-0.62, ax1.get_ylim()[1] * 0.97, "First 10-K\nunder the\nmandate", fontsize=7, color=INK2, va="top", ha="right")
    ax1.set_xticks(range(-4, 4))
    ax1.set_xlim(-4.4, 5.6)
    ax1.set_xlabel("Filings relative to the first 10-K filed after 9 Nov 2020")
    ax1.set_ylabel("Change in AI score vs. last pre-mandate 10-K")
    ax1.set_title("(a) AI scores, firms in the sample", fontsize=8.5, loc="left", color=INK)

    styles_b = {("S1", 2020): ("AI S, Item 1: FY2020 cohort", BLUE, "o", "-", BLUE),
                ("S1", 2021): ("AI S, Item 1: FY2021 cohort", BLUE, "o", "--", "white"),
                ("S_LSEG", 2020): ("LSEG S: FY2020 cohort", ORANGE, "s", "-", ORANGE),
                ("S_LSEG", 2021): ("LSEG S: FY2021 cohort", ORANGE, "s", "--", "white")}
    for (v, c), (lab, col, mk, ls, mfc) in styles_b.items():
        x = b[(b.variable == v) & (b.cohort == c)].sort_values("year")
        ax2.plot(x.year, x["mean"], color=col, marker=mk, ms=4.2, lw=1.4, ls=ls, label=lab, mfc=mfc, mec=col, mew=1.0)
    ax2.set_xticks(range(2018, 2023))
    ax2.set_xticklabels([f"FY{y}" for y in range(2018, 2023)])
    ax2.set_ylabel("Mean score (0-100)")
    ax2.set_title("(b) AI and LSEG social scores, by adoption cohort", fontsize=8.5, loc="left", color=INK)
    ax2.legend(loc="upper left", fontsize=7, handlelength=2.6)
    ax2.set_ylim(min(b["mean"].min(), 35) - 2, b["mean"].max() + 17)
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(RES / "figure1.pdf")
    fig.savefig(RES / "figure1.png", dpi=300)


if __name__ == "__main__":
    main()
