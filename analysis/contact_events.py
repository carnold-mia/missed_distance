"""
contact_events.py
=================
Event-frequency histograms for ball-contact classification.

Plot 1  — Full swing breakdown (all rows, pre-filter)
          Categories: Swings · Misses · Ball Contacts · In Sweet Spot ·
                      Jammed · Capped · Swung Over · Swung Under

Plot 2  — Inside sweet-spot zone contacts (post-miss exclusion)
          Sub-groups: Swung Over+Capped · Swung Under+Capped ·
                      Swung Over+Jammed · Swung Under+Jammed

Plot 3  — Outside sweet-spot zone contacts (post-miss exclusion)
          Same sub-groups as Plot 2, different population

Data encoding:
    OUTCOME                      : BALL_CONTACT, MISS, CHECK_SWING, or BAD
    IN_SWEET_SPOT_ZONE           : 1.0 = inside, 0.0 = outside
    CAPPED / JAMMED              : complementary binary flags (1.0 / 0.0)
    SWUNG_OVER / SWUNG_UNDER     : complementary binary flags (1.0 / 0.0)

Usage
-----
    python analysis/contact_events.py
"""

import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# ==============================================================
# PATHS & OUTPUT
# ==============================================================
PATH        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PATH, "figures", "contact_events")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================================================
# DATA LOADING  (same pattern as bat_contact_3d.py)
# ==============================================================
discrete_pattern = os.path.join(PATH, "data", "discrete", "*_discrete.csv")
discrete_files   = sorted(glob.glob(discrete_pattern))

if not discrete_files:
    raise FileNotFoundError(
        f"[ERROR] No discrete CSVs found at: {discrete_pattern}\n"
        "Check that analysis/data/discrete/ exists and contains *_discrete.csv files."
    )

print(f"[INFO] Loading {len(discrete_files)} game file(s)...")
game_dfs = []
for fpath in discrete_files:
    game_id = os.path.basename(fpath).replace("_discrete.csv", "")
    df_g    = pd.read_csv(fpath)
    df_g["source_game_id"] = game_id   # data-lineage tag
    game_dfs.append(df_g)

df = pd.concat(game_dfs, ignore_index=True)
n_games = len(discrete_files)
print(f"[INFO] {n_games} games · {len(df)} total rows loaded")

# ==============================================================
# SHARED HELPERS
# ==============================================================

# Colour palette — consistent across all three plots
PALETTE = {
    "Swung Over + Capped" : "#e06c4e",   # warm red-orange
    "Swung Under + Capped": "#f0a04b",   # amber
    "Swung Over + Jammed" : "#5b9bd5",   # mid blue
    "Swung Under + Jammed": "#3e7abd",   # deep blue
}


def flag(col: str, val=1.0) -> pd.Series:
    """Return boolean mask where *col* equals *val* (handles NaN cleanly)."""
    return df[col] == val


def label_match(col: str, label: str | None = None) -> pd.Series:
    """Return boolean mask where *col* equals its own column name (string label)."""
    lbl = label if label is not None else col
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col] == lbl


def outcome_match(label: str) -> pd.Series:
    """Return rows matching a canonical OUTCOME label, with legacy fallback."""
    if "OUTCOME" in df.columns:
        outcome = df["OUTCOME"].astype(str).str.strip().str.upper()
        aliases = {
            "BALL_CONTACT": {"BALL_CONTACT", "HIT"},
            "MISS": {"MISS"},
            "CHECK_SWING": {"CHECK_SWING"},
            "BAD": {"BAD"},
        }
        return outcome.isin(aliases.get(label, {label}))
    return label_match(label)


def full_swing_match() -> pd.Series:
    """Return rows representing valid full swings."""
    if "OUTCOME" in df.columns:
        return outcome_match("BALL_CONTACT") | outcome_match("MISS")
    return label_match("SWING")


def pct(n: int, total: int) -> str:
    """Format a count as 'n (p%)' for axis annotations."""
    return f"{n}\n({100 * n / total:.1f}%)" if total > 0 else str(n)


def annotate_bars(ax: plt.Axes, total: int, fmt: str = "h") -> None:
    """
    Add count + percentage labels to every bar.
    fmt='h' → horizontal bars, fmt='v' → vertical bars.
    """
    for patch in ax.patches:
        if fmt == "h":
            w = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            if w > 0:
                ax.text(
                    w + 2, y, pct(int(w), total),
                    va="center", ha="left", fontsize=9, color="#333333",
                )
        else:
            h = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            if h > 0:
                ax.text(
                    x, h + 1, pct(int(h), total),
                    va="bottom", ha="center", fontsize=9, color="#333333",
                )


# ==============================================================
# POPULATION DEFINITIONS
# ==============================================================

# --- All rows that represent a full swing (not a take / check-swing) ----------
df_swings = df[full_swing_match()].copy()

# --- Ball-contact rows; misses are already excluded --------------------------
df_contacts = df[outcome_match("BALL_CONTACT")].copy()

# Partition contacts by sweet-spot zone membership
df_inside  = df_contacts[df_contacts["IN_SWEET_SPOT_ZONE"] == 1.0].copy()
df_outside = df_contacts[df_contacts["IN_SWEET_SPOT_ZONE"] == 0.0].copy()

print(
    f"[INFO] Swings={len(df_swings)} | Contacts={len(df_contacts)} "
    f"| Inside SS={len(df_inside)} | Outside SS={len(df_outside)}"
)

# ==============================================================
# Shared bar-chart helper for Plot 1a and 1b
# ==============================================================

def horizontal_bar(event_list, title: str, filename: str, n_total: int) -> None:
    """
    Draw a horizontal bar chart from a list of (label, count, colour) tuples.
    Each bar is annotated with its count and percentage of n_total.
    """
    labels, counts, colours = zip(*event_list)
    fig, ax = plt.subplots(figsize=(11, len(labels) * 0.9 + 1.5))

    bars = ax.barh(labels, counts, color=colours, edgecolor="white", height=0.6, zorder=3)

    for bar, n in zip(bars, counts):
        ax.text(
            n + 4,
            bar.get_y() + bar.get_height() / 2,
            pct(int(n), n_total),
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.22)
    ax.grid(axis="x", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {out}")


# ==============================================================
# PLOT 1a — Outcome-level breakdown: Swings · Ball Contacts · Misses · In Sweet Spot
# Denominator is total swings so all bars are comparable on the same base.
# ==============================================================
n_swings = len(df_swings)

horizontal_bar(
    event_list=[
        ("Swings",        n_swings,                                              "#6baed6"),
        ("Ball Contacts", outcome_match("BALL_CONTACT").sum(),                   "#31a354"),
        ("Misses",        outcome_match("MISS").sum(),                           "#de2d26"),
        ("In Sweet Spot", (df_swings["IN_SWEET_SPOT_ZONE"] == 1).sum(),         "#74c476"),
    ],
    title=(
        f"Plot 1a — Outcome Breakdown  ({n_games} games, n={n_swings} swings)\n"
        "Contacts + Misses + In Sweet Spot are subsets of Swings"
    ),
    filename="plot1a_outcome_breakdown.png",
    n_total=n_swings,
)

# ==============================================================
# PLOT 1b — Contact-quality flags: Jammed · Capped · Swung Over · Swung Under
# Denominator is still total swings for consistent % across both plots.
# CAPPED/JAMMED and SWUNG_OVER/SWUNG_UNDER are complementary pairs,
# so each pair sums to 100 % of swings with valid flag data.
# ==============================================================
horizontal_bar(
    event_list=[
        ("Jammed",      (df_swings["JAMMED"]      == 1).sum(), "#fd8d3c"),
        ("Capped",      (df_swings["CAPPED"]       == 1).sum(), "#9ecae1"),
        ("Swung Over",  (df_swings["SWUNG_OVER"]  == 1).sum(), "#a1d99b"),
        ("Swung Under", (df_swings["SWUNG_UNDER"] == 1).sum(), "#fdae6b"),
    ],
    title=(
        f"Plot 1b — Contact Quality Flags  ({n_games} games, n={n_swings} swings)\n"
        "Jammed/Capped are complementary · Swung Over/Under are complementary"
    ),
    filename="plot1b_contact_quality_flags.png",
    n_total=n_swings,
)


# ==============================================================
# SHARED HELPER — sub-group bar chart for Plots 2 & 3
# ==============================================================

def plot_contact_subgroups(
    df_pop: pd.DataFrame,
    population_label: str,
    filename: str,
    fig_num: int,
) -> None:
    """
    Vertical grouped bar chart of the four contact sub-groups within
    a given ball-contact population (inside or outside sweet spot).

    Sub-groups defined by combining SWUNG_OVER/SWUNG_UNDER with CAPPED/JAMMED:
        Swung Over  + Capped → contact is capped and bat was over the ball
        Swung Under + Capped → contact is capped and bat was under the ball
        Swung Over  + Jammed → contact is jammed and bat was over the ball
        Swung Under + Jammed → contact is jammed and bat was under the ball
    """
    n_pop = len(df_pop)

    # --- Count each sub-group -------------------------------------------
    # CAPPED and JAMMED are complementary; SWUNG_OVER and SWUNG_UNDER are complementary,
    # so these four categories partition the population cleanly.
    sub_counts = {
        "Swung Over + Capped" : int(((df_pop["SWUNG_OVER"]  == 1.0) & (df_pop["CAPPED"]  == 1.0)).sum()),
        "Swung Under + Capped": int(((df_pop["SWUNG_UNDER"] == 1.0) & (df_pop["CAPPED"]  == 1.0)).sum()),
        "Swung Over + Jammed" : int(((df_pop["SWUNG_OVER"]  == 1.0) & (df_pop["JAMMED"] == 1.0)).sum()),
        "Swung Under + Jammed": int(((df_pop["SWUNG_UNDER"] == 1.0) & (df_pop["JAMMED"] == 1.0)).sum()),
    }

    labels  = list(sub_counts.keys())
    counts  = list(sub_counts.values())
    colours = [PALETTE[k] for k in labels]
    total_categorised = sum(counts)

    print(
        f"[INFO] {population_label}: "
        + " | ".join(f"{k}={v}" for k, v in sub_counts.items())
        + f" | categorised={total_categorised}/{n_pop}"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    x_pos = np.arange(len(labels))
    bars  = ax.bar(
        x_pos,
        counts,
        color=colours,
        edgecolor="white",
        width=0.55,
        zorder=3,
    )

    # Count + percentage annotation above each bar
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            n + 0.5,
            pct(int(n), n_pop),
            va="bottom", ha="center", fontsize=10, color="#333333",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Plot {fig_num} — Contact Sub-Groups: {population_label}\n"
        f"({n_games} games, n={n_pop} contacts · miss events excluded)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, max(counts) * 1.25 if counts else 1)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Legend patches to clarify axes semantics
    legend_elements = [
        mpatches.Patch(color="#e06c4e", label="Swung over = bat/sweet spot was over the ball"),
        mpatches.Patch(color="#f0a04b", label="Swung under = bat/sweet spot was under the ball"),
        mpatches.Patch(color="#5b9bd5", label="Capped = contact at barrel tip"),
        mpatches.Patch(color="#3e7abd", label="Jammed = contact near handle"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),   # below the plot, outside axes
        ncol=2,
        fontsize=9,
        framealpha=0.85,
    )

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot {fig_num} saved → {out}")
    plt.close(fig)


# ==============================================================
# PLOT 2 — Inside sweet-spot zone contacts
# ==============================================================
plot_contact_subgroups(
    df_pop=df_inside,
    population_label="Inside Sweet Spot Zone",
    filename="plot2_inside_sweet_spot_subgroups.png",
    fig_num=2,
)

# ==============================================================
# PLOT 3 — Outside sweet-spot zone contacts
# ==============================================================
plot_contact_subgroups(
    df_pop=df_outside,
    population_label="Outside Sweet Spot Zone",
    filename="plot3_outside_sweet_spot_subgroups.png",
    fig_num=3,
)

print("\n[DONE] All 3 plots written to:", FIGURES_DIR)
