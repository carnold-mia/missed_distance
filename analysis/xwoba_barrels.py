import glob
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon, Ellipse
import os
import sys

# Set Seaborn theme for nicer plots
sns.set_theme(style="whitegrid", palette="muted")

# ==============================================================
# BAT GEOMETRY & SWEET SPOT CONSTANTS
# All physical bat and target-zone values live here.
# Change a value once — every function and plot updates automatically.
# ==============================================================

# --- Bat Y-axis boundaries (metres, along bat length) --------
BAT_TIP_X      =  0.2134   # far end of barrel tip
BAT_TIP_ROUND  =  0.19     # where rounded tip profile begins
BAT_BARREL_END = -0.10     # straight barrel → taper transition
BAT_TAPER_END  = -0.65     # taper → handle transition
BAT_HANDLE_END = -0.83     # handle → knob-flare transition
BAT_KNOB_START = -0.845    # knob-flare → knob-cap transition
BAT_KNOB_X     = -0.8536   # far end of knob

# --- Bat cross-sectional radii (metres) ----------------------
BAT_BARREL_R   =  0.0465   # straight barrel radius
BAT_HANDLE_R   =  0.012    # handle radius
BAT_KNOB_R     =  0.025    # knob cap radius

# --- 80 % sweet spot definition ------------------------------
# Origin along the bat (Y = 0 is the pipeline's 80 % reference point).
# Z offset raises the target above the bat's geometric centreline.
SS_ORIGIN_Y    =  0.0      # m  along bat  (sweet spot Y)
SS_ORIGIN_Z    =  0.0    # m  across bat (raise above centreline)

# --- Target ellipse semi-axes (metres) -----------------------
# Semi-major: half-length along the bat (Y direction)
# Semi-minor: half-width  across the bat (Z direction)
SS_SEMI_MAJOR  =  0.06     # m
SS_SEMI_MINOR  =  0.04     # m  (expanded from original 0.02)

# --- xwOBA filter threshold for Plot 3 -----------------------
XWOBA_THRESHOLD = 0.85

# --- Plot 4 xwOBA floor filter --------------------------------
# Only contacts at or above this value enter the heat map.
# Raise to focus on mid contacts; lower to show more of the bat face.
HEATMAP_XWOBA_MIN = 0.6

# --- hexbin settings for Plot 5 -------------------------------
# gridsize controls hex density along the longer axis.
# Increase for finer resolution; decrease if bins are too sparse.
HEX_GRIDSIZE = 18

# ==============================================================

#==============================================
# Load data
#==============================================
# PATH is the file's parent directory
PATH = os.path.dirname(os.path.abspath(__file__))

# --- Step 1: Glob all per-game discrete CSVs from the discrete/ subfolder -----
# Pattern matches any file named <game_id>_discrete.csv
discrete_pattern = os.path.join(PATH, 'data/discrete/*_discrete.csv')
discrete_files = sorted(glob.glob(discrete_pattern))

if not discrete_files:
    raise FileNotFoundError(
        f"[ERROR] No discrete CSVs found at: {discrete_pattern}\n"
        "Check that the 'data/discrete/' folder exists and contains *_discrete.csv files."
    )

print(f"[INFO] Found {len(discrete_files)} discrete game files:")
for f in discrete_files:
    print(f"       {os.path.basename(f)}")

# --- Step 2: Read and concatenate all game CSVs into one master DataFrame -----
# ignore_index=True resets the row index so it is unique across games.
# The source_file column provides data lineage — which game each row came from.
game_dfs = []
for fpath in discrete_files:
    game_id = os.path.basename(fpath).replace('_discrete.csv', '')
    df_game = pd.read_csv(fpath)
    df_game['source_game_id'] = game_id   # explicit lineage tag
    game_dfs.append(df_game)

df_all_swings = pd.concat(game_dfs, ignore_index=True)
print(f"[INFO] Total swings across all games: {len(df_all_swings)}")

# --- Step 3: Load the xwOBA barrels reference table (all games in one file) ---
df_xwoba = pd.read_csv(os.path.join(PATH, 'data/xwoba_barrels.csv'))
print(f"[INFO] xwOBA barrel rows available: {len(df_xwoba)}")

# --- Step 4: Merge on MLBAM_GUID (KT swing ID) = mlbam_guid (Statcast pitch ID)
# how='left' retains every swing row; non-barrel rows receive NaN for xwobacon_ev_la.
df_merged = pd.merge(
    df_all_swings,
    df_xwoba[['mlbam_guid', 'xwobacon_ev_la']],
    left_on='MLBAM_GUID',
    right_on='mlbam_guid',
    how='left'
)

# Drop the redundant right-side join key now that the merge is complete
df_merged = df_merged.drop(columns=['mlbam_guid'])

# --- Step 5: Validate match counts per game so silent failures are visible ----
n_matched_total = df_merged['xwobacon_ev_la'].notna().sum()
print(f"\n[INFO] Match summary (barrel contacts with xwOBA data):")
match_summary = (
    df_merged.groupby('source_game_id')['xwobacon_ev_la']
    .agg(total_swings='count', matched_barrels=lambda x: x.notna().sum())
    .reset_index()
)
print(match_summary.to_string(index=False))
print(f"\n[INFO] Total matched: {n_matched_total} / {len(df_merged)} swings across all games")

# --- Step 6: Write combined merged output to disk ----------------------------
out_path = os.path.join(PATH, 'data/all_games_discrete_xwoba_barrels.csv')
df_merged.to_csv(out_path, index=False)
print(f"[INFO] Combined output saved → {out_path}")

# --- Step 7: Filter to barrel-contact rows only for plotting -----------------
df_merge_clean = df_merged.dropna(subset=['xwobacon_ev_la'])
print(f"[INFO] Barrel rows for plot: {len(df_merge_clean)}")

#==============================================
# Helper: build bat profile geometry
#==============================================
def build_bat_polygon(label='Bat Profile', facecolor='#C4A484', alpha=0.4) -> Polygon:
    """
    Build the bat outline polygon from the global BAT_* constants.
    All hardcoded numbers are replaced with named globals so changing
    a constant at the top of the file immediately changes the drawing.
    """
    x_half = np.linspace(BAT_TIP_X, BAT_KNOB_X, 500)
    y_half = np.zeros_like(x_half)

    for i, x in enumerate(x_half):
        if x > BAT_TIP_ROUND:
            # Rounded tip — smooth elliptical profile
            t = (x - BAT_TIP_ROUND) / (BAT_TIP_X - BAT_TIP_ROUND)
            y_half[i] = BAT_BARREL_R * np.sqrt(1 - 0.4 * t**2)
        elif x > BAT_BARREL_END:
            # Straight barrel — constant radius
            y_half[i] = BAT_BARREL_R
        elif x > BAT_TAPER_END:
            # Taper — smooth S-curve from barrel down to handle
            t = (x - BAT_TAPER_END) / (BAT_BARREL_END - BAT_TAPER_END)
            y_half[i] = BAT_HANDLE_R + (BAT_BARREL_R - BAT_HANDLE_R) * (
                0.5 + 0.5 * np.sin(np.pi * (t - 0.5))
            )
        elif x > BAT_HANDLE_END:
            # Handle — constant thin radius
            y_half[i] = BAT_HANDLE_R
        elif x > BAT_KNOB_START:
            # Knob flare — linearly widens from handle to knob
            t = (x - BAT_KNOB_START) / (BAT_HANDLE_END - BAT_KNOB_START)
            y_half[i] = BAT_KNOB_R - (BAT_KNOB_R - BAT_HANDLE_R) * t
        else:
            # Knob cap — rounded end
            t = (x - BAT_KNOB_START) / (BAT_KNOB_X - BAT_KNOB_START)
            y_half[i] = BAT_KNOB_R * np.sqrt(1 - 0.5 * t**2)

    bat_x = np.concatenate([x_half, x_half[::-1]])
    bat_y = np.concatenate([y_half, -y_half[::-1]])

    return Polygon(
        xy=list(zip(bat_x, bat_y)),
        closed=True,
        facecolor=facecolor,
        edgecolor='#5C4033',
        linewidth=1.5,
        alpha=alpha,
        label=label,
    )


def draw_bat_axes(ax: plt.Axes) -> None:
    """
    Add the bat polygon, boundary reference lines, and sweet-spot marker
    to an axes object. All positions are read from the global constants.
    """
    ax.add_patch(build_bat_polygon())
    # Bat tip and knob end dashed boundaries
    ax.axvline(x=BAT_TIP_X,  color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=BAT_KNOB_X, color='black', linestyle='--', alpha=0.3)
    # Barrel diameter reference lines (±barrel radius)
    ax.axhline(y= BAT_BARREL_R, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=-BAT_BARREL_R, color='gray', linestyle='--', alpha=0.3)
    # 80 % sweet spot vertical marker at SS_ORIGIN_Y
    ax.axvline(
        x=SS_ORIGIN_Y, color='red', linestyle=':', linewidth=2, alpha=0.7,
        label=f'80% Sweet Spot (Y={SS_ORIGIN_Y} m)'
    )


def format_bat_plot(fig, ax, sc, title: str) -> None:
    """Apply shared axis formatting for all contact-location plots."""
    plt.colorbar(sc, ax=ax, label='xwOBAcon (EV + LA)', fraction=0.05, pad=0.02)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Miss Vector Local Y (m) [Along Bat Length]', fontsize=12)
    ax.set_ylabel('Miss Vector Local Z (m) [Across Bat Width]', fontsize=12)
    ax.set_xlim(0.25, -0.90)
    ax.set_ylim(-0.15, 0.15)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))


#==============================================
# Shared constants
#==============================================
FIGURES_DIR = os.path.join(PATH, 'figures/xwoba_barrels')
os.makedirs(FIGURES_DIR, exist_ok=True)
n_games = df_merged['source_game_id'].nunique()

# Compute the global xwOBA range once so all three plots share the same
# colorbar scale — a point at 0.8 looks identical green on every plot.
VMIN = df_merged['xwobacon_ev_la'].min()
VMAX = df_merged['xwobacon_ev_la'].max()
print(f"[INFO] Global xwOBAcon range: {VMIN:.3f} – {VMAX:.3f}  (used as shared colorbar scale)")

# (Sweet spot and bat geometry constants are defined at the top of the file)

#==============================================
# Population setup
#==============================================
df_contact = df_merged[df_merged['OUTCOME'] == 'BALL_CONTACT'].copy()
# Sort ascending: high-xwOBA points drawn last so they sit on top
df_contact_xwoba    = df_contact[df_contact['xwobacon_ev_la'].notna()].sort_values('xwobacon_ev_la')
df_contact_no_xwoba = df_contact[df_contact['xwobacon_ev_la'].isna()]
df_elite = df_contact[df_contact['xwobacon_ev_la'] >= XWOBA_THRESHOLD].sort_values('xwobacon_ev_la')
# Filter for quality contact hex maps
df_hex = df_contact_xwoba[df_contact_xwoba['xwobacon_ev_la'] > HEATMAP_XWOBA_MIN]
print(
    f"[INFO] BALL_CONTACT rows: {len(df_contact)} total  |  "
    f"{len(df_contact_xwoba)} with xwOBA  |  "
    f"{len(df_contact_no_xwoba)} without xwOBA"
)
print(f"[INFO] Elite (xwOBAcon >= {XWOBA_THRESHOLD}): {len(df_elite)}")
print(f"[INFO] Hex filter (xwOBAcon > {HEATMAP_XWOBA_MIN}): {len(df_hex)}")


#==============================================
# Shared hex helper
# All hex plots share this function to stay DRY.
# Only the source dataframe, title, and filename differ.
#==============================================
def draw_hex_bat_plot(df_plot, title: str, filename: str) -> None:
    """
    Hexbin mean-xwOBAcon plot with bat outline and sweet spot overlay.
    Each hexagon shows the mean xwOBA of contacts whose miss-vector
    local (Y, Z) position fell inside it. Empty bins are hidden (mincnt=1).
    linewidths=0 removes borders so adjacent cells blend smoothly.
    """
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.grid(False)

    hb = ax.hexbin(
        df_plot['MISS_VECTOR_LOCAL_Y'],
        df_plot['MISS_VECTOR_LOCAL_Z'],
        C=df_plot['xwobacon_ev_la'],
        reduce_C_function=np.nanmean,   # mean xwOBA per bin
        gridsize=HEX_GRIDSIZE,
        cmap='RdYlGn',
        vmin=VMIN, vmax=VMAX,           # global colour scale — consistent across all plots
        mincnt=1,                       # hide empty bins
        linewidths=0,                   # borderless for smooth gradient look
        zorder=2,
    )

    # Bat outline on top so it frames the data
    ax.add_patch(build_bat_polygon(facecolor='none', alpha=1.0))
    ax.axvline(x=BAT_TIP_X,     color='#444', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=BAT_KNOB_X,    color='#444', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axhline(y= BAT_BARREL_R, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axhline(y=-BAT_BARREL_R, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axvline(x=SS_ORIGIN_Y, color='crimson', linestyle=':', linewidth=2, alpha=0.8,
               label=f'80% Sweet Spot (Y={SS_ORIGIN_Y} m)')
    ax.add_patch(Ellipse(
        xy=(SS_ORIGIN_Y, SS_ORIGIN_Z),
        width=2 * SS_SEMI_MAJOR, height=2 * SS_SEMI_MINOR,
        facecolor='none', edgecolor='crimson', linestyle='-', linewidth=2.0,
        label=f'Target Zone (±{SS_SEMI_MAJOR}×±{SS_SEMI_MINOR} m)', zorder=6,
    ))

    plt.colorbar(hb, ax=ax, label='Mean xwOBAcon (EV + LA)', fraction=0.05, pad=0.02)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Miss Vector Local Y (m) [Along Bat Length]', fontsize=12)
    ax.set_ylabel('Miss Vector Local Z (m) [Across Bat Width]', fontsize=12)
    ax.set_xlim(0.25, -0.90)
    ax.set_ylim(-0.15, 0.15)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved → {filename}  (n={len(df_plot)})")


#==============================================
# Plot 1 (DOT): All BALL_CONTACT — gray for no xwOBA, colored for scored contacts
# This is the only scatter plot; all other plots are hexbin.
# High-xwOBA points sorted to top so the quality-contact centre is visible.
#==============================================
fig1, ax1 = plt.subplots(figsize=(14, 4))
draw_bat_axes(ax1)

if len(df_contact_no_xwoba):
    ax1.scatter(
        df_contact_no_xwoba['MISS_VECTOR_LOCAL_Y'],
        df_contact_no_xwoba['MISS_VECTOR_LOCAL_Z'],
        color='#AAAAAA', alpha=0.45, s=35, edgecolors='none',
        label=f'Contact — no xwOBA (n={len(df_contact_no_xwoba)})', zorder=4,
    )

sc1 = ax1.scatter(
    df_contact_xwoba['MISS_VECTOR_LOCAL_Y'],
    df_contact_xwoba['MISS_VECTOR_LOCAL_Z'],
    c=df_contact_xwoba['xwobacon_ev_la'],
    cmap='RdYlGn', vmin=VMIN, vmax=VMAX,
    alpha=0.9, s=55, edgecolors='black', linewidth=0.5,
    label=f'Contact + xwOBA (n={len(df_contact_xwoba)})', zorder=5,
)

format_bat_plot(fig1, ax1, sc1,
    title=f'All Ball Contacts — Dot Plot  ({n_games} games, n={len(df_contact)})')
fig1.tight_layout()
fig1.savefig(os.path.join(FIGURES_DIR, 'xwoba_dot_all_contacts.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"[INFO] Plot 1 (dot) saved → xwoba_dot_all_contacts.png  (n={len(df_contact)})")


#==============================================
# Plot 2 (HEX): All contacts with xwOBA — full range, no floor filter
#==============================================
draw_hex_bat_plot(
    df_contact_xwoba,
    title=f'Mean xwOBAcon Hex Map — All Scored Contacts  ({n_games} games, n={len(df_contact_xwoba)}, gridsize={HEX_GRIDSIZE})',
    filename='xwoba_hex_all_barrels.png',
)

#==============================================
# Plot 3 (HEX): Elite contacts — xwOBAcon >= XWOBA_THRESHOLD
#==============================================
draw_hex_bat_plot(
    df_elite,
    title=f'Mean xwOBAcon Hex Map — xwOBAcon ≥ {XWOBA_THRESHOLD}  ({n_games} games, n={len(df_elite)}, gridsize={HEX_GRIDSIZE})',
    filename='xwoba_hex_elite.png',
)

#==============================================
# Plot 4 (HEX): Quality filter — xwOBAcon > HEATMAP_XWOBA_MIN (default 0.6)
#==============================================
draw_hex_bat_plot(
    df_hex,
    title=f'Mean xwOBAcon Hex Map — xwOBAcon > {HEATMAP_XWOBA_MIN}  ({n_games} games, n={len(df_hex)}, gridsize={HEX_GRIDSIZE})',
    filename='xwoba_hex_filtered.png',
)

plt.show()
