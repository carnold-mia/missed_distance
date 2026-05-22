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
SS_ORIGIN_Z    =  0.01    # m  across bat (raise above centreline)

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

# --- histogram2d bin settings for Plot 4 ----------------------
# Bins are defined independently per axis so pixel aspect ratio
# roughly matches the physical bat face (length ~1.15 m, width ~0.30 m).
# Rule of thumb: N_BINS_Y / N_BINS_Z ≈ BAT length / BAT width ≈ 3.8.
# Increase bin counts for finer resolution when more data is available.
N_BINS_Y  = 46   # bins along bat length  (0.25 → −0.90 m, step ≈ 0.025 m)
N_BINS_Z  = 12   # bins across bat width  (−0.15 → 0.15 m, step ≈ 0.025 m)

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
    ax.set_xlabel('Ball Y at Contact (m) [Length of Bat]', fontsize=12)
    ax.set_ylabel('Ball Z at Contact (m) [Width of Bat]', fontsize=12)
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
# Plot 1: All barrel contacts (xwOBAcon >= 0)
#==============================================
fig1, ax1 = plt.subplots(figsize=(14, 4))
draw_bat_axes(ax1)

sc1 = ax1.scatter(
    df_merge_clean['BALL_IN_BAT_AT_TMIN_K80_Y'],
    df_merge_clean['BALL_IN_BAT_AT_TMIN_K80_Z'],
    c=df_merge_clean['xwobacon_ev_la'],
    cmap='RdYlGn',
    vmin=VMIN, vmax=VMAX,   # global scale — consistent across all 3 plots
    alpha=0.9,
    s=60,
    edgecolors='black',
    linewidth=0.5,
    label=f'All Contacts (n={len(df_merge_clean)})',
    zorder=5
)

format_bat_plot(
    fig1, ax1, sc1,
    title=f'Ball Location at MD vs xwOBAcon — All Barrels ({n_games} games)'
)

fig1.tight_layout()
fig1.savefig(os.path.join(FIGURES_DIR, 'xwoba_all.png'), dpi=300, bbox_inches='tight')
print(f"[INFO] Plot 1 saved → xwoba_all.png  (n={len(df_merge_clean)} points)")

#==============================================
# Plot 2: BALL_CONTACT only (all contact swings)
# Source is df_merged (all swings) filtered to rows where BALL_CONTACT is set.
# Contacts WITH xwOBA data are colored by xwOBAcon; contacts WITHOUT xwOBA
# (no barrel classification) are rendered in neutral gray so no data is hidden.
#==============================================
df_contact = df_merged[df_merged['BALL_CONTACT'] == 'BALL_CONTACT'].copy()
df_contact_xwoba     = df_contact[df_contact['xwobacon_ev_la'].notna()]
df_contact_no_xwoba  = df_contact[df_contact['xwobacon_ev_la'].isna()]
print(
    f"[INFO] BALL_CONTACT rows: {len(df_contact)} total  |  "
    f"{len(df_contact_xwoba)} with xwOBA  |  "
    f"{len(df_contact_no_xwoba)} without xwOBA"
)

fig2, ax2 = plt.subplots(figsize=(14, 4))
draw_bat_axes(ax2)

# Layer 1 — contacts without xwOBA classification (gray, behind colored points)
if len(df_contact_no_xwoba):
    ax2.scatter(
        df_contact_no_xwoba['BALL_IN_BAT_AT_TMIN_K80_Y'],
        df_contact_no_xwoba['BALL_IN_BAT_AT_TMIN_K80_Z'],
        color='#AAAAAA',
        alpha=0.5,
        s=40,
        edgecolors='none',
        label=f'Contact — no xwOBA (n={len(df_contact_no_xwoba)})',
        zorder=4
    )

# Layer 2 — contacts with xwOBA classification (colored, on top)
sc2 = ax2.scatter(
    df_contact_xwoba['BALL_IN_BAT_AT_TMIN_K80_Y'],
    df_contact_xwoba['BALL_IN_BAT_AT_TMIN_K80_Z'],
    c=df_contact_xwoba['xwobacon_ev_la'],
    cmap='RdYlGn',
    vmin=VMIN, vmax=VMAX,   # global scale — consistent across all 3 plots
    alpha=0.9,
    s=60,
    edgecolors='black',
    linewidth=0.5,
    label=f'Contact + xwOBA (n={len(df_contact_xwoba)})',
    zorder=5
)

format_bat_plot(
    fig2, ax2, sc2,
    title=f'Ball Location at MD — BALL_CONTACT Only ({n_games} games)'
)

fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, 'xwoba_ball_contact_only.png'), dpi=300, bbox_inches='tight')
print(f"[INFO] Plot 2 saved → xwoba_ball_contact_only.png  (n={len(df_contact)} points)")

#==============================================
# Plot 3: xwOBAcon >= 1.0 filter — same global color scale as Plots 1 & 2
# Points below 1.0 are excluded entirely (not shown, not grayed).
# vmin/vmax stay locked to the dataset-wide range so a green point here
# is the exact same shade of green as on Plots 1 and 2.
#==============================================
# XWOBA_THRESHOLD is defined in the global constants block at the top of the file.
# Start from df_contact (BALL_CONTACT rows only) then apply the threshold,
# so Plot 3 is the intersection: contacted the ball AND xwOBAcon >= XWOBA_THRESHOLD.
df_elite = df_contact[df_contact['xwobacon_ev_la'] >= XWOBA_THRESHOLD].copy()
print(f"[INFO] Elite contacts (xwOBAcon >= {XWOBA_THRESHOLD}): {len(df_elite)} / {len(df_merge_clean)} barrel rows")

fig3, ax3 = plt.subplots(figsize=(14, 4))
draw_bat_axes(ax3)

sc3 = ax3.scatter(
    df_elite['BALL_IN_BAT_AT_TMIN_K80_Y'],
    df_elite['BALL_IN_BAT_AT_TMIN_K80_Z'],
    c=df_elite['xwobacon_ev_la'],
    cmap='RdYlGn',
    vmin=VMIN, vmax=VMAX,   # global scale — do NOT re-anchor at threshold
    alpha=0.9,
    s=60,
    edgecolors='black',
    linewidth=0.5,
    label=f'xwOBAcon ≥ {XWOBA_THRESHOLD} (n={len(df_elite)})',
    zorder=5
)

# Target ellipse — all values pulled from the global constants block.
# xy center: (SS_ORIGIN_Y, SS_ORIGIN_Z) — positioned along and above the bat centreline.
# Ellipse width/height are full diameters, so multiply each semi-axis by 2.
sweet_spot_ellipse = Ellipse(
    xy=(SS_ORIGIN_Y, SS_ORIGIN_Z),
    width=2 * SS_SEMI_MAJOR,
    height=2 * SS_SEMI_MINOR,
    angle=0,
    facecolor='none',
    edgecolor='red',
    linestyle='-',
    linewidth=2.0,
    label=(
        f'Target Zone  Y={SS_ORIGIN_Y} m, Z={SS_ORIGIN_Z} m  '
        f'(±{SS_SEMI_MAJOR} × ±{SS_SEMI_MINOR} m)'
    ),
    zorder=6,
)
ax3.add_patch(sweet_spot_ellipse)

format_bat_plot(
    fig3, ax3, sc3,
    title=f'Ball Contact Location — xwOBAcon ≥ {XWOBA_THRESHOLD} ({n_games} games)'
)

fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, 'xwoba_contact_gte1.png'), dpi=300, bbox_inches='tight')
print(f"[INFO] Plot 3 saved → xwoba_contact_gte1.png  (n={len(df_elite)} points)")

#==============================================
# Plot 4: 2-D histogram heat map — mean xwOBAcon per rectangular spatial bin
# Source: df_contact (all BALL_CONTACT rows that have xwOBAcon values).
#
# Approach: np.histogram2d (weighted sum) ÷ np.histogram2d (counts) = per-bin mean.
# Bins are defined independently on each axis so pixel aspect ratio matches
# the physical bat face — avoids the hexagon distortion from hexbin on an
# elongated equal-aspect plot. See:
# https://stackoverflow.com/a/65085993
#
# Contacts without xwOBA (NaN) are silently dropped by .dropna() before
# binning so they never contribute to a bin's weighted sum or count.
#==============================================

# --- Build bin edges from global axis extents -------------------------
# histogram2d requires monotonically INCREASING edges — always go low → high.
# The visual axis reversal (bat tip on left, knob on right) is handled later
# via ax4.set_xlim(0.25, -0.90) which flips the display without touching the bins.
y_edges = np.linspace(-0.90,  0.25, N_BINS_Y + 1)   # bat length axis (Y), low→high
z_edges = np.linspace(-0.15,  0.15, N_BINS_Z + 1)   # bat width  axis (Z)

# --- Drop rows without xwOBA, then apply floor filter -----------------
# Step 1: remove contacts with no barrel classification (NaN xwOBA).
# Step 2: keep only contacts at or above HEATMAP_XWOBA_MIN so the heat map
#         highlights quality contact zones rather than averaging in weak hits.
df_hex = df_contact.dropna(subset=['xwobacon_ev_la']).copy()
df_hex = df_hex[df_hex['xwobacon_ev_la'] > HEATMAP_XWOBA_MIN]
print(
    f"[INFO] Plot 4 — contacts with xwOBAcon > {HEATMAP_XWOBA_MIN}: "
    f"{len(df_hex)} / {len(df_contact)} total contacts"
)

y_vals = df_hex['BALL_IN_BAT_AT_TMIN_K80_Y'].values
z_vals = df_hex['BALL_IN_BAT_AT_TMIN_K80_Z'].values
w_vals = df_hex['xwobacon_ev_la'].values

# Weighted sum of xwOBA per bin
H_sum, _, _ = np.histogram2d(y_vals, z_vals, bins=[y_edges, z_edges], weights=w_vals)
# Count of contacts per bin
H_cnt, _, _ = np.histogram2d(y_vals, z_vals, bins=[y_edges, z_edges])

# Mean = sum / count; bins with zero contacts → NaN (imshow renders as blank)
with np.errstate(invalid='ignore'):   # suppress "divide by zero" for empty bins
    H_mean = H_sum / H_cnt
    H_mean[H_cnt == 0] = np.nan

# --- Plot ---------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(14, 4))

# imshow: transpose so rows = Z axis, columns = Y axis.
# origin='lower' keeps Z increasing upward.
# extent aligns pixel edges with data coordinates.
im = ax4.imshow(
    H_mean.T,
    origin='lower',
    aspect='auto',                      # fills the axes without equal-aspect distortion
    cmap='RdYlGn',
    vmin=VMIN, vmax=VMAX,               # shared global colour scale
    extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],
    zorder=2,
    interpolation='nearest',            # nearest-neighbour keeps bin boundaries crisp
)

# Bat outline only — facecolor='none' so the heat map shows through
ax4.add_patch(build_bat_polygon(facecolor='none', alpha=1.0))
ax4.axvline(x=BAT_TIP_X,    color='black', linestyle='--', linewidth=1.0, alpha=0.5)
ax4.axvline(x=BAT_KNOB_X,   color='black', linestyle='--', linewidth=1.0, alpha=0.5)
ax4.axhline(y= BAT_BARREL_R, color='gray',  linestyle='--', linewidth=0.8, alpha=0.4)
ax4.axhline(y=-BAT_BARREL_R, color='gray',  linestyle='--', linewidth=0.8, alpha=0.4)
ax4.axvline(
    x=SS_ORIGIN_Y, color='red', linestyle=':', linewidth=2, alpha=0.7,
    label=f'80 % Sweet Spot (Y={SS_ORIGIN_Y} m)'
)

# Target ellipse — same global constants as Plots 3
ax4.add_patch(Ellipse(
    xy=(SS_ORIGIN_Y, SS_ORIGIN_Z),
    width=2 * SS_SEMI_MAJOR,
    height=2 * SS_SEMI_MINOR,
    angle=0,
    facecolor='none',
    edgecolor='red',
    linestyle='-',
    linewidth=2.0,
    label=(
        f'Target Zone  Y={SS_ORIGIN_Y} m, Z={SS_ORIGIN_Z} m  '
        f'(±{SS_SEMI_MAJOR}×±{SS_SEMI_MINOR} m)'
    ),
    zorder=6,
))

plt.colorbar(im, ax=ax4, label='Mean xwOBAcon per bin (EV + LA)', fraction=0.05, pad=0.02)
ax4.set_title(
    f'Mean xwOBAcon Heat Map — xwOBAcon > {HEATMAP_XWOBA_MIN}  '
    f'({n_games} games, {N_BINS_Y}×{N_BINS_Z} bins)',
    fontsize=14, fontweight='bold'
)
ax4.set_xlabel('Ball Y at Contact (m) [Length of Bat]', fontsize=12)
ax4.set_ylabel('Ball Z at Contact (m) [Width of Bat]', fontsize=12)
ax4.set_xlim(0.25, -0.90)
ax4.set_ylim(-0.15, 0.15)
ax4.set_aspect('equal', adjustable='box')
ax4.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

fig4.tight_layout()
fig4.savefig(os.path.join(FIGURES_DIR, 'xwoba_contact_heatmap.png'), dpi=300, bbox_inches='tight')
print(
    f"[INFO] Plot 4 saved → xwoba_contact_heatmap.png  "
    f"(n={len(df_hex)} rows, xwOBAcon > {HEATMAP_XWOBA_MIN}, {N_BINS_Y}×{N_BINS_Z} bins)"
)

#==============================================
# Plot 5: hexbin heat map — mean xwOBAcon per hexagonal spatial bin
# Source: same df_hex used for Plot 4 (xwOBAcon > HEATMAP_XWOBA_MIN).
#
# hexbin with C= and reduce_C_function=np.nanmean is the matplotlib-native
# approach — no manual binning required. Each hexagon shows the mean xwOBA
# of all contacts whose (Y, Z) coordinate fell inside it.
# mincnt=1 hides empty hex cells (renders as background rather than zero).
#
# Note: hexagons distort on equal-aspect elongated axes vs the rectangular
# bins in Plot 4 — both are kept so the approaches can be compared directly.
# Adjust HEX_GRIDSIZE in the global constants block to change resolution.
#==============================================
# White canvas so the sequential colourmap fades cleanly to background
fig5, ax5 = plt.subplots(figsize=(14, 4), facecolor='white')
ax5.set_facecolor('white')

# Remove the seaborn grid lines for this plot — the smooth hex aesthetic
# is cleaner on a plain white background with no competing grid lines.
ax5.grid(False)

# hexbin: linewidths=0 removes the borders between hexagons so adjacent
# cells blend smoothly — this is the primary change that creates the
# gradient-fade look seen in the reference image.
# cmap='YlGn' is a sequential single-hue map (light → dark) which reads
# naturally as "more/better" and fades to white at the floor value — same
# visual language as the reference. RdYlGn is used on Plots 1–3 for
# diverging good/bad signal; here a sequential map is more appropriate
# because after the HEATMAP_XWOBA_MIN filter all values are already "good".
hb = ax5.hexbin(
    df_hex['BALL_IN_BAT_AT_TMIN_K80_Y'],
    df_hex['BALL_IN_BAT_AT_TMIN_K80_Z'],
    C=df_hex['xwobacon_ev_la'],       # z-values averaged within each hex bin
    reduce_C_function=np.nanmean,     # mean xwOBA per bin; NaN contacts excluded
    gridsize=HEX_GRIDSIZE,            # set in global constants block
    cmap='RdYlGn',                    # diverging red→yellow→green; matches Plots 1–3
    vmin=VMIN,                        # global dataset minimum — same scale as Plots 1–3
    vmax=VMAX,                        # top of global xwOBA range
    mincnt=1,                         # hide empty hex cells (show as white background)
    linewidths=0,                     # no borders → smooth gradient between cells
    zorder=2,
)

# Bat outline drawn after hexbin so it sits on top of the heat map
ax5.add_patch(build_bat_polygon(facecolor='none', alpha=1.0))
ax5.axvline(x=BAT_TIP_X,    color='#444444', linestyle='--', linewidth=1.0, alpha=0.5)
ax5.axvline(x=BAT_KNOB_X,   color='#444444', linestyle='--', linewidth=1.0, alpha=0.5)
ax5.axhline(y= BAT_BARREL_R, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax5.axhline(y=-BAT_BARREL_R, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax5.axvline(
    x=SS_ORIGIN_Y, color='crimson', linestyle=':', linewidth=2, alpha=0.8,
    label=f'80 % Sweet Spot (Y={SS_ORIGIN_Y} m)'
)

# Target ellipse — same global parameters as Plots 3 and 4
ax5.add_patch(Ellipse(
    xy=(SS_ORIGIN_Y, SS_ORIGIN_Z),
    width=2 * SS_SEMI_MAJOR,
    height=2 * SS_SEMI_MINOR,
    angle=0,
    facecolor='none',
    edgecolor='crimson',
    linestyle='-',
    linewidth=2.0,
    label=(
        f'Target Zone  Y={SS_ORIGIN_Y} m, Z={SS_ORIGIN_Z} m  '
        f'(±{SS_SEMI_MAJOR}×±{SS_SEMI_MINOR} m)'
    ),
    zorder=6,
))

plt.colorbar(hb, ax=ax5, label='Mean xwOBAcon per bin (EV + LA)', fraction=0.05, pad=0.02)
ax5.set_title(
    f'Mean xwOBAcon Hex Map — xwOBAcon > {HEATMAP_XWOBA_MIN}  '
    f'({n_games} games, gridsize={HEX_GRIDSIZE})',
    fontsize=14, fontweight='bold'
)
ax5.set_xlabel('Ball Y at Contact (m) [Length of Bat]', fontsize=12)
ax5.set_ylabel('Ball Z at Contact (m) [Width of Bat]', fontsize=12)
ax5.set_xlim(0.25, -0.90)
ax5.set_ylim(-0.15, 0.15)
ax5.set_aspect('equal', adjustable='box')
ax5.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

fig5.tight_layout()
fig5.savefig(os.path.join(FIGURES_DIR, 'xwoba_contact_hexbin.png'), dpi=300, bbox_inches='tight')
print(
    f"[INFO] Plot 5 saved → xwoba_contact_hexbin.png  "
    f"(n={len(df_hex)} rows, xwOBAcon > {HEATMAP_XWOBA_MIN}, gridsize={HEX_GRIDSIZE})"
)

plt.show()
