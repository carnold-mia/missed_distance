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

def one_to_one_plot(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
    """
    Scatter plot for two variables on the SAME scale/unit (e.g. global vs local speed).
    Draws a 1:1 reference line so systematic bias is immediately visible as deviation
    above or below the diagonal.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, alpha=0.6, s=50, edgecolor=None, ax=ax)

    # 1:1 line spans the full shared range of both variables
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color='red', linestyle='--', linewidth=2, label='1:1 Line')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def correlation_scatter_plot(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
    """
    Scatter plot for two variables on DIFFERENT scales/units (e.g. MD speed vs xwOBAcon).
    A 1:1 line is meaningless here — instead a linear regression trend line is drawn
    so the direction and strength of the relationship is visible without implying
    the axes are comparable.
    Pearson r and n are annotated on the plot for quick reference.
    """
    # Drop rows where either variable is NaN so regression is computed on paired data
    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_clean, y=y_clean, alpha=0.6, s=50, edgecolor=None, ax=ax)

    # Linear regression via numpy for the trend line
    m, b = np.polyfit(x_clean, y_clean, deg=1)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 200)
    ax.plot(x_line, m * x_line + b,
            color='red', linestyle='--', linewidth=2, label=f'Trend  (slope={m:.4f})')

    # Pearson r annotation — tells us correlation strength at a glance
    r = np.corrcoef(x_clean, y_clean)[0, 1]
    ax.annotate(
        f'r = {r:.3f}   n = {len(x_clean):,}',
        xy=(0.05, 0.92), xycoords='axes fraction',
        fontsize=11, color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'),
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Pull columns and validate presence before plotting
# =============================================================================
md_speed_global = df_merged['MAX_MISS_SPEED_K80_GLOBAL']
md_speed_local  = df_merged['MAX_MISS_SPEED_K80_LOCAL']
xwobacon_ev_la  = df_merged['xwobacon_ev_la']

# Flag suspicious local-speed outliers — local >> global on same swing implies
# a numerical instability or unit inconsistency in the local method.
outlier_mask = md_speed_local > (md_speed_global * 5)
n_outliers = outlier_mask.sum()
if n_outliers > 0:
    print(
        f"\n[WARNING] {n_outliers} rows where local speed > 5× global speed — "
        "possible numerical instability in local MD method. Review these rows:"
    )
    print(df_merged.loc[outlier_mask, ['source_game_id', 'MLBAM_GUID',
                                       'MAX_MISS_SPEED_K80_GLOBAL',
                                       'MAX_MISS_SPEED_K80_LOCAL']].to_string(index=False))

save_dir = os.path.join(PATH, 'figures/md_velocity_exploration')
os.makedirs(save_dir, exist_ok=True)

# Plot 1: same unit — 1:1 line is appropriate
one_to_one_plot(
    md_speed_global, md_speed_local,
    title='MD Speed — Global vs Local',
    xlabel='Global MD Speed (m/s)',
    ylabel='Local MD Speed (m/s)',
    save_path=os.path.join(save_dir, 'md_speed_global_vs_local.png'),
)
print("[INFO] Plot saved → md_speed_global_vs_local.png")

# Plot 2: different units/scales — regression trend line replaces 1:1 line
correlation_scatter_plot(
    md_speed_global, xwobacon_ev_la,
    title='xwOBAcon vs Global MD Speed',
    xlabel='Global MD Speed (m/s)',
    ylabel='xwOBAcon (EV + LA)',
    save_path=os.path.join(save_dir, 'xwobacon_ev_la_vs_md_speed_global.png'),
)
print("[INFO] Plot saved → xwobacon_ev_la_vs_md_speed_global.png")


# =============================================================================
# TIME SERIES ANALYSIS — single game, single GUID
# =============================================================================
# Configure the game and trial to inspect here.
# Set TS_GUID to '' to auto-select the first BALL_CONTACT trial in TS_GAME_ID.
TS_GAME_ID = '822817'
TS_GUID    = ''          # e.g. 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'

# --- Load the time series CSV for the selected game ----------------------
ts_path = os.path.join(PATH, f'data/time_series/{TS_GAME_ID}_time_series.csv')
if not os.path.exists(ts_path):
    raise FileNotFoundError(f"[ERROR] Time series file not found: {ts_path}")

df_ts_all = pd.read_csv(ts_path)
print(f"\n[INFO] Time series loaded — {len(df_ts_all)} rows across "
      f"{df_ts_all['MLBAM_GUID'].nunique()} trials in game {TS_GAME_ID}")

# --- Resolve GUID: use supplied value or auto-pick first BALL_CONTACT ------
if not TS_GUID:
    # Pull BALL_CONTACT GUIDs from this game's discrete summary
    discrete_game_path = os.path.join(PATH, f'data/discrete/{TS_GAME_ID}_discrete.csv')
    df_disc = pd.read_csv(discrete_game_path)
    contact_guids = df_disc.loc[
        df_disc['BALL_CONTACT'] == 'BALL_CONTACT', 'MLBAM_GUID'
    ].tolist()
    if not contact_guids:
        raise ValueError(f"[ERROR] No BALL_CONTACT rows found in {TS_GAME_ID}_discrete.csv")
    TS_GUID = contact_guids[0]
    print(f"[INFO] No GUID specified — auto-selected first BALL_CONTACT: {TS_GUID}")
else:
    print(f"[INFO] Using specified GUID: {TS_GUID}")

# --- Filter time series to the single trial --------------------------------
df_trial = df_ts_all[df_ts_all['MLBAM_GUID'] == TS_GUID].copy()
if df_trial.empty:
    raise ValueError(f"[ERROR] GUID {TS_GUID} not found in {ts_path}")

df_trial = df_trial.sort_values('FRAME').reset_index(drop=True)
print(f"[INFO] Trial frames: {int(df_trial['FRAME'].min())} → {int(df_trial['FRAME'].max())}  "
      f"(n={len(df_trial)} frames)")

# --- Retrieve tmin_k80 for this trial (vertical reference line) ------------
tmin_row = df_disc[df_disc['MLBAM_GUID'] == TS_GUID] if 'df_disc' in dir() else pd.DataFrame()
if not tmin_row.empty and 'T_MIN_GLOBAL_K80' in tmin_row.columns:
    tmin_frame = float(tmin_row['T_MIN_GLOBAL_K80'].iloc[0])
    print(f"[INFO] T_MIN_GLOBAL_K80 (frame of minimum miss distance): {tmin_frame:.0f}")
else:
    # Fall back: frame where MISSED_DISTANCE_GLOBAL_K80 is smallest
    dist_col = 'MISSED_DISTANCE_GLOBAL_K80'
    valid = df_trial[dist_col].dropna()
    tmin_frame = float(df_trial.loc[valid.idxmin(), 'FRAME']) if not valid.empty else None
    print(f"[INFO] tmin_frame derived from min({dist_col}): {tmin_frame}")

frames = df_trial['FRAME'].values

ts_save_dir = os.path.join(PATH, 'figures/md_velocity_exploration')
os.makedirs(ts_save_dir, exist_ok=True)

# =============================================================================
# Figure A — Ball-in-bat position (X, Y, Z) over time
# These are the ball coordinates expressed in the bat's local coordinate frame:
#   X  —  depth (in/out face of barrel)
#   Y  —  along bat length (barrel → knob)
#   Z  —  across bat width (top / bottom)
# All three components share the frame axis so phase relationships are visible.
# =============================================================================
BIB_COLS   = ['BALL_IN_BAT_X', 'BALL_IN_BAT_Y', 'BALL_IN_BAT_Z']
BIB_LABELS = ['X — In/Out Depth (m)', 'Y — Bat Length (m)', 'Z — Top/Bottom (m)']
BIB_COLORS = ['#1f77b4', '#2ca02c', '#d62728']

fig_a, axes_a = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig_a.suptitle(
    f'Ball-in-Bat Position — Game {TS_GAME_ID}\n{TS_GUID}',
    fontsize=13, fontweight='bold'
)

for ax, col, label, color in zip(axes_a, BIB_COLS, BIB_LABELS, BIB_COLORS):
    series = df_trial[col]
    if series.notna().any():
        ax.plot(frames, series, color=color, linewidth=1.8, label=col)
    else:
        ax.text(0.5, 0.5, f'{col} — no data', transform=ax.transAxes,
                ha='center', va='center', color='gray')

    # Vertical reference at tmin — the moment of closest ball-bat approach
    if tmin_frame is not None:
        ax.axvline(tmin_frame, color='black', linestyle='--', linewidth=1.2,
                   alpha=0.7, label=f'tmin={tmin_frame:.0f}')

    ax.set_ylabel(label, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes_a[-1].set_xlabel('Frame', fontsize=11)
fig_a.tight_layout()
fig_a.savefig(os.path.join(ts_save_dir, f'ts_{TS_GAME_ID}_{TS_GUID[:8]}_ball_in_bat.png'),
              dpi=300, bbox_inches='tight')
print(f"[INFO] Fig A saved → ts_{TS_GAME_ID}_{TS_GUID[:8]}_ball_in_bat.png")
plt.close(fig_a)

# =============================================================================
# Figure B — Miss vector velocity (global frame) + scalar speed over time
# The miss vector velocity describes how fast the ball is moving relative to
# the sweet spot in each direction.  The scalar speed (magnitude) shows the
# peak and its timing relative to tmin — if peak speed occurs well before tmin
# it suggests the swing was decelerating through contact.
# =============================================================================
VEL_COLS   = [
    'MISS_VECTOR_VELOCITY_GLOBAL_K80_X',
    'MISS_VECTOR_VELOCITY_GLOBAL_K80_Y',
    'MISS_VECTOR_VELOCITY_GLOBAL_K80_Z',
    'MISSED_DISTANCE_GLOBAL_K80_SPEED',   # scalar magnitude
]
VEL_LABELS = [
    'Velocity X — Global (m/s)',
    'Velocity Y — Global (m/s)',
    'Velocity Z — Global (m/s)',
    'Scalar MD Speed — Global (m/s)',
]
VEL_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']

fig_b, axes_b = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig_b.suptitle(
    f'Miss Vector Velocity — Game {TS_GAME_ID}\n{TS_GUID}',
    fontsize=13, fontweight='bold'
)

for ax, col, label, color in zip(axes_b, VEL_COLS, VEL_LABELS, VEL_COLORS):
    series = df_trial[col]
    if series.notna().any():
        ax.plot(frames, series, color=color, linewidth=1.8, label=col)
        # Mark the peak speed frame
        peak_frame = int(df_trial.loc[series.abs().idxmax(), 'FRAME'])
        peak_val   = float(series.abs().max())
        ax.axvline(peak_frame, color=color, linestyle=':', linewidth=1.2, alpha=0.6)
        ax.annotate(
            f'peak={peak_val:.1f}  @f{peak_frame}',
            xy=(peak_frame, peak_val),
            xytext=(8, -12), textcoords='offset points',
            fontsize=8, color=color,
        )
    else:
        ax.text(0.5, 0.5, f'{col} — no data', transform=ax.transAxes,
                ha='center', va='center', color='gray')

    if tmin_frame is not None:
        ax.axvline(tmin_frame, color='black', linestyle='--', linewidth=1.2,
                   alpha=0.7, label=f'tmin={tmin_frame:.0f}')

    ax.set_ylabel(label, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes_b[-1].set_xlabel('Frame', fontsize=11)
fig_b.tight_layout()
fig_b.savefig(os.path.join(ts_save_dir, f'ts_{TS_GAME_ID}_{TS_GUID[:8]}_miss_velocity.png'),
              dpi=300, bbox_inches='tight')
print(f"[INFO] Fig B saved → ts_{TS_GAME_ID}_{TS_GUID[:8]}_miss_velocity.png")
plt.close(fig_b)