"""
bat_contact_3d.py
=================
3-D visualisation of ball contact location on the bat, coloured by xwOBAcon.

The bat coordinate system used throughout the pipeline:
    Y  —  along the bat length   (positive toward the barrel tip)
    Z  —  across the bat width   (positive toward the top of the bat)
    X  —  depth / in-out face    (positive toward the catcher / away from pitcher)

The 2D plots in xwoba_barrels.py project onto the Y-Z plane, silently collapsing X.
This script restores the full 3D picture by:
    1. Revolving the bat's 2D cross-section profile around the Y-axis to build a
       parametric surface mesh (the bat solid of revolution).
    2. Scattering each ball-contact point at its true (Y, X, Z) location in that space.
    3. Producing three complementary views:
         Fig 1  — 3D perspective view  (full bat + contact cloud)
         Fig 2  — End-on  view looking along the bat axis (X-Z cross-section only)
         Fig 3  — Top-down view from above (Y-Z plane, same orientation as 2D plots)

All bat geometry constants are kept at the top of this file and mirror the values
in xwoba_barrels.py.  Refactor into a shared bat_config.py when the two files
diverge further.

Usage
-----
    python analysis/bat_contact_3d.py
"""

import glob
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3D projection

# ==============================================================
# BAT GEOMETRY & SWEET SPOT CONSTANTS
# Mirror of the values in xwoba_barrels.py — change in both places
# until a shared bat_config.py is created.
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
SS_ORIGIN_Y    =  0.0      # m  along bat
SS_ORIGIN_Z    =  0.01     # m  above centreline
SS_SEMI_MAJOR  =  0.06     # m  half-length along Y
SS_SEMI_MINOR  =  0.04     # m  half-width along Z

# --- xwOBA filter for 3D scatter (shows only quality contact) -
XWOBA_MIN_3D   =  0.6      # change here to widen / narrow the cloud

# --- 3D axis limits ------------------------------------------
# Keep the same Y range as xwoba_barrels.py (tip side → knob side).
# X and Z limits are symmetric around 0 and equal to each other so
# the circular cross-section renders as a true circle, not an ellipse.
Y3D_MAX =  0.25             # m — bat tip side (matches xwoba_barrels x-axis)
Y3D_MIN = -0.90             # m — knob side
XZ_LIM  =  BAT_BARREL_R * 1.6   # m — cross-section half-extent (adds 60 % margin)

# Box aspect ratio: physical bat length vs cross-section diameter.
# Tells matplotlib how tall/wide/deep the 3D box should be so 1 m on
# the length axis equals 1 m on the X and Z axes visually.
_BAT_LEN = Y3D_MAX - Y3D_MIN           # ~1.15 m
_XZ_SPAN = XZ_LIM * 2                  # ~0.149 m
BAT_BOX_ASPECT = [_BAT_LEN / _XZ_SPAN, 1.0, 1.0]   # ≈ [7.7, 1, 1]

# --- 3D surface mesh resolution ------------------------------
# Higher values = smoother bat surface; lower = faster rendering.
N_THETA  = 72    # angular steps around bat circumference (360 / 5°)
N_Y_SURF = 300   # longitudinal steps along bat length

# ==============================================================


# ==============================================================
# BAT PROFILE FUNCTION
# Returns the barrel radius (metres) at any longitudinal position y.
# The profile follows the same piecewise geometry used in
# build_bat_polygon() in xwoba_barrels.py.
# ==============================================================

def bat_radius_at_y(y: float) -> float:
    """
    Piecewise radius of the bat as a function of Y position (metres).
    Zones (high → low Y, tip → knob):
      Rounded tip cap   : quarter-circle from 0 to BAT_BARREL_R
      Straight barrel   : constant BAT_BARREL_R
      Taper             : linear interpolation barrel → handle
      Handle            : constant BAT_HANDLE_R
      Knob flare        : linear interpolation handle → knob
      Knob cap          : quarter-circle from 0 to BAT_KNOB_R
    """
    if y >= BAT_TIP_ROUND:
        # Quarter-circle tip cap: r tapers from BAT_BARREL_R at BAT_TIP_ROUND
        # down to 0 at BAT_TIP_X (the very end of the tip).
        t = (y - BAT_TIP_ROUND) / (BAT_TIP_X - BAT_TIP_ROUND)
        return float(BAT_BARREL_R * np.sqrt(max(0.0, 1.0 - t ** 2)))

    elif y >= BAT_BARREL_END:
        # Straight cylindrical barrel
        return BAT_BARREL_R

    elif y >= BAT_TAPER_END:
        # Linear taper from barrel radius down to handle radius
        t = (y - BAT_BARREL_END) / (BAT_TAPER_END - BAT_BARREL_END)
        return BAT_BARREL_R + t * (BAT_HANDLE_R - BAT_BARREL_R)

    elif y >= BAT_HANDLE_END:
        # Straight cylindrical handle
        return BAT_HANDLE_R

    elif y >= BAT_KNOB_START:
        # Linear flare from handle radius up to knob radius
        t = (y - BAT_HANDLE_END) / (BAT_KNOB_START - BAT_HANDLE_END)
        return BAT_HANDLE_R + t * (BAT_KNOB_R - BAT_HANDLE_R)

    elif y >= BAT_KNOB_X:
        # Quarter-circle knob cap: r tapers from BAT_KNOB_R at BAT_KNOB_START
        # down to 0 at BAT_KNOB_X.
        t = (y - BAT_KNOB_START) / (BAT_KNOB_X - BAT_KNOB_START)
        return float(BAT_KNOB_R * np.sqrt(max(0.0, 1.0 - t ** 2)))

    return 0.0


def project_sweet_spot_to_surface(n_t: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the sweet spot ellipse boundary onto the physical barrel surface.

    The ellipse is defined in the Y-Z plane (same as the 2D plots):
        Y(t) = SS_ORIGIN_Y + SS_SEMI_MAJOR * cos(t)
        Z(t) = SS_ORIGIN_Z + SS_SEMI_MINOR * sin(t)

    Each point is lifted onto the barrel surface by computing:
        X = +sqrt(R(Y)² − Z²)
    which places it on the front hemisphere of the bat (the side facing
    the incoming pitch).  Z values that would exceed the barrel radius
    are clamped to the surface edge so the curve stays on the bat.

    Returns (ss_y, ss_x, ss_z) — three 1-D arrays ready for ax.plot().
    """
    t      = np.linspace(0, 2 * np.pi, n_t)
    ss_y   = SS_ORIGIN_Y + SS_SEMI_MAJOR * np.cos(t)
    ss_z_raw = SS_ORIGIN_Z + SS_SEMI_MINOR * np.sin(t)

    ss_x = np.empty(n_t)
    ss_z = np.empty(n_t)
    for i, (y_val, z_raw) in enumerate(zip(ss_y, ss_z_raw)):
        R          = bat_radius_at_y(y_val)
        z_clamped  = float(np.clip(z_raw, -R, R))   # keep point on the barrel
        ss_z[i]    = z_clamped
        ss_x[i]    = np.sqrt(max(0.0, R ** 2 - z_clamped ** 2))  # front hemisphere

    return ss_y, ss_x, ss_z


def build_sweet_spot_patch(n_y: int = 50, n_z: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a filled surface patch for the sweet spot region on the barrel.

    For each Y slice through the ellipse the Z extent of the ellipse at
    that Y is computed, then each Z value within that range is lifted onto
    the barrel surface exactly as in project_sweet_spot_to_surface.

    Returns (Y_patch, X_patch, Z_patch) — 2-D arrays of shape (n_y, n_z)
    suitable for ax.plot_surface().
    """
    y_vals  = np.linspace(SS_ORIGIN_Y - SS_SEMI_MAJOR,
                          SS_ORIGIN_Y + SS_SEMI_MAJOR, n_y)
    Y_patch = np.empty((n_y, n_z))
    X_patch = np.empty((n_y, n_z))
    Z_patch = np.empty((n_y, n_z))

    for i, y_val in enumerate(y_vals):
        # Half-width of the ellipse at this Y slice
        dy      = (y_val - SS_ORIGIN_Y) / SS_SEMI_MAJOR
        z_half  = SS_SEMI_MINOR * np.sqrt(max(0.0, 1.0 - dy ** 2))
        z_vals  = np.linspace(SS_ORIGIN_Z - z_half, SS_ORIGIN_Z + z_half, n_z)

        R = bat_radius_at_y(y_val)
        z_clamped = np.clip(z_vals, -R, R)
        x_vals    = np.sqrt(np.maximum(0.0, R ** 2 - z_clamped ** 2))

        Y_patch[i, :] = y_val
        X_patch[i, :] = x_vals
        Z_patch[i, :] = z_clamped

    return Y_patch, X_patch, Z_patch


def style_3d_axes(ax) -> None:
    """
    Apply consistent axis limits and box aspect to a 3D Axes object.

    - Y axis (bat length) spans Y3D_MIN → Y3D_MAX, matching xwoba_barrels.py.
    - X and Z axes are equal and symmetric so the circular bat cross-section
      renders as a true circle rather than an ellipse.
    - set_box_aspect enforces the physical aspect ratio of the rendered box
      so 1 metre on the length axis visually equals 1 metre on the cross axes.
    """
    ax.set_xlim(Y3D_MAX, Y3D_MIN)       # reversed: barrel tip on the left
    ax.set_ylim(-XZ_LIM, XZ_LIM)        # X cross-section — equal to Z
    ax.set_zlim(-XZ_LIM, XZ_LIM)        # Z cross-section — equal to X
    ax.set_box_aspect(BAT_BOX_ASPECT)   # physical length:width:height ratio


def build_bat_surface_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 3D parametric surface mesh for the bat by revolving the
    bat_radius_at_y profile 360° around the Y-axis.

    Returns
    -------
    Y_mesh, X_mesh, Z_mesh : 2D np.ndarray (shape N_THETA × N_Y_SURF)
        Ready to pass directly to ax.plot_surface(Y_mesh, X_mesh, Z_mesh).
        The bat length axis is Y; X and Z form the circular cross-section.
    """
    y_vals     = np.linspace(BAT_KNOB_X, BAT_TIP_X, N_Y_SURF)
    theta_vals = np.linspace(0, 2 * np.pi, N_THETA)

    # Build a 2D radius profile — vectorize handles the piecewise logic cleanly
    r_vals = np.vectorize(bat_radius_at_y)(y_vals)   # shape (N_Y_SURF,)

    # Meshgrid: rows = theta, columns = y
    Theta, Y_mesh = np.meshgrid(theta_vals, y_vals, indexing='ij')
    R_mesh        = np.meshgrid(theta_vals, r_vals,  indexing='ij')[1]

    # Revolve: X and Z are the two perpendicular cross-section axes
    X_mesh = R_mesh * np.cos(Theta)
    Z_mesh = R_mesh * np.sin(Theta)

    return Y_mesh, X_mesh, Z_mesh


# ==============================================================
# DATA LOADING  (mirrors xwoba_barrels.py Steps 1–5)
# ==============================================================
PATH = os.path.dirname(os.path.abspath(__file__))

discrete_pattern = os.path.join(PATH, 'data/discrete/*_discrete.csv')
discrete_files   = sorted(glob.glob(discrete_pattern))
if not discrete_files:
    raise FileNotFoundError(
        f"[ERROR] No discrete CSVs found at: {discrete_pattern}"
    )

game_dfs = []
for fpath in discrete_files:
    game_id  = os.path.basename(fpath).replace('_discrete.csv', '')
    df_game  = pd.read_csv(fpath)
    df_game['source_game_id'] = game_id
    game_dfs.append(df_game)

df_all = pd.concat(game_dfs, ignore_index=True)
n_games = len(discrete_files)
print(f"[INFO] Loaded {len(df_all)} swings from {n_games} games")

# Merge xwOBA barrel classifications
df_xwoba  = pd.read_csv(os.path.join(PATH, 'data/xwoba_barrels.csv'))
df_merged = pd.merge(
    df_all,
    df_xwoba[['mlbam_guid', 'xwobacon_ev_la']],
    left_on='MLBAM_GUID', right_on='mlbam_guid', how='left',
)
df_merged = df_merged.drop(columns=['mlbam_guid'])

# BALL_CONTACT is a string sentinel ('BALL_CONTACT') not a boolean —
# matches the filter pattern used in xwoba_barrels.py.
df_contact = df_merged[df_merged['BALL_CONTACT'] == 'BALL_CONTACT'].copy()
df_plot    = df_contact.dropna(subset=['xwobacon_ev_la'])
df_plot    = df_plot[df_plot['xwobacon_ev_la'] >= XWOBA_MIN_3D].copy()
print(
    f"[INFO] Contact rows: {len(df_contact)} total  |  "
    f"{len(df_plot)} with xwOBAcon ≥ {XWOBA_MIN_3D}"
)

# Shared colour scale anchored to the global dataset range (not the filtered range)
# so a green point here maps to the same shade as on the 2D scatter plots.
VMIN = df_merged['xwobacon_ev_la'].min()
VMAX = df_merged['xwobacon_ev_la'].max()

# Pull the three contact coordinates into plain arrays for clarity
y_pts = df_plot['BALL_IN_BAT_AT_TMIN_K80_Y'].values
x_pts = df_plot['BALL_IN_BAT_AT_TMIN_K80_X'].values
z_pts = df_plot['BALL_IN_BAT_AT_TMIN_K80_Z'].values
c_pts = df_plot['xwobacon_ev_la'].values

print(f"[INFO] X contact range (in-out depth): {x_pts.min():.4f} → {x_pts.max():.4f} m")
print(f"[INFO] Y contact range (bat length)  : {y_pts.min():.4f} → {y_pts.max():.4f} m")
print(f"[INFO] Z contact range (top-bottom)  : {z_pts.min():.4f} → {z_pts.max():.4f} m")

# Build bat surface mesh once — shared across all figures
Y_mesh, X_mesh, Z_mesh = build_bat_surface_mesh()

# Pre-compute sweet spot surface projections once — reused across all figures.
# ss_* : 1-D boundary curve projected onto the barrel surface
# ss_patch_* : 2-D filled patch for plot_surface
ss_y, ss_x, ss_z           = project_sweet_spot_to_surface()
ss_py, ss_px, ss_pz        = build_sweet_spot_patch()

# Colour mapper used by all scatter calls
norm    = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
cmap    = cm.RdYlGn
colours = cmap(norm(c_pts))

# Output folder
FIGURES_DIR = os.path.join(PATH, 'figures/bat_contact_3d')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================================================
# FIGURE 1 — Full 3D perspective view
# Shows the complete bat surface with contact points floating in 3D space.
# The bat surface is rendered semi-transparent so points inside the barrel
# are still visible (common for slightly inside/outside contacts).
# ==============================================================
fig1 = plt.figure(figsize=(16, 7))
ax1  = fig1.add_subplot(111, projection='3d')

# Bat surface — wheat/wood colour, very low alpha so contacts show through
ax1.plot_surface(
    Y_mesh, X_mesh, Z_mesh,
    color='wheat',
    alpha=0.12,           # semi-transparent so contact points visible inside
    linewidth=0,
    antialiased=True,
    zorder=1,
)

# Contact scatter — each point coloured by its xwOBAcon value
sc1 = ax1.scatter(
    y_pts, x_pts, z_pts,
    c=c_pts,
    cmap='RdYlGn',
    norm=norm,
    s=60,
    edgecolors='black',
    linewidth=0.4,
    depthshade=True,      # matplotlib 3D depth shading — darker = further away
    zorder=5,
)

# Sweet spot filled patch — wraps around the front hemisphere of the barrel.
# X values are computed from X = sqrt(R(Y)² - Z²) so the patch sits on the
# physical barrel surface rather than floating through the bat's interior.
ax1.plot_surface(
    ss_py, ss_px, ss_pz,
    color='red', alpha=0.35,
    linewidth=0, antialiased=True,
    zorder=4,
)
# Boundary curve on top of the filled patch for a crisp outline
ax1.plot(ss_y, ss_x, ss_z, color='red', linewidth=2.5, label='Target Zone', zorder=6)

# Vertical reference line at the sweet spot Y position (bat length axis)
ax1.plot(
    [SS_ORIGIN_Y, SS_ORIGIN_Y],
    [-XZ_LIM, XZ_LIM],
    [0, 0],
    color='red', linestyle=':', linewidth=1.5, alpha=0.5,
)

fig1.colorbar(sc1, ax=ax1, label='xwOBAcon (EV + LA)', shrink=0.5, pad=0.1)

style_3d_axes(ax1)
ax1.set_xlabel('Y — Bat Length (m)', labelpad=10)
ax1.set_ylabel('X — In/Out Depth (m)', labelpad=10)
ax1.set_zlabel('Z — Top/Bottom (m)', labelpad=10)
ax1.set_title(
    f'3D Contact Location — xwOBAcon ≥ {XWOBA_MIN_3D}  '
    f'({n_games} games, n={len(df_plot)})',
    fontsize=13, fontweight='bold', pad=15,
)
ax1.legend(loc='upper left')

# Standard isometric view: slightly elevated, looking from the barrel side
ax1.view_init(elev=22, azim=-55)

fig1.tight_layout()
fig1.savefig(os.path.join(FIGURES_DIR, 'bat_contact_3d_perspective.png'), dpi=200, bbox_inches='tight')
print("[INFO] Fig 1 saved → bat_contact_3d_perspective.png")


# ==============================================================
# FIGURE 2 — Two supplementary 2D projections
# Left : end-on view down the bat axis (X-Z cross-section)
#         This is the new dimension — shows inner vs outer face contact.
# Right: top-down view (Y-Z plane) matching the orientation of the 2D plots
#         in xwoba_barrels.py so you can directly compare.
# ==============================================================
fig2, (ax_end, ax_top) = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: end-on cross-section (X-Z plane) -------------------------
# Draw a circle representing the barrel cross-section for reference
theta_ref = np.linspace(0, 2 * np.pi, 300)
ax_end.plot(
    BAT_BARREL_R * np.cos(theta_ref),
    BAT_BARREL_R * np.sin(theta_ref),
    color='#5C4033', linewidth=1.5, alpha=0.6, label='Barrel edge',
)
# Centreline crosshairs
ax_end.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
ax_end.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)

sc_end = ax_end.scatter(
    x_pts, z_pts,
    c=c_pts, cmap='RdYlGn', norm=norm,
    s=60, edgecolors='black', linewidth=0.4, alpha=0.85,
)
fig2.colorbar(sc_end, ax=ax_end, label='xwOBAcon (EV + LA)', fraction=0.046, pad=0.04)
ax_end.set_xlabel('X — In/Out Depth (m)', fontsize=12)
ax_end.set_ylabel('Z — Top/Bottom (m)', fontsize=12)
ax_end.set_title('End-On View (looking along bat axis)\nX–Z cross-section', fontsize=12, fontweight='bold')
ax_end.set_aspect('equal')

# --- Right: top-down view (Y-Z plane) --------------------------------
# Barrel radius reference lines (same as 2D plots)
ax_top.axhline( BAT_BARREL_R, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax_top.axhline(-BAT_BARREL_R, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax_top.axvline(SS_ORIGIN_Y, color='red', linestyle=':', linewidth=1.8, alpha=0.7)

# Sweet spot ellipse in Y-Z
from matplotlib.patches import Ellipse as MplEllipse
ax_top.add_patch(MplEllipse(
    xy=(SS_ORIGIN_Y, SS_ORIGIN_Z),
    width=2 * SS_SEMI_MAJOR, height=2 * SS_SEMI_MINOR,
    facecolor='none', edgecolor='red', linewidth=2.0,
    label='Target Zone',
))

sc_top = ax_top.scatter(
    y_pts, z_pts,
    c=c_pts, cmap='RdYlGn', norm=norm,
    s=60, edgecolors='black', linewidth=0.4, alpha=0.85,
)
fig2.colorbar(sc_top, ax=ax_top, label='xwOBAcon (EV + LA)', fraction=0.046, pad=0.04)
ax_top.set_xlabel('Y — Bat Length (m)', fontsize=12)
ax_top.set_ylabel('Z — Top/Bottom (m)', fontsize=12)
ax_top.set_title('Top-Down View (Y–Z plane)\nmatches 2D xwoba_barrels plots', fontsize=12, fontweight='bold')
ax_top.set_xlim(0.25, -0.90)          # reversed — barrel tip on the left
ax_top.set_ylim(-0.15, 0.15)
ax_top.set_aspect('equal', adjustable='box')
ax_top.legend(loc='lower right')

fig2.suptitle(
    f'3D Contact Projections — xwOBAcon ≥ {XWOBA_MIN_3D}  '
    f'({n_games} games, n={len(df_plot)})',
    fontsize=13, fontweight='bold', y=1.02,
)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, 'bat_contact_3d_projections.png'), dpi=200, bbox_inches='tight')
print("[INFO] Fig 2 saved → bat_contact_3d_projections.png")


# ==============================================================
# FIGURE 3 — 3D perspective rendered from three additional angles
# Saved as a 1×3 panel: side, end-on, barrel-top views.
# Useful for presentations where a single static angle can be misleading.
# ==============================================================
VIEWS = [
    ('Side view (default)',       22,  -55),
    ('End-on (knob → barrel)',     0,    0),
    ('Top-down (above barrel)',   90,  -90),
]

fig3, axes3 = plt.subplots(
    1, 3,
    figsize=(21, 6),
    subplot_kw={'projection': '3d'},
)

for ax3, (label, elev, azim) in zip(axes3, VIEWS):
    ax3.plot_surface(
        Y_mesh, X_mesh, Z_mesh,
        color='wheat', alpha=0.12, linewidth=0, antialiased=True,
    )
    ax3.scatter(
        y_pts, x_pts, z_pts,
        c=c_pts, cmap='RdYlGn', norm=norm,
        s=40, edgecolors='black', linewidth=0.3, depthshade=True,
    )
    ax3.plot_surface(ss_py, ss_px, ss_pz, color='red', alpha=0.35, linewidth=0, antialiased=True)
    ax3.plot(ss_y, ss_x, ss_z, color='red', linewidth=1.8)
    ax3.view_init(elev=elev, azim=azim)
    style_3d_axes(ax3)
    ax3.set_title(label, fontsize=10, fontweight='bold')
    ax3.set_xlabel('Y', fontsize=8)
    ax3.set_ylabel('X', fontsize=8)
    ax3.set_zlabel('Z', fontsize=8)

fig3.suptitle(
    f'3D Contact — Three Views  ({n_games} games, n={len(df_plot)}, xwOBAcon ≥ {XWOBA_MIN_3D})',
    fontsize=13, fontweight='bold',
)
fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, 'bat_contact_3d_three_views.png'), dpi=200, bbox_inches='tight')
print("[INFO] Fig 3 saved → bat_contact_3d_three_views.png")

plt.show()
