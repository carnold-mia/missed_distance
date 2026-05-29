"""
bat_contact_3d_interactive.py
==============================
Interactive 3-D visualisation of ball contact location on the bat, built with
Plotly.  Output is a single self-contained HTML file (~3–5 MB) that opens in
any browser — click-drag to rotate, scroll to zoom, hover a point to see its
xwOBAcon value and GUID.

To embed in Notion:
    1. Upload the HTML to SharePoint / OneDrive.
    2. Copy the sharing link (set to "Anyone with the link can view").
    3. In Notion, type /embed and paste the URL.

Bat coordinate system (same as every other script in this pipeline):
    Y  —  along bat length  (positive = barrel tip)
    Z  —  across bat width  (positive = top of bat)
    X  —  in/out depth face (positive = front, facing pitcher)

Usage
-----
    python analysis/bat_contact_3d_interactive.py
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==============================================================
# BAT GEOMETRY & SWEET SPOT CONSTANTS  (mirrors bat_contact_3d.py)
# ==============================================================
BAT_TIP_X      =  0.2134
BAT_TIP_ROUND  =  0.19
BAT_BARREL_END = -0.10
BAT_TAPER_END  = -0.65
BAT_HANDLE_END = -0.83
BAT_KNOB_START = -0.845
BAT_KNOB_X     = -0.8536

BAT_BARREL_R   =  0.0465
BAT_HANDLE_R   =  0.012
BAT_KNOB_R     =  0.025
BAT_CENTER_Z   = -0.01

SS_ORIGIN_Y    =  0.0
SS_ORIGIN_Z    =  0.0
SS_SEMI_MAJOR  =  0.06
SS_SEMI_MINOR  =  0.04

# xwOBA floor filter — contacts below this are excluded from the scatter
XWOBA_MIN_3D   =  0.6

# Surface mesh resolution — matches bat_contact_3d.py exactly so the bat
# shape is identical across the static matplotlib and interactive plotly outputs.
N_THETA  = 72    # angular steps around circumference  (360 / 5°)
N_Y_SURF = 300   # steps along bat length

# ==============================================================


# ==============================================================
# BAT PROFILE HELPERS  (identical logic to bat_contact_3d.py)
# ==============================================================

def bat_radius_at_y(y: float) -> float:
    """Returns bat radius (m) at longitudinal position y (m)."""
    if y >= BAT_TIP_ROUND:
        t = (y - BAT_TIP_ROUND) / (BAT_TIP_X - BAT_TIP_ROUND)
        return float(BAT_BARREL_R * np.sqrt(max(0.0, 1.0 - t ** 2)))
    elif y >= BAT_BARREL_END:
        return BAT_BARREL_R
    elif y >= BAT_TAPER_END:
        t = (y - BAT_BARREL_END) / (BAT_TAPER_END - BAT_BARREL_END)
        return BAT_BARREL_R + t * (BAT_HANDLE_R - BAT_BARREL_R)
    elif y >= BAT_HANDLE_END:
        return BAT_HANDLE_R
    elif y >= BAT_KNOB_START:
        t = (y - BAT_HANDLE_END) / (BAT_KNOB_START - BAT_HANDLE_END)
        return BAT_HANDLE_R + t * (BAT_KNOB_R - BAT_HANDLE_R)
    elif y >= BAT_KNOB_X:
        t = (y - BAT_KNOB_START) / (BAT_KNOB_X - BAT_KNOB_START)
        return float(BAT_KNOB_R * np.sqrt(max(0.0, 1.0 - t ** 2)))
    return 0.0


def build_bat_surface_mesh():
    """
    Revolve the bat profile 360° around the Y axis.
    Returns (Y, X, Z) meshgrids of shape (N_THETA × N_Y_SURF) for go.Surface.
    """
    y_vals    = np.linspace(BAT_KNOB_X, BAT_TIP_X, N_Y_SURF)
    theta     = np.linspace(0, 2 * np.pi, N_THETA)
    r_vals    = np.vectorize(bat_radius_at_y)(y_vals)

    Theta, Y_mesh = np.meshgrid(theta, y_vals, indexing='ij')
    R_mesh        = np.meshgrid(theta, r_vals,  indexing='ij')[1]

    X_mesh = R_mesh * np.cos(Theta)
    Z_mesh = BAT_CENTER_Z + R_mesh * np.sin(Theta)
    return Y_mesh, X_mesh, Z_mesh


def project_sweet_spot(n_t: int = 400):
    """
    Project sweet spot ellipse boundary onto the barrel surface using
    angular wrapping so the curve forms a complete closed oval.

    SS_ORIGIN_Z is in the SWEET_SPOT_ORIGIN local frame, so it is converted
    to a bat-centreline relative angle before mapping back to local Z.

        θ_centre = arcsin((SS_ORIGIN_Z - BAT_CENTER_Z) / R)
        θ_half   = arcsin(SS_SEMI_MINOR / R)
        θ(t)     = θ_centre + θ_half * sin(t)
        X = R*cos(θ),  Z = BAT_CENTER_Z + R*sin(θ)

    Returns (ss_y, ss_x, ss_z) 1-D arrays.
    """
    t    = np.linspace(0, 2 * np.pi, n_t)
    ss_y = SS_ORIGIN_Y + SS_SEMI_MAJOR * np.cos(t)
    ss_x = np.empty(n_t)
    ss_z = np.empty(n_t)
    for i, (yv, tv) in enumerate(zip(ss_y, t)):
        R          = bat_radius_at_y(yv)
        theta_cen  = np.arcsin(np.clip((SS_ORIGIN_Z - BAT_CENTER_Z) / R, -1.0, 1.0))
        theta_half = np.arcsin(np.clip(SS_SEMI_MINOR / R, -1.0, 1.0))
        theta      = theta_cen + theta_half * np.sin(tv)
        ss_x[i]    = R * np.cos(theta)
        ss_z[i]    = BAT_CENTER_Z + R * np.sin(theta)
    return ss_y, ss_x, ss_z


def build_sweet_spot_patch(n_y: int = 60, n_z: int = 30):
    """
    Build filled surface patch for sweet spot on barrel using angular
    wrapping so the patch curves over the top of the bat rather than
    being clipped at the barrel edge.

    For each Y slice the angular half-width is scaled by the ellipse
    factor sqrt(1 − dy²) to match the elliptical footprint in plan view.

    Returns (Y, X, Z) 2-D arrays for go.Surface.
    """
    y_vals  = np.linspace(SS_ORIGIN_Y - SS_SEMI_MAJOR,
                          SS_ORIGIN_Y + SS_SEMI_MAJOR, n_y)
    Y_p = np.empty((n_y, n_z))
    X_p = np.empty((n_y, n_z))
    Z_p = np.empty((n_y, n_z))
    for i, yv in enumerate(y_vals):
        R          = bat_radius_at_y(yv)
        dy         = (yv - SS_ORIGIN_Y) / SS_SEMI_MAJOR
        theta_cen  = np.arcsin(np.clip((SS_ORIGIN_Z - BAT_CENTER_Z) / R, -1.0, 1.0))
        theta_half = np.arcsin(np.clip(SS_SEMI_MINOR / R, -1.0, 1.0))
        theta_half_local = theta_half * np.sqrt(max(0.0, 1.0 - dy ** 2))
        thetas = np.linspace(theta_cen - theta_half_local,
                             theta_cen + theta_half_local, n_z)
        Y_p[i] = yv
        X_p[i] = R * np.cos(thetas)
        Z_p[i] = BAT_CENTER_Z + R * np.sin(thetas)
    return Y_p, X_p, Z_p


# ==============================================================
# DATA LOADING
# ==============================================================
PATH = os.path.dirname(os.path.abspath(__file__))

discrete_files = sorted(glob.glob(os.path.join(PATH, 'data/discrete/*_discrete.csv')))
if not discrete_files:
    raise FileNotFoundError("[ERROR] No discrete CSVs found in data/discrete/")

game_dfs = []
for fpath in discrete_files:
    gid  = os.path.basename(fpath).replace('_discrete.csv', '')
    df_g = pd.read_csv(fpath)
    df_g['source_game_id'] = gid
    game_dfs.append(df_g)

df_all   = pd.concat(game_dfs, ignore_index=True)
n_games  = len(discrete_files)
df_xwoba = pd.read_csv(os.path.join(PATH, 'data/xwoba_barrels.csv'))

df_merged = pd.merge(
    df_all,
    df_xwoba[['mlbam_guid', 'xwobacon_ev_la']],
    left_on='MLBAM_GUID', right_on='mlbam_guid', how='left',
).drop(columns=['mlbam_guid'])

def outcome_mask(df: pd.DataFrame, label: str) -> pd.Series:
    """Return rows matching the canonical OUTCOME label, with legacy fallback."""
    if 'OUTCOME' in df.columns:
        outcome = df['OUTCOME'].astype(str).str.strip().str.upper()
        aliases = {'BALL_CONTACT': {'BALL_CONTACT', 'HIT'}}
        return outcome.isin(aliases.get(label, {label}))
    if label in df.columns:
        return df[label] == label
    return pd.Series(False, index=df.index)


# All ball-contact rows (xwOBA may be NaN for non-barrel contacts)
df_contact = df_merged[outcome_mask(df_merged, 'BALL_CONTACT')].copy()

# Filtered version — only contacts with an xwOBA value above the floor
df_filtered = df_contact.dropna(subset=['xwobacon_ev_la'])
df_filtered = df_filtered[df_filtered['xwobacon_ev_la'] >= XWOBA_MIN_3D].copy()

# Global colour scale anchored to the full dataset range so both files
# share the same colour meaning — a green point in v1 == green in v2.
VMIN = df_merged['xwobacon_ev_la'].min()
VMAX = df_merged['xwobacon_ev_la'].max()

print(f"[INFO] {n_games} games | {len(df_contact)} total contacts | "
      f"{len(df_filtered)} with xwOBAcon ≥ {XWOBA_MIN_3D}")

# Build geometry once — shared by both figures
Y_mesh, X_mesh, Z_mesh = build_bat_surface_mesh()
ss_y, ss_x, ss_z       = project_sweet_spot()
ss_py, ss_px, ss_pz    = build_sweet_spot_patch()

# Visual aspect ratio (capped so the bat reads as bat-shaped, not a line)
ASP = 6.0

FIGURES_DIR = os.path.join(PATH, 'figures/bat_contact_3d')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================
# FIGURE BUILDER
# Accepts any subset of df_contact so the same layout / geometry
# is used for both the unfiltered and filtered versions.
# ==============================================================

def build_figure(df_plot: pd.DataFrame, title: str) -> go.Figure:
    """
    Build a fully configured interactive Plotly 3D bat figure.

    Parameters
    ----------
    df_plot : rows to scatter (must have MISS_VECTOR_LOCAL_Y/X/Z
              and xwobacon_ev_la columns; NaN xwOBA shown as grey).
    title   : main title string displayed above the plot.
    """
    fig = go.Figure()

    # --- Bat surface -----------------------------------------------
    fig.add_trace(go.Surface(
        x=Y_mesh, y=X_mesh, z=Z_mesh,
        colorscale=[[0, '#A0785A'], [1, '#A0785A']],
        showscale=False,
        opacity=0.35,
        lighting=dict(ambient=0.65, diffuse=0.7, roughness=0.5, fresnel=0.2),
        contours=dict(
            x=dict(show=True, color='rgba(80,50,20,0.55)', width=1),
            z=dict(show=True, color='rgba(80,50,20,0.30)', width=1),
        ),
        name='Bat Surface',
        hoverinfo='skip',
    ))

    # --- Sweet spot patch + boundary -------------------------------
    fig.add_trace(go.Surface(
        x=ss_py, y=ss_px, z=ss_pz,
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False, opacity=0.45,
        name='Target Zone', hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=ss_y, y=ss_x, z=ss_z,
        mode='lines', line=dict(color='red', width=4),
        name='Target Zone Boundary', hoverinfo='skip',
    ))

    # --- Contact scatter -------------------------------------------
    # Contacts with a valid xwOBA value use the RdYlGn colour scale.
    # Contacts without xwOBA (non-barrel hits) are shown in grey so
    # they are visible but clearly distinguished from scored contacts.
    has_xwoba  = df_plot['xwobacon_ev_la'].notna()
    df_scored  = df_plot[has_xwoba]
    df_unscored = df_plot[~has_xwoba]

    # Unscored contacts (grey) — only present in the unfiltered version
    if not df_unscored.empty:
        fig.add_trace(go.Scatter3d(
            x=df_unscored['MISS_VECTOR_LOCAL_Y'].values,
            y=df_unscored['MISS_VECTOR_LOCAL_X'].values,
            z=df_unscored['MISS_VECTOR_LOCAL_Z'].values,
            mode='markers',
            marker=dict(size=5, color='#999999', opacity=0.5,
                        line=dict(color='#666666', width=0.3)),
            text=[
                f"GUID: {g[:8]}…<br>No xwOBA data<br>"
                f"Y: {y:.4f} m  X: {x:.4f} m  Z: {z:.4f} m<br>Game: {gid}"
                for g, y, x, z, gid in zip(
                    df_unscored['MLBAM_GUID'],
                    df_unscored['MISS_VECTOR_LOCAL_Y'],
                    df_unscored['MISS_VECTOR_LOCAL_X'],
                    df_unscored['MISS_VECTOR_LOCAL_Z'],
                    df_unscored['source_game_id'],
                )
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Contact (no xwOBA)',
        ))

    # Scored contacts (RdYlGn colour scale)
    if not df_scored.empty:
        fig.add_trace(go.Scatter3d(
            x=df_scored['MISS_VECTOR_LOCAL_Y'].values,
            y=df_scored['MISS_VECTOR_LOCAL_X'].values,
            z=df_scored['MISS_VECTOR_LOCAL_Z'].values,
            mode='markers',
            marker=dict(
                size=6,
                color=df_scored['xwobacon_ev_la'].values,
                colorscale='RdYlGn',
                cmin=VMIN, cmax=VMAX,
                colorbar=dict(
                    title=dict(text='xwOBAcon<br>(EV + LA)', side='right'),
                    thickness=18, len=0.55,
                    x=0.90, xanchor='left',
                    y=0.5,  yanchor='middle',
                    outlinewidth=1, outlinecolor='#cccccc',
                ),
                line=dict(color='black', width=0.5),
                opacity=0.9,
            ),
            text=[
                f"GUID: {g[:8]}…<br>"
                f"xwOBAcon: {w:.3f}<br>"
                f"Y: {y:.4f} m  X: {x:.4f} m  Z: {z:.4f} m<br>Game: {gid}"
                for g, w, y, x, z, gid in zip(
                    df_scored['MLBAM_GUID'],
                    df_scored['xwobacon_ev_la'],
                    df_scored['MISS_VECTOR_LOCAL_Y'],
                    df_scored['MISS_VECTOR_LOCAL_X'],
                    df_scored['MISS_VECTOR_LOCAL_Z'],
                    df_scored['source_game_id'],
                )
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Contact (xwOBAcon)',
        ))

    # --- Sweet spot Y reference line -------------------------------
    fig.add_trace(go.Scatter3d(
        x=[SS_ORIGIN_Y, SS_ORIGIN_Y],
        y=[-BAT_BARREL_R, BAT_BARREL_R],
        z=[SS_ORIGIN_Z, SS_ORIGIN_Z],
        mode='lines', line=dict(color='red', width=2, dash='dot'),
        name='Sweet Spot Origin Y', hoverinfo='skip',
    ))

    # --- Layout ----------------------------------------------------
    # autosize=True + no fixed width/height = Plotly fills whatever
    # container the browser gives it (critical for Notion embeds).
    fig.update_layout(
        autosize=True,
        title=dict(
            text=f'{title}<br>'
                 '<sup>Click-drag to rotate · Scroll to zoom · Hover for details</sup>',
            font=dict(size=15),
        ),
        scene=dict(
            xaxis=dict(title='Y — Bat Length (m)', autorange='reversed',
                       showbackground=True, backgroundcolor='rgba(240,240,240,0.4)'),
            yaxis=dict(title='X — In/Out Depth (m)',
                       showbackground=True, backgroundcolor='rgba(240,240,240,0.4)'),
            zaxis=dict(title='Z — Top/Bottom (m)',
                       showbackground=True, backgroundcolor='rgba(240,240,240,0.4)'),
            # Reserve right margin for the colorbar by capping the scene at 88% width
            domain=dict(x=[0.0, 0.88], y=[0.0, 1.0]),
            aspectmode='manual',
            aspectratio=dict(x=ASP, y=1, z=1),
            camera=dict(
                eye=dict(x=0.0, y=-2.8, z=1.2),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
                projection=dict(type='perspective'),
            ),
            dragmode='turntable',
        ),
        legend=dict(x=0.01, y=0.95,
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#cccccc', borderwidth=1),
        # Right margin keeps the colorbar visible; top margin gives title room
        margin=dict(l=0, r=140, b=0, t=80),
    )
    return fig


# ==============================================================
# GENERATE BOTH VERSIONS
# ==============================================================

VERSIONS = [
    # (dataframe,    filename,                          title)
    (
        df_contact,
        'bat_contact_3d_all_contacts.html',
        f'3D Ball Contact — All Contacts  ({n_games} games, n={len(df_contact)})',
    ),
    (
        df_filtered,
        'bat_contact_3d_filtered.html',
        f'3D Ball Contact — xwOBAcon ≥ {XWOBA_MIN_3D}  '
        f'({n_games} games, n={len(df_filtered)})',
    ),
]

for df_v, filename, title in VERSIONS:
    fig = build_figure(df_v, title)
    out_path = os.path.join(FIGURES_DIR, filename)
    # responsive=True tells Plotly JS to resize on window resize events,
    # which is essential for Notion's iframe to fill correctly.
    # The post_script injects CSS that makes html/body/the plotly div
    # fill 100% of the iframe viewport so there is no dead white space.
    fig.write_html(
        out_path,
        include_plotlyjs='cdn',
        full_html=True,
        config={'responsive': True},
        post_script=(
            # Force the page and plotly div to fill the full iframe viewport.
            # min-height ensures a usable size even in small default embeds;
            # 100vh fills larger ones when the user drags the block taller.
            "document.documentElement.style.cssText='height:100%;margin:0;padding:0;';"
            "document.body.style.cssText='height:100%;margin:0;padding:0;background:#fff;';"
            "var divs=document.querySelectorAll('.plotly-graph-div');"
            "divs.forEach(function(d){"
            "  d.style.width='100%';"
            "  d.style.height='100vh';"
            "  d.style.minHeight='800px';"
            "});"
            "window.dispatchEvent(new Event('resize'));"
        ),
    )
    print(f"[INFO] Saved → {filename}  ({os.path.getsize(out_path) / 1024:.0f} KB)")

print()
print("To embed in Notion:")
print("  1. Upload the HTML file(s) to OneDrive/SharePoint")
print("  2. Copy the sharing link (Anyone with the link → View)")
print("  3. In Notion: /embed → paste URL")
