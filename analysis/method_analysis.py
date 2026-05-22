import glob
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import os
import sys

# Set Seaborn theme for nicer plots
sns.set_theme(style="whitegrid", palette="muted")

#==============================================
# Load data — all games from data/discrete/
#==============================================
# PATH is the file's parent directory
PATH = os.path.dirname(os.path.abspath(__file__))

# Glob every *_discrete.csv in the discrete subfolder
discrete_pattern = os.path.join(PATH, 'data/discrete/*_discrete.csv')
discrete_files = sorted(glob.glob(discrete_pattern))

if not discrete_files:
    raise FileNotFoundError(
        f"[ERROR] No discrete CSVs found at: {discrete_pattern}\n"
        "Ensure data/discrete/ contains *_discrete.csv files."
    )

print(f"[INFO] Loading {len(discrete_files)} discrete game files:")
game_dfs = []
for fpath in discrete_files:
    game_id = os.path.basename(fpath).replace('_discrete.csv', '')
    gdf = pd.read_csv(fpath)
    gdf['source_game_id'] = game_id   # data lineage tag
    game_dfs.append(gdf)
    print(f"       {os.path.basename(fpath)}  ({len(gdf)} rows)")

# Concatenate all games into one analysis DataFrame
df = pd.concat(game_dfs, ignore_index=True)
N_GAMES = df['source_game_id'].nunique()
print(f"[INFO] Combined: {len(df)} total rows across {N_GAMES} games")

# print(df.columns)

#==============================================
# define variables   
#==============================================
# 80% sweet spot md
md_80_global_x = df['MISS_VECTOR_GLOBAL_K80_X']
md_80_global_y = df['MISS_VECTOR_GLOBAL_K80_Y']
md_80_global_z = df['MISS_VECTOR_GLOBAL_K80_Z']
md_80_global = df['MISSED_DISTANCE_GLOBAL_K80']

md_80_local_x = df['MISS_VECTOR_LOCAL_K80_X']
md_80_local_y = df['MISS_VECTOR_LOCAL_K80_Y']
md_80_local_z = df['MISS_VECTOR_LOCAL_K80_Z']
md_80_local = df['MISSED_DISTANCE_LOCAL_K80']

# 80% sweet spot max velocity
max_md_80_velocity_global_x = df['MAX_MISS_VELOCITY_K80_GLOBAL_X']
max_md_80_velocity_global_y = df['MAX_MISS_VELOCITY_K80_GLOBAL_Y']
max_md_80_velocity_global_z = df['MAX_MISS_VELOCITY_K80_GLOBAL_Z']
max_md_80_speed_global = df['MAX_MISS_SPEED_K80_GLOBAL']

max_md_80_velocity_local_x = df['MAX_MISS_VELOCITY_K80_LOCAL_X']
max_md_80_velocity_local_y = df['MAX_MISS_VELOCITY_K80_LOCAL_Y']
max_md_80_velocity_local_z = df['MAX_MISS_VELOCITY_K80_LOCAL_Z']
max_md_80_speed_local = df['MAX_MISS_SPEED_K80_LOCAL']

tmin__global_k80 = df['T_MIN_GLOBAL_K80']
tmin__local_k80 = df['T_MIN_LOCAL_K80']

#==============================================
# data dictionary   
#==============================================
md_dict = {
    '80': {
        'tmin': {
            'global': {
                'tmin': tmin__global_k80
            },
            'local': {
                'tmin': tmin__local_k80
            }
        },
        'position': {
            'global': {
                'x': df['MISS_VECTOR_GLOBAL_K80_X'],
                'y': df['MISS_VECTOR_GLOBAL_K80_Y'],
                'z': df['MISS_VECTOR_GLOBAL_K80_Z'],
                'mag': df['MISSED_DISTANCE_GLOBAL_K80']
            },
            'local': {
                'x': df['MISS_VECTOR_LOCAL_K80_X'],
                'y': df['MISS_VECTOR_LOCAL_K80_Y'],
                'z': df['MISS_VECTOR_LOCAL_K80_Z'],
                'mag': df['MISSED_DISTANCE_LOCAL_K80']
            }
        },
        'velocity': {
            'global': {
                'x': df['MAX_MISS_VELOCITY_K80_GLOBAL_X'],
                'y': df['MAX_MISS_VELOCITY_K80_GLOBAL_Y'],
                'z': df['MAX_MISS_VELOCITY_K80_GLOBAL_Z'],
                'mag': df['MAX_MISS_SPEED_K80_GLOBAL']
            },
            'local': {
                'x': df['MAX_MISS_VELOCITY_K80_LOCAL_X'],
                'y': df['MAX_MISS_VELOCITY_K80_LOCAL_Y'],
                'z': df['MAX_MISS_VELOCITY_K80_LOCAL_Z'],
                'mag': df['MAX_MISS_SPEED_K80_LOCAL']
            }
        }
    }
}

#==============================================
# Plotting Functions
#==============================================

def plot_residual_distribution(residuals: pd.Series, title: str, xlabel: str, save_path: str) -> None:
    """
    Plot the distribution of calculated residuals using Seaborn and save to file.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color='indigo', edgecolor='black', alpha=0.6)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() # Close figure to free up memory

def one_to_one_plot(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
    """
    Plot a one-to-one scatterplot using Seaborn, with a dynamic 1:1 reference line, and save to file.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, alpha=0.6, s=50, edgecolor=None)

    # Calculate min and max across both sets for a dynamic 1:1 line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() # Close figure to free up memory

#==============================================
# Execute Loops (Calculate Residuals & Plot)
#==============================================

for k in md_dict.keys(): # Loop through sweet spots (e.g., '80')
    # Create the target directory for the current sweet spot if it doesn't exist
    save_dir = os.path.join(PATH, 'figures', 'method')
    os.makedirs(save_dir, exist_ok=True)

    for metric in md_dict[k].keys(): # Loop through 'tmin', 'position', and 'velocity'
        # Initialize a new dictionary space for residuals
        md_dict[k][metric]['residuals'] = {}

        # Set the correct axes to loop through based on the metric type
        if metric == 'tmin':
            axes = ['tmin']
        else:
            axes = ['x', 'y', 'z', 'mag']

        for local_axis in axes: # Loop through dynamic axes based on metric
            
            # Map the local axis to the correct global axis
            if local_axis == 'x':
                global_axis = 'z'
            elif local_axis == 'y':
                global_axis = 'x'
            elif local_axis == 'z':
                global_axis = 'y'
            else:
                global_axis = local_axis # Fallback for 'mag' and 'tmin'

            global_series = md_dict[k][metric]['global'][global_axis]
            
            # Flip the Global Y axis values to match Local Z orientation
            if global_axis == 'y':
                global_series = global_series * -1

            local_series  = md_dict[k][metric]['local'][local_axis]

            # Calculate Residuals (Local - Global) and store back into dict
            residuals = local_series - global_series
            md_dict[k][metric]['residuals'][local_axis] = residuals

            # String formatting for readable titles and filenames
            metric_label = metric.capitalize()
            local_label = local_axis.upper()
            
            # Add indication to the global label if it was flipped
            if global_axis == 'y':
                global_label = f"-{global_axis.upper()}"
            else:
                global_label = global_axis.upper()

            # Define file paths for saving
            file_prefix = f"{metric}_Local-{local_label}_Global-{global_axis.upper()}"
            path_1to1 = os.path.join(save_dir, f"{file_prefix}_1to1.png")
            path_res = os.path.join(save_dir, f"{file_prefix}_residual.png")

            # 1. Generate and Save One-to-One Plot
            title_1to1 = (
                f"Sweet Spot {k}% | {metric_label} - Global ({global_label}) vs Local ({local_label})"
                f"  [{N_GAMES} games, n={len(global_series.dropna())}]"
            )
            xlabel_1to1 = f"Global {metric_label} {global_label}"
            ylabel_1to1 = f"Local {metric_label} {local_label}"

            one_to_one_plot(global_series, local_series, title_1to1, xlabel_1to1, ylabel_1to1, path_1to1)

            # 2. Generate and Save Residual Distribution Plot
            title_res = (
                f"Sweet Spot {k}% | {metric_label} Residuals [Local ({local_label}) - Global ({global_label})]"
                f"  [{N_GAMES} games, n={len(residuals.dropna())}]"
            )
            xlabel_res = f"Residual {metric_label} (Local {local_label} - Global {global_label})"

            plot_residual_distribution(residuals, title_res, xlabel_res, path_res)

print("All plots generated and saved successfully!")
