# 7. FINAL LABELING (REVISED: COMPLETE SOURCES & RELAXED BOUNDARIES)
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("Performing Labeling (Mode: 4 Sources + Relaxed FLAME)...")

if 'df_gabungan' in locals():
    df_final = df_gabungan.copy()
else:
    raise ValueError("df_gabungan not found!")

# 1. Ensure Numerical
cols_numeric = ['evolstage_flame', 'phot_g_mean_mag', 'parallax', 'bp_rp0', 'abs_G0']
for c in cols_numeric:
    if c in df_final.columns:
        df_final[c] = pd.to_numeric(df_final[c], errors='coerce')

# --- A. LABELING LOGIC BASED ON SOURCE (HIGHEST PRIORITY) ---
# We trust our query data sources.

df_final['fase_evolusi_final'] = 'Others'
source_str = df_final['dataset_source'].astype(str).str.lower()

# 1. Supergiant (Scientific Papers)
mask_sg = source_str.str.contains('supergiant|scientific|pantaleoni|messineo|hohle', regex=True)
df_final.loc[mask_sg, 'fase_evolusi_final'] = 'Supergiant'

# 2. White Dwarf (Gentile Catalog)
mask_wd = source_str.str.contains('gentile|wd', regex=True)
df_final.loc[mask_wd, 'fase_evolusi_final'] = 'White Dwarf'

# 3. Giant (Specialized Giant Query)
mask_src_giant = source_str.str.contains('giant_query', regex=True)
df_final.loc[mask_src_giant, 'fase_evolusi_final'] = 'Giant'

# --- B. FLAME LOGIC (FOR MAIN DATASET) ---
# Target: 'Others' data (usually from Gaia_DR3_Main) that haven't been labeled above
mask_target = (df_final['fase_evolusi_final'] == 'Others')

# Range Definition (100-419 MS, 420-489 Sub-Giant, >=490 Giant)
# Giant boundary set to 490 to avoid overlap with 489.

# A. MAIN SEQUENCE (Extended to 419)
mask_ms = df_final['evolstage_flame'].between(100, 419)
df_final.loc[mask_target & mask_ms, 'fase_evolusi_final'] = 'Main Sequence'

# B. SUB-GIANT (Narrowed: 420 - 489)
mask_sub = df_final['evolstage_flame'].between(420, 489)
df_final.loc[mask_target & mask_sub, 'fase_evolusi_final'] = 'Sub-Giant'

# C. GIANT (Starting from 490 / Base of RGB)
mask_giant_flame = df_final['evolstage_flame'] >= 490
df_final.loc[mask_target & mask_giant_flame, 'fase_evolusi_final'] = 'Giant'

# --- C. PHYSICS RESCUE (SAFETY NET) ---
# Filling data where FLAME code is NaN but position in HR Diagram is clear

# Rescue Giant (Bright & Red)
mask_rescue_giant = (
    (df_final['fase_evolusi_final'] == 'Others') & 
    (df_final['abs_G0'] < 3.0) & 
    (df_final['bp_rp0'] > 1.0)
)
df_final.loc[mask_rescue_giant, 'fase_evolusi_final'] = 'Giant'

# Rescue MS (Remaining faint stars)
mask_rescue_ms = (
    (df_final['fase_evolusi_final'] == 'Others') & 
    (df_final['abs_G0'] > 3.5) & 
    (df_final['abs_G0'] < 15.0) 
)
df_final.loc[mask_rescue_ms, 'fase_evolusi_final'] = 'Main Sequence'

# Cleanup: Drop remaining 'Others' (usually noise/error)
df_final = df_final[df_final['fase_evolusi_final'] != 'Others'].copy()

# --- D. REGRESSION TARGET PREPARATION (FINAL MERGE) ---
print("\nUnifying Regression Targets (Mass & Age)...")

# Initialization
df_final['mass_final'] = np.nan
df_final['age_final'] = np.nan

# 1. Extract from FLAME (Primary priority for living stars)
df_final['mass_final'] = df_final['mass_final'].fillna(df_final['mass_flame'])
df_final['age_final'] = df_final['age_final'].fillna(df_final['age_flame'])

# 2. Extract from WD (For White Dwarfs)
mask_wd_final = df_final['fase_evolusi_final'] == 'White Dwarf'
df_final.loc[mask_wd_final, 'mass_final'] = df_final.loc[mask_wd_final, 'mass_wd']
df_final.loc[mask_wd_final, 'age_final'] = df_final.loc[mask_wd_final, 'age_wd_cooling']

# Check label availability
df_final['has_labels'] = df_final['mass_final'].notnull() & df_final['age_final'].notnull()

# --- E. REPORT & VISUALIZATION ---
print("\nFINAL CLASS DISTRIBUTION REPORT:")
print("-" * 40)
print(df_final['fase_evolusi_final'].value_counts())
print("-" * 40)

# HR Diagram Visualization
plt.figure(figsize=(10, 8))
hue_order = ['White Dwarf', 'Main Sequence', 'Sub-Giant', 'Giant', 'Supergiant']
palette_dict = {
    'Main Sequence': 'tab:blue', 
    'Sub-Giant': 'orange', 
    'Giant': 'tab:red', 
    'Supergiant': 'purple', 
    'White Dwarf': 'cyan'
}

# Sampling for lighter plotting
sample_df = df_final.groupby('fase_evolusi_final').apply(lambda x: x.sample(min(len(x), 10000))).reset_index(drop=True)

sns.scatterplot(
    data=sample_df, 
    x='bp_rp0', y='abs_G0', 
    hue='fase_evolusi_final', hue_order=hue_order,
    palette=palette_dict, s=10, alpha=0.6, edgecolor='none'
)

plt.gca().invert_yaxis()
plt.title("HR Diagram: 4 Integrated Datasets")
plt.xlabel("Color ($G_{BP} - G_{RP}$)")
plt.ylabel("Absolute Magnitude ($M_G$)")
plt.legend(title='Evolutionary Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Update df_gabungan
df_gabungan = df_final.copy()

# 8. DATA VISUALIZATION: FINAL VISUALIZATION SUITE: SCIENTIFIC PUBLICATION QUALITY (CLEAN & NO GRID)
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import shutil
from astropy.coordinates import SkyCoord
from astropy import units as u
from mpl_toolkits.mplot3d import Axes3D 

# --- 0. JOURNAL AESTHETIC CONFIGURATION ---
# Setting fonts and line widths for scientific publication standards
plt.rcParams['font.family'] = 'serif'          # Using serif font (formal)
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 1.2           # Emphasize axis lines
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
# Removing grid by default
plt.rcParams['axes.grid'] = False

# --- 1. SETUP & LOAD DATA ---
output_dir = "/kaggle/working/Final_Thesis_Visualizations"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"Output Folder: {output_dir}")

print("Loading Data & Coordinates...")
if 'df_gabungan' in locals():
    df = df_gabungan.copy()
else:
    try:
        df = pd.read_csv("FINAL_MASTER_DATASET.csv")
    except:
        print("Error: Dataset not found.")
        df = pd.DataFrame()

if not df.empty:
    # Translate Labels
    label_translation = {
        'Main Sequence': 'Main Sequence', 
        'Subgiant': 'Subgiant', 
        'Red Giant': 'Red Giant', 
        'Supergiant': 'Supergiant', 
        'White Dwarf': 'White Dwarf'
    }
    if 'fase_evolusi_final' in df.columns:
        df['fase_evolusi_final'] = df['fase_evolusi_final'].replace(label_translation)

    # Coordinates calculation
    df_geo = df[df['parallax'] > 0.1].copy()
    c = SkyCoord(
        ra=df_geo['ra'].values*u.deg, 
        dec=df_geo['dec'].values*u.deg, 
        distance=(1000/df_geo['parallax'].values)*u.pc, 
        frame='icrs'
    ).galactic
    
    df_geo['X'], df_geo['Y'], df_geo['Z'] = c.cartesian.x.value, c.cartesian.y.value, c.cartesian.z.value
    df_geo['l_rad'], df_geo['b_rad'] = c.l.wrap_at(180*u.deg).radian, c.b.radian

    print(f"Data Ready: {len(df):,} rows.")

    # Scientific Palette (High Contrast & Professional)
    palette_dict = {
        'Main Sequence': 'tab:blue', 
        'Subgiant': 'tab:orange', 
        'Red Giant': 'tab:red', 
        'Supergiant': 'tab:purple', 
        'White Dwarf': 'tab:cyan'
    }
    hue_order = ['White Dwarf', 'Main Sequence', 'Subgiant', 'Red Giant', 'Supergiant']

    # ==========================================================================
    # PLOT 1: HR DIAGRAM (CLEAN)
    # ==========================================================================
    print("\n[1/11] Plotting HR Diagram...")
    

[Image of Hertzsprung-Russell diagram]

    plt.figure(figsize=(10, 8))
    sample_hr = df.groupby('fase_evolusi_final').apply(lambda x: x.sample(min(len(x), 100000))).reset_index(drop=True)
    
    sns.scatterplot(
        data=sample_hr, x='bp_rp0', y='abs_G0', 
        hue='fase_evolusi_final', hue_order=hue_order, palette=palette_dict,
        s=3, alpha=0.6, marker='o', 
        edgecolor='none', linewidth=0
    )
    plt.gca().invert_yaxis()
    plt.title("Hertzsprung-Russell Diagram", fontweight='bold')
    plt.xlabel("Color Index ($G_{BP} - G_{RP}$)")
    plt.ylabel("Absolute Magnitude ($M_G$)")
    plt.legend(title="Evolutionary Phase", markerscale=3, frameon=True, framealpha=0.9, loc='upper right')
    sns.despine() 
    plt.savefig(f"{output_dir}/1_HR_Diagram.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 2: CLASS DISTRIBUTION (CLEAN)
    # ==========================================================================
    print("[2/11] Plotting Class Distribution...")
    plt.figure(figsize=(10, 6))
    class_counts = df['fase_evolusi_final'].value_counts()
    
    ax = sns.barplot(x=class_counts.values, y=class_counts.index, palette=palette_dict, edgecolor='none', linewidth=0)
    
    plt.xscale('log')
    plt.title("Stellar Class Distribution (Log Scale)", fontweight='bold')
    plt.xlabel("Number of Stars ($N$)")
    plt.bar_label(ax.containers[0], fmt='%.0f', padding=5, fontsize=10)
    sns.despine(left=True, bottom=False)
    plt.savefig(f"{output_dir}/2_Class_Distribution.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 3: MASS HISTOGRAM (CLEAN STEP)
    # ==========================================================================
    print("[3/11] Plotting Mass Histogram...")
    plt.figure(figsize=(10, 6))
    df_mass = df[df['mass_final'] > 0].dropna(subset=['mass_final'])
    
    sns.histplot(
        data=df_mass, x='mass_final', hue='fase_evolusi_final', 
        palette=palette_dict, hue_order=hue_order, 
        element="step", fill=True, 
        linewidth=0, 
        log_scale=True, common_norm=False, alpha=0.7
    )
    plt.title("Stellar Mass Distribution", fontweight='bold')
    plt.xlabel("Mass ($M_{\odot}$)")
    plt.ylabel("Count")
    sns.despine()
    plt.savefig(f"{output_dir}/3_Mass_Distribution.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 4: AGE HISTOGRAM (CLEAN STEP)
    # ==========================================================================
    print("[4/11] Plotting Age Histogram...")
    plt.figure(figsize=(10, 6))
    df_age = df[df['age_final'] > 0.001].dropna(subset=['age_final'])
    
    sns.histplot(
        data=df_age, x='age_final', hue='fase_evolusi_final', 
        palette=palette_dict, hue_order=hue_order, 
        element="step", fill=True, 
        linewidth=0,
        log_scale=True, common_norm=False, alpha=0.7
    )
    plt.title("Stellar Age Distribution", fontweight='bold')
    plt.xlabel("Age (Gyr)")
    sns.despine()
    plt.savefig(f"{output_dir}/4_Age_Distribution.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 5: GALACTIC MAP 2D (CLEAN)
    # ==========================================================================
    print("[5/11] Plotting 2D Galactic Map...")
    
    plt.figure(figsize=(12, 8)); ax = plt.subplot(111, projection='aitoff')
    sample_map = df_geo.sample(min(len(df_geo), 2000000), random_state=42)
    colors = sample_map['mass_final'].fillna(sample_map['mass_final'].median())
    
    sc = ax.scatter(
        sample_map['l_rad'], sample_map['b_rad'], 
        c=colors, cmap='magma', s=1, alpha=0.7, 
        edgecolors='none', linewidths=0 
    )
    ax.set_title("Galactic Spatial Distribution (Mass Weighted)", fontweight='bold')
    ax.tick_params(axis='x', colors='gray', labelsize=8)
    ax.tick_params(axis='y', colors='gray', labelsize=8)
    
    cb = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
    cb.set_label("Mass ($M_{\odot}$)")
    cb.outline.set_linewidth(0)
    plt.savefig(f"{output_dir}/5_Galactic_Map_2D.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 6: CORRELATION MATRIX (CLEAN)
    # ==========================================================================
    print("[6/11] Plotting Correlation Matrix...")
    plt.figure(figsize=(9, 8))
    features = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp0', 'parallax', 'ruwe', 'abs_G0', 'mass_final', 'age_final']
    corr = df[features].corr(); mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", 
        cmap='coolwarm', vmin=-1, vmax=1, 
        linewidths=0, 
        square=True, cbar_kws={"shrink": .7, "ticks": [-1, -0.5, 0, 0.5, 1]}
    )
    plt.title("Feature Correlation Matrix", fontweight='bold')
    plt.savefig(f"{output_dir}/6_Correlation_Matrix.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 7: GALACTIC MAP 3D (CLEAN)
    # ==========================================================================
    print("[7/11] Plotting 3D Galaxy Map...")
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
    sample_3d = df_geo.sample(min(len(df_geo), 50000), random_state=42)
    
    if 'fase_evolusi_final' in sample_3d.columns:
        colors_3d = sample_3d['fase_evolusi_final'].map(palette_dict)
    
    ax.scatter(
        sample_3d['X'], sample_3d['Y'], sample_3d['Z'], 
        c=colors_3d, s=2, alpha=0.5, 
        edgecolors='none', linewidths=0 
    )
    
    # Clean 3D panels
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)
    
    ax.set_title("3D Heliocentric Distribution", fontweight='bold')
    ax.set_xlabel("X (pc)"); ax.set_ylabel("Y (pc)"); ax.set_zlabel("Z (pc)")
    ax.set_xlim(-3000, 3000); ax.set_ylim(-3000, 3000); ax.set_zlim(-1500, 1500)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=l, markersize=8) for l, c in palette_dict.items()]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    plt.savefig(f"{output_dir}/7_Galactic_Map_3D.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 9: SKY DENSITY MAP (CLEAN)
    # ==========================================================================
    print("[9/11] Plotting Sky Density Map...")
    fig = plt.figure(figsize=(14, 8)); ax = fig.add_subplot(111, projection='aitoff')
    
    hb = ax.hexbin(
        df_geo['l_rad'], df_geo['b_rad'], 
        gridsize=150, cmap='plasma', bins='log', mincnt=1, 
        edgecolors='face', linewidths=0
    )
    ax.set_title("Sky Density Map (Galactic Coordinates)", fontweight='bold', pad=20)
    ax.text(0, 0, '+ GC', ha='center', va='center', color='cyan', fontweight='bold', fontsize=12)
    ax.grid(False)
    
    cb = plt.colorbar(hb, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
    cb.set_label('Log Star Count ($log_{10} N$)')
    cb.outline.set_linewidth(0)
    plt.savefig(f"{output_dir}/9_Sky_Density.png", dpi=300, bbox_inches='tight'); plt.close()

    # ==========================================================================
    # PLOT 10 & 11: TEMP & MAG HISTOGRAM (CLEAN STEP)
    # ==========================================================================
    print("[10 & 11] Plotting Temp & Mag Histograms...")
    
    # Temp
    plt.figure(figsize=(10, 6))
    df_teff = df[(df['teff_gspphot'] > 0) & (df['teff_gspphot'] < 30000)]
    sns.histplot(
        data=df_teff, x='teff_gspphot', hue='fase_evolusi_final', 
        palette=palette_dict, hue_order=hue_order, 
        element="step", fill=True, linewidth=0, 
        log_scale=(False, True), common_norm=False, bins=50, alpha=0.7
    )
    plt.title("Effective Temperature Distribution", fontweight='bold')
    plt.xlabel("Teff (K)"); plt.xlim(0, 20000)
    sns.despine()
    plt.savefig(f"{output_dir}/10_Temp_Distribution.png", dpi=300, bbox_inches='tight'); plt.close()

    # Mag
    plt.figure(figsize=(10, 6))
    df_abs = df[df['abs_G0'].notnull()]
    sns.histplot(
        data=df_abs, x='abs_G0', hue='fase_evolusi_final', 
        palette=palette_dict, hue_order=hue_order, 
        element="step", fill=True, linewidth=0, 
        log_scale=(False, True), common_norm=False, bins=50, alpha=0.7
    )
    plt.title("Luminosity Distribution", fontweight='bold')
    plt.xlabel("Absolute Magnitude $G$"); plt.gca().invert_xaxis()
    sns.despine()
    plt.savefig(f"{output_dir}/11_Mag_Distribution.png", dpi=300, bbox_inches='tight'); plt.close()

    # ZIP ALL
    shutil.make_archive("/kaggle/working/Final_Scientific_Visualizations", 'zip', output_dir)
    print("\nDONE! Please download 'Final_Scientific_Visualizations.zip'.")

else:
    print("Error: Dataset is empty.")
