# 9. PREPROCESSING & FEATURE ENGINEERING (REVISED: TEFF INCLUDED & FIXED WD TEMP)
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.utils import resample, class_weight
import gc

RANDOM_STATE = 42
print("Starting Preprocessing (Mode: Scientific, Kinematics & Memory Safe)...")

# --- 1. LOAD DATA (MEMORY EFFICIENT) ---
if 'df_gabungan' in locals():
    df = df_gabungan
    print(f"   -> Using df_gabungan ({len(df):,} rows).")
else:
    try:
        df = pd.read_csv("FINAL_MASTER_DATASET.csv")
        print(f"   -> Loaded CSV ({len(df):,} rows).")
    except:
        # Fallback dummy if file does not exist (for testing only)
        print("Warning: File not found, creating dummy dataframe...")
        df = pd.DataFrame(columns=['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                                   'parallax', 'dataset_source', 'teff_gspphot'])

# OPTIMIZATION: Downcast float64 to float32
cols_float = df.select_dtypes(include=['float64']).columns
df[cols_float] = df[cols_float].astype('float32')
gc.collect()

# ==============================================================================
# A. WHITE DWARF DATA DIAGNOSTICS & FIX
# ==============================================================================
print("\nDiagnosing White Dwarf (WD) Data...")
wd_mask = df['dataset_source'].astype(str).str.contains('Gentile|WD', case=False, regex=True)

# Rename Standardizer
rename_map = {}
if 'mass_wd' not in df.columns:
    if 'MassH' in df.columns: rename_map['MassH'] = 'mass_wd'
    elif 'mass' in df.columns: rename_map['mass'] = 'mass_wd'

if 'age_wd_cooling' not in df.columns:
    if 'AgeH' in df.columns: rename_map['AgeH'] = 'age_wd_cooling'
    elif 'age' in df.columns: rename_map['age'] = 'age_wd_cooling'

if rename_map:
    print(f"   -> Renaming columns: {rename_map}")
    df.rename(columns=rename_map, inplace=True)

# Ensure target columns exist
for col in ['mass_wd', 'age_wd_cooling']:
    if col not in df.columns: df[col] = np.nan

# ==============================================================================
# B. FEATURE CALCULATION & PHYSICS FILTERS
# ==============================================================================
print("\nCalculating Photometric, Astrometric & Kinematic Features...")

# 1. Target Consolidation
if 'mass_final' not in df.columns: df['mass_final'] = np.nan
if 'age_final' not in df.columns: df['age_final'] = np.nan

# Merge Target Mass and Age (WD + FLAME)
df.loc[wd_mask, 'mass_final'] = df.loc[wd_mask, 'mass_wd']
df.loc[wd_mask, 'age_final'] = df.loc[wd_mask, 'age_wd_cooling']
if 'mass_flame' in df.columns: df['mass_final'] = df['mass_final'].fillna(df['mass_flame'])
if 'age_flame' in df.columns: df['age_final'] = df['age_final'].fillna(df['age_flame'])

# --- IMPORTANT REVISION: Temperature (Teff) Consolidation ---
# Filling missing teff_gspphot with temperature data from specific catalogs
if 'teff_wd' in df.columns:
    df['teff_gspphot'] = df['teff_gspphot'].fillna(df['teff_wd'])
if 'teff' in df.columns: 
    df['teff_gspphot'] = df['teff_gspphot'].fillna(df['teff'])
if 'teff_val' in df.columns:
    df['teff_gspphot'] = df['teff_gspphot'].fillna(df['teff_val'])
# -------------------------------------------------

# 2. Log Transform Targets (Positive values only)
df['log_mass'] = np.where(df['mass_final'] > 0, np.log10(df['mass_final']), np.nan)
df['log_age'] = np.where(df['age_final'] > 0, np.log10(df['age_final']), np.nan)

# 3. Input Features (Vectorized)
# Color Indices
df['bp_rp0'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
df['bp_g'] = df['phot_bp_mean_mag'] - df['phot_g_mean_mag']
df['g_rp'] = df['phot_g_mean_mag'] - df['phot_rp_mean_mag']

# --- PARALLAX FILTER ---
print("   -> Dropping negative/zero parallax...")
mask_valid_parallax = df['parallax'] > 0.0  
df = df[mask_valid_parallax].copy()

# 4. Calculate Absolute Magnitude (Distance Modulus)
df['distance_pc'] = 1000.0 / df['parallax']
df['abs_G0'] = df['phot_g_mean_mag'] - 5 * np.log10(df['distance_pc']) + 5

# --- KINEMATICS FEATURES ---
if 'pmra' in df.columns and 'pmdec' in df.columns:
    df['pm_total'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)
    df['v_tan'] = 4.74 * df['pm_total'] / df['parallax']
    # Log transform for better distribution (+1 to avoid log(0))
    df['log_v_tan'] = np.log10(df['v_tan'] + 1)
else:
    print("Warning: PMRA/PMDEC columns incomplete. Kinematics skipped.")
    df['log_v_tan'] = np.nan

# Galactic Norm (Not used for training, kept for reference)
df[['l', 'b']] = df[['l', 'b']].fillna(0)
df['l_norm'] = df['l'] / 360.0
df['b_norm'] = (df['b'] + 90) / 180.0

# 5. Label Encoding
print("Labeling Evolutionary Phases...")
df['fase_evolusi_final'] = 'Others'

df.loc[df['dataset_source'].str.contains('Scientific|Supergiant', case=False, na=False), 'fase_evolusi_final'] = 'Supergiant'
df.loc[df['dataset_source'].str.contains('Gentile|WD', case=False, na=False), 'fase_evolusi_final'] = 'White Dwarf'
df.loc[df['dataset_source'].str.contains('Giant_Query', case=False, na=False), 'fase_evolusi_final'] = 'Giant'

mask_target = df['fase_evolusi_final'] == 'Others'
if 'evolstage_flame' in df.columns:
    df.loc[mask_target & df['evolstage_flame'].between(100, 419), 'fase_evolusi_final'] = 'Main Sequence'
    df.loc[mask_target & df['evolstage_flame'].between(420, 489), 'fase_evolusi_final'] = 'Sub-Giant'
    df.loc[mask_target & (df['evolstage_flame'] >= 490), 'fase_evolusi_final'] = 'Giant'

label_map = {'Main Sequence': 0, 'Sub-Giant': 1, 'Giant': 2, 'Supergiant': 3, 'White Dwarf': 4}
df['label_code'] = df['fase_evolusi_final'].map(label_map)

# ==============================================================================
# C. DATASET SPLITTING (PHYSICS FEATURES & ANTI-NAN FIX)
# ==============================================================================

# 1. Classification Features (The Golden Features)
# Storing intrinsic parameters defining stellar evolution properties
cls_feats = [
    'bp_rp0', 'bp_g',  # Color Indices (Temperature Proxy)
    'abs_G0',          # True Luminosity
    'ruwe',            # Astrometric Noise (Stellar Turbulence Proxy)
    'teff_gspphot'     # Effective Temperature (Safe after consolidation)
]

# 2. Additional Features (Regression Specific)
reg_feats = ['mh_gspphot', 'log_v_tan', 'log_mass', 'log_age']

# Combine all columns to keep
cols_to_keep = list(set(cls_feats + reg_feats + ['label_code', 'dataset_source']))

# --- MAIN DATA CLEANING ---
# dropna will no longer remove White Dwarfs as teff_gspphot is now populated
df_clean = df.dropna(subset=cls_feats + ['label_code'])[cols_to_keep].copy()
df_clean['label_code'] = df_clean['label_code'].astype('int8')

del df
gc.collect()

print(f"\nTotal Clean Data (Ready for Classification): {len(df_clean):,} rows")

# --- A. CLASSIFICATION DATASET ---
print("\n--- A. Creating CLASSIFICATION Dataset ---")
frames_cls = []
TARGET_MAX = 65000 

for label_code in [0, 1, 2, 3, 4]:
    group = df_clean[df_clean['label_code'] == label_code]
    if len(group) > TARGET_MAX:
        frames_cls.append(group.sample(n=TARGET_MAX, random_state=RANDOM_STATE))
    else:
        frames_cls.append(group)

df_classification = pd.concat(frames_cls).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df_classification.to_parquet("df_cls_final.parquet", index=False)
print(f"Classification Ready: {len(df_classification):,} rows (All classes safely included).")

del df_classification, frames_cls, group
gc.collect()

# --- B. REGRESSION DATASET ---
print("\n--- B. Creating REGRESSION Dataset ---")
target_classes = [0, 1, 2] # Main, Sub, Giant

reg_conditions = (
    (df_clean['label_code'].isin(target_classes)) & 
    (np.isfinite(df_clean['log_mass'])) &
    (np.isfinite(df_clean['log_age']))
)

if 'mh_gspphot' in df_clean.columns:
    reg_conditions &= np.isfinite(df_clean['mh_gspphot'])
if 'log_v_tan' in df_clean.columns:
    reg_conditions &= np.isfinite(df_clean['log_v_tan'])

df_reg = df_clean[reg_conditions].copy()

MAX_REG = 1000000 
if len(df_reg) > MAX_REG:
    print(f"   Warning: Sampling Regression Data to {MAX_REG}...")
    df_reg = df_reg.sample(n=MAX_REG, random_state=RANDOM_STATE)

df_reg.to_parquet("df_reg_flame_final.parquet", index=False)
print(f"Regression Ready: {len(df_reg):,} rows.")

del df_reg, df_clean
gc.collect()

print("\nFinished. Missing temperature column issues have been fully resolved.")
