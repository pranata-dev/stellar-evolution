# 0. INSTALLATION AND IMPORT LIBRARIES
# ==============================================================================

# Install main libraries
!pip install astroquery -q
!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow -q

# Import general libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time

# Import astroquery GAIA
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

# Keras & ML tools
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Reshape,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Enable async mode for astroquery Gaia (so large queries don't timeout)
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # default to EDR3
Gaia.ROW_LIMIT = -1  # fetch all data without limit
Vizier.ROW_LIMIT = -1

print("All libraries successfully installed and imported.")

# 1. MAIN SEQUENCE AND SUBGIANT STAR QUERY

print("Fetching main data (Main Sequence, Subgiant, Giant)...")

query_utama_final = """
SELECT
    g.source_id, g.ra, g.dec,
    g.l, g.b,
    g.parallax, g.parallax_over_error,
    g.pmra, g.pmdec,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
    g.phot_bp_rp_excess_factor,
    g.ruwe, g.phot_variable_flag,

    -- FLAME & GSP-Phot PARAMETERS
    df.teff_gspphot, df.logg_gspphot, df.mh_gspphot,
    df.ag_gspphot, df.ebpminrp_gspphot,
    df.mg_gspphot,
    df.mass_flame, df.age_flame, df.evolstage_flame,
    df.flags_flame
FROM
    gaiadr3.gaia_source AS g
JOIN
    gaiadr3.astrophysical_parameters AS df
    ON g.source_id = df.source_id
WHERE
    g.parallax > 0.5 AND g.parallax_over_error > 10    AND g.phot_g_mean_mag < 18
    AND g.phot_bp_mean_mag IS NOT NULL
    AND g.phot_rp_mean_mag IS NOT NULL
    AND g.phot_g_mean_mag < 18
    AND g.ruwe < 1.4
    AND df.flags_flame LIKE '0%'
    AND df.evolstage_flame IS NOT NULL
    AND df.mass_flame IS NOT NULL
    AND df.age_flame IS NOT NULL
    AND df.teff_gspphot BETWEEN 2500 AND 50000
    AND df.logg_gspphot BETWEEN -0.5 AND 6.0
    AND df.mh_gspphot > -5
    AND df.mg_gspphot BETWEEN -5 AND 15
    AND df.lum_flame IS NOT NULL
    AND g.random_index < 50000000
"""
try:
    job_utama = Gaia.launch_job_async(query_utama_final)
    df_utama = job_utama.get_results().to_pandas()
    print(f"Main data acquisition completed. Received {len(df_utama)} rows.")

    if 'l' in df_utama.columns:
      print(" -> Galactic columns (l, b) successfully fetched.")
except Exception as e:
    print(f"An error occurred: {e}")

# 2. GIANTS STAR QUERY

print("[Additional Query] Fetching SPECIFIC GIANT data...")

query_giant = """
SELECT
    g.source_id, g.ra, g.dec,
    g.l, g.b,
    g.parallax, g.parallax_over_error,
    g.pmra, g.pmdec,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
    g.phot_bp_rp_excess_factor,
    g.ruwe, g.phot_variable_flag,

    -- FLAME PARAMETERS
    df.teff_gspphot, df.logg_gspphot, df.mh_gspphot,
    df.ag_gspphot, df.ebpminrp_gspphot,
    df.mg_gspphot,
    df.mass_flame, df.age_flame, df.evolstage_flame,
    df.flags_flame
FROM
    gaiadr3.gaia_source AS g
JOIN
    gaiadr3.astrophysical_parameters AS df
    ON g.source_id = df.source_id
WHERE
    g.parallax > 0.1             -- Minimum distance (can be further than MS)
    AND g.parallax_over_error > 5 
    AND g.phot_g_mean_mag < 18.5 -- Slightly dimmer allowed
    AND g.phot_bp_mean_mag IS NOT NULL
    AND g.phot_rp_mean_mag IS NOT NULL
    
    -- RELAXED FILTER SPECIFIC TO GIANTS
    AND g.ruwe < 3.0             -- Giants are "shaky", so RUWE < 3.0 is okay
    -- WE REMOVE 'flags_flame' filter so "difficult" Giants are still included
    
    -- ONLY FETCH GIANTS (Base RGB to AGB)
    AND df.evolstage_flame BETWEEN 490 AND 1290
    
    -- Ensure targets exist
    AND df.mass_flame IS NOT NULL
    AND df.age_flame IS NOT NULL
    
    -- Random Limit (Fetching 50,000 Giants is enough for balancing)
    AND g.random_index < 20000000 
"""

try:
    job_giant = Gaia.launch_job_async(query_giant)
    df_giant_raw = job_giant.get_results().to_pandas()
    
    # Initial Labeling
    df_giant_raw['dataset_source'] = 'Gaia_DR3_Giant_Query'
    df_giant_raw['fase_evolusi'] = 'Giant' 
    
    print(f"SUCCESS: Obtained {len(df_giant_raw):,} Pure Giant Stars.")
    print(f"    Evolstage Range: {df_giant_raw['evolstage_flame'].min()} - {df_giant_raw['evolstage_flame'].max()}")

except Exception as e:
    print(f"Giant Query Failed: {e}")
    df_giant_raw = pd.DataFrame()

# 3. WHITE DWARF DATA ACQUISITION (FINAL - WITHOUT AGE TARGET)
# ==============================================================================
from astroquery.vizier import Vizier
import pandas as pd
import numpy as np

print("INFO: Fetching White Dwarf Catalog (Gentile Fusillo et al. 2021)...")

v = Vizier(row_limit=150000, columns=["**"])
v.TIMEOUT = 120 

try:
    catalogs = v.get_catalogs("J/MNRAS/508/3877")
    
    target_catalog = None
    for i, table in enumerate(catalogs):
        temp_df = table.to_pandas()
        if 'MassH' in temp_df.columns:
            target_catalog = temp_df
            print(f"SUCCESS: Target table found with {len(temp_df)} raw rows.")
            break
            
    if target_catalog is not None:
        df_wd_raw = target_catalog
    else:
        print("ERROR: No valid table found.")
        df_wd_raw = pd.DataFrame()

except Exception as e:
    print(f"ERROR: Data acquisition failed: {e}")
    df_wd_raw = pd.DataFrame()

# ==============================================================================
# 4. WD DATA HARMONIZATION
# ==============================================================================
def process_gentile_data_final(df):
    if df.empty: return df
    data = df.copy()

    data = data.loc[:, ~data.columns.duplicated()].copy()

    # 1. WD Probability Filter (PWD > 0.95)
    pwd_cols = [c for c in data.columns if 'PWD' in c.upper()]
    if pwd_cols:
        data = data[data[pwd_cols[0]] > 0.95].copy()
        print("INFO: WD probability filter > 0.95 completed.")

    # 2. Fallback logic ONLY for Mass
    if 'MassH' in data.columns:
        data['mass_wd'] = data['MassH']
        if 'MassHe' in data.columns:
            data['mass_wd'] = data['mass_wd'].fillna(data['MassHe'])
        if 'Massmix' in data.columns:
            data['mass_wd'] = data['mass_wd'].fillna(data['Massmix'])

    # 3. Extraction of Basic Photometric Features
    target_mapping = {
        'ra': ['RAJ2000', 'RA_ICRS', 'RA'],
        'dec': ['DEJ2000', 'DE_ICRS', 'DE'],
        'parallax': ['PLX'],
        'parallax_error': ['E_PLX'],
        'phot_g_mean_mag': ['GMAG'],
        'phot_bp_mean_mag': ['BPMAG'],
        'phot_rp_mean_mag': ['RPMAG'],
        'teff_wd': ['TEFFH', 'TEFF'],
        'logg_wd': ['LOGGH', 'LOGG'],
        'source_id': ['SOURCE', 'EDR3', 'GAIAEDR3']
    }

    rename_dict = {}
    for target, candidates in target_mapping.items():
        for cand in candidates:
            matching_cols = [c for c in data.columns if c.upper() == cand]
            if matching_cols:
                rename_dict[matching_cols[0]] = target
                break 
                
    data.rename(columns=rename_dict, inplace=True)
    data = data.loc[:, ~data.columns.duplicated()].copy()

    # 4. Feature Engineering
    if 'parallax' in data.columns and 'parallax_error' in data.columns:
        data['parallax_over_error'] = data['parallax'] / data['parallax_error']
    
    if 'phot_bp_mean_mag' in data.columns and 'phot_rp_mean_mag' in data.columns:
        data['bp_rp'] = data['phot_bp_mean_mag'] - data['phot_rp_mean_mag']

    if 'phot_g_mean_mag' in data.columns and 'parallax' in data.columns:
        data = data[data['parallax'] > 0].copy()
        data['abs_G0'] = data['phot_g_mean_mag'] + 5 * np.log10(data['parallax']/1000.0) + 5

    # 5. Labeling
    data['fase_evolusi'] = 'White Dwarf'
    data['dataset_source'] = 'GentileFusillo2021'

    # 6. Final Target Validation (Mass Only)
    if 'mass_wd' in data.columns:
        before = len(data)
        data = data.dropna(subset=['mass_wd'])
        print("SUCCESS: Mass target successfully isolated.")
        print(f"STATISTICS: Final Clean Data: {len(data):,} rows ({before - len(data):,} NaN removed).")
    else:
        print("WARNING: Failed to process mass column.")

    return data

# --- EXECUTION ---
if not df_wd_raw.empty:
    df_wd_clean = process_gentile_data_final(df_wd_raw)
    
    if not df_wd_clean.empty:
        df_wd_clean.to_csv("gaia_white_dwarfs_clean.csv", index=False)
        print(f"FILE: Saved gaia_white_dwarfs_clean.csv ({len(df_wd_clean)} rows)")

# 5. SCIENTIFIC PAPER CROSS-MATCH (REVISED: EXTRA RICH CATALOGS)
# ==============================================================================
import pandas as pd
import numpy as np
import time
import gc
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy import units as u

print("STARTING SUPERGIANT HUNT (Target: Extra Rich Catalogs)...")

# ------------------------------------------------------------------------------
# 1. TARGET CATALOG DEFINITIONS (MAXIMUM OVERDRIVE VERSION)
# ------------------------------------------------------------------------------
target_papers = [
    # --- BLUE SUPERGIANTS / OB STARS (COMPLETE FLEET) ---
    {"name": "Pantaleoni (ALS II)", "id": "J/MNRAS/504/2968", "type": "Blue Supergiant/OB"},
    {"name": "GOSC (O-Stars)", "id": "V/145", "type": "Blue Supergiant/OB"}, 
    {"name": "Hohle (OB Stars)", "id": "J/AN/331/349", "type": "Blue Supergiant/OB"},
    {"name": "Reed (OB Photometry)", "id": "V/125", "type": "Blue Supergiant/OB"},
    {"name": "LAMOST DR5 (OB)", "id": "J/ApJS/249/31", "type": "Blue Supergiant/OB"},
    {"name": "LAMOST DR1 (OB)", "id": "J/ApJS/215/1", "type": "Blue Supergiant/OB"},
    {"name": "LAMOST OB (Chen)", "id": "J/A+A/625/A104", "type": "Blue Supergiant/OB"},

    # --- RED SUPERGIANT SPECIALISTS (NEW ADDITIONS) ---
    {"name": "Messineo (RSG)", "id": "J/AJ/158/20", "type": "Red Supergiant"},
    {"name": "Zhang (RSG)", "id": "J/ApJ/889/33/table1", "type": "Red Supergiant"}, # ~1,500 RSG
    {"name": "Chen (Galactic RSG)", "id": "J/A+A/671/A139/table1", "type": "Red Supergiant"}, # ~3,000 RSG (Latest)
    {"name": "Massey (MW RSG)", "id": "J/ApJS/247/9/table4", "type": "Red Supergiant"}, # Massive RSG Catalog
    {"name": "Yang (LMC/SMC RSG)", "id": "J/A+A/629/A91/table3", "type": "Red Supergiant"} # RSG in Magellanic Clouds
]

# ------------------------------------------------------------------------------
# 2. VIZIER DATA ACQUISITION LOOP
# ------------------------------------------------------------------------------
print("\n[1/3] Fetching Coordinates from Vizier (Filtering Supergiants)...")

v = Vizier(columns=["**"], row_limit=50000) 
v.TIMEOUT = 30 # Avoid hanging

reference_stars = []

for paper in target_papers:
    print(f"   -> Attempting: {paper['name']}...", end=" ")
    try:
        cats = v.get_catalogs(paper['id'])
        if not cats:
            print("FAILED: Not found.")
            continue
            
        # Select largest table
        df_res = cats[0].to_pandas()

        # --- SPECIFIC FILTER LOGIC ---
        if paper['id'] == "V/50":
            # Bright Star Catalog: Take class I, II, or giant classes (G/K/M with 'I' or 'II')
            if 'SpType' in df_res.columns:
                df_res = df_res[df_res['SpType'].str.contains(r'\bI\b|Iab|Ia|Ib|\bII\b', na=False, regex=True)]
        
        if df_res.empty:
            print("FAILED: Empty data after filtering.")
            continue

        print(f"SUCCESS: Obtained {len(df_res)} rows.")
        
        # --- COORDINATE COLUMN DETECTION ---
        cols = df_res.columns.tolist()
        c_ra = next((c for c in cols if c in ['RAJ2000', 'RA_ICRS', 'RAdeg', '_RA', 'RA']), None)
        c_dec = next((c for c in cols if c in ['DEJ2000', 'DE_ICRS', 'DEdeg', '_DE', 'DE']), None)

        if c_ra and c_dec:
            temp = pd.DataFrame()
            # Handle "HH MM SS" string format vs decimal float
            sample = str(df_res[c_ra].iloc[0])
            if " " in sample:
                coords = SkyCoord(df_res[c_ra].astype(str), df_res[c_dec].astype(str), 
                                  unit=(u.hourangle, u.deg), frame='icrs')
                temp['ra_ref'] = coords.ra.deg
                temp['dec_ref'] = coords.dec.deg
            else:
                temp['ra_ref'] = pd.to_numeric(df_res[c_ra], errors='coerce')
                temp['dec_ref'] = pd.to_numeric(df_res[c_dec], errors='coerce')

            temp['paper_source'] = paper['name']
            temp['star_type'] = paper['type']
            reference_stars.append(temp.dropna(subset=['ra_ref']))
            
    except Exception as e:
        print(f"ERROR (Timeout/Server Down): {str(e)[:50]}...")

# Concatenate references
if reference_stars:
    df_refs = pd.concat(reference_stars, ignore_index=True)
    print(f"\nTotal Reference Database: {len(df_refs):,} Stars.")
else:
    df_refs = pd.DataFrame()
    print("\nFAILED: Could not pull any references.")

# ------------------------------------------------------------------------------
# 3. GAIA DR3 CROSS-MATCH (SAFE BATCH MODE)
# ------------------------------------------------------------------------------
if not df_refs.empty:
    print("\n[2/3] Cross-Matching to Gaia DR3 (Batch size: 10)...")
    gaia_results = []
    targets = df_refs.to_dict('records')
    batch_size = 10 # Avoid Error 400/500 on Gaia TAP Server
    
    start_time = time.time()
    for i in range(0, len(targets), batch_size):
        batch = targets[i : i + batch_size]
        conditions = [f"1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {t['ra_ref']:.5f}, {t['dec_ref']:.5f}, 0.00028))" 
                      for t in batch if pd.notna(t['ra_ref'])]
        
        if not conditions: continue
        
        query = f"""
        SELECT TOP 100 source_id, ra, dec, parallax, ruwe,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
               teff_gspphot, logg_gspphot, mh_gspphot
        FROM gaiadr3.gaia_source
        WHERE ({" OR ".join(conditions)}) AND phot_g_mean_mag < 18
        """
        
        try:
            job = Gaia.launch_job(query)
            res = job.get_results().to_pandas()
            
            if not res.empty:
                for _, g_row in res.iterrows():
                    # Find closest match in batch
                    dists = [np.sqrt((g_row['ra'] - b['ra_ref'])**2 + (g_row['dec'] - b['dec_ref'])**2) for b in batch]
                    idx_min = np.argmin(dists)
                    if dists[idx_min] < 0.0004:
                        d = g_row.to_dict()
                        d.update({
                            'paper_source': batch[idx_min]['paper_source'],
                            'star_type': batch[idx_min]['star_type']
                        })
                        gaia_results.append(d)
                        
            if (i // batch_size) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"\r   Progress: {i}/{len(targets)} | Match: {len(gaia_results)} | Time: {elapsed:.1f}s", end="")
                gc.collect() # Clean memory every 50 batches
                
        except Exception as e:
            # If occasional failure occurs, skip the batch to keep the process running
            time.sleep(1) 
            continue

    print(f"\n\nDONE! Found {len(gaia_results):,} valid matches.")

    # 4. SAVE & CLEANUP
    if gaia_results:
        df_final = pd.DataFrame(gaia_results).drop_duplicates(subset='source_id')
        # Remove non-physical parallax data for Supergiants
        df_final = df_final[df_final['parallax'] > 0]
        
        # Calculate Absolute Magnitude (Distance Modulus)
        df_final['abs_G0'] = df_final['phot_g_mean_mag'] - 5 * np.log10(1000.0/df_final['parallax']) + 5
        
        print("\nSupergiant Source Distribution:")
        print(df_final['paper_source'].value_counts())
        
        df_final.to_csv("gaia_scientific_supergiants_fixed.csv", index=False)
        print(f"File saved: gaia_scientific_supergiants_fixed.csv")
else:
    print("FAILED: No reference data successfully pulled.")

# 6. FINAL DATA MERGING (REVISED: 4 SOURCES INCLUDING GIANT QUERY)
# ==============================================================================
import pandas as pd
import numpy as np

print("INFO: Merging 4 Datasets (Main + Giant + Scientific SG + WD)...")

frames = []

# --- A. MAIN DATASET (Main Sequence/Subgiant) ---
if 'df_utama' in locals() and not df_utama.empty:
    df_u = df_utama.copy()
    df_u['dataset_source'] = 'Gaia_DR3_Main'
    # Focused on MS & Subgiant, specialized query handles Giants
    # (Optional: Filter < 490 for cleaner data, though deduplication handles this later)
    
    # Calculate Features
    df_u['bp_rp0'] = df_u['phot_bp_mean_mag'] - df_u['phot_rp_mean_mag']
    with np.errstate(divide='ignore', invalid='ignore'):
        df_u['abs_G0'] = df_u['phot_g_mean_mag'] - 5*np.log10(1000.0/df_u['parallax']) + 5
        
    df_u['mass_wd'] = np.nan
    df_u['age_wd_cooling'] = np.nan
    
    frames.append(df_u)
    print(f"   -> [Main] {len(df_u):,} stars ready.")

# --- B. SPECIALIZED GIANT DATASET (Additional Query) ---
if 'df_giant_raw' in locals() and not df_giant_raw.empty:
    df_g = df_giant_raw.copy()
    df_g['dataset_source'] = 'Gaia_DR3_Giant_Query'
    # Temporary labeling (will be refined in Step 6)
    
    # Calculate Features
    df_g['bp_rp0'] = df_g['phot_bp_mean_mag'] - df_g['phot_rp_mean_mag']
    with np.errstate(divide='ignore', invalid='ignore'):
        df_g['abs_G0'] = df_g['phot_g_mean_mag'] - 5*np.log10(1000.0/df_g['parallax']) + 5
    
    df_g['mass_wd'] = np.nan
    df_g['age_wd_cooling'] = np.nan
    
    frames.append(df_g)
    print(f"   -> [Giant Query] {len(df_g):,} stars ready.")

# --- C. SUPERGIANT DATASET (Scientific - High Priority) ---
try:
    if 'df_final' in locals() and 'paper_source' in df_final.columns:
        df_sg_sci = df_final.copy()
    else:
        df_sg_sci = pd.read_csv("gaia_scientific_supergiants_fixed.csv")
        
    df_sg_sci['dataset_source'] = 'Scientific_Paper_SG'
    
    # Dummy Code for SG
    df_sg_sci['evolstage_flame'] = 400 
    
    if 'mass_ref' in df_sg_sci.columns:
        df_sg_sci.rename(columns={'mass_ref': 'mass_flame'}, inplace=True)
    
    if 'bp_rp0' not in df_sg_sci.columns:
        if 'bp_rp' in df_sg_sci.columns:
             df_sg_sci['bp_rp0'] = df_sg_sci['bp_rp']
        else:
             df_sg_sci['bp_rp0'] = df_sg_sci['phot_bp_mean_mag'] - df_sg_sci['phot_rp_mean_mag']
    
    if 'abs_G0' not in df_sg_sci.columns and 'G_abs' in df_sg_sci.columns:
         df_sg_sci['abs_G0'] = df_sg_sci['G_abs']
         
    if 'age_flame' not in df_sg_sci.columns:
        df_sg_sci['age_flame'] = np.nan
        
    frames.append(df_sg_sci)
    print(f"   -> [Scientific SG] {len(df_sg_sci):,} stars (Gold Standard).")

except Exception as e:
    print(f"WARNING: Scientific SG dataset not found ({e}).")

# --- D. WHITE DWARF DATASET (Gentile Fusillo) ---
if 'df_wd_clean' in locals() and not df_wd_clean.empty:
    df_w = df_wd_clean.copy()
    
    if 'RUWE' in df_w.columns:
        df_w.rename(columns={'RUWE': 'ruwe'}, inplace=True)
        
    df_w['mass_flame'] = np.nan
    df_w['age_flame'] = np.nan
    df_w['evolstage_flame'] = 500 
    
    # Labeling & Column Mapping
    df_w['dataset_source'] = 'GentileFusillo2021'
    
    if 'bp_rp0' not in df_w.columns:
        if 'bp_rp' in df_w.columns:
            df_w['bp_rp0'] = df_w['bp_rp']
        elif 'phot_bp_mean_mag' in df_w.columns and 'phot_rp_mean_mag' in df_w.columns:
            df_w['bp_rp0'] = df_w['phot_bp_mean_mag'] - df_w['phot_rp_mean_mag']
            
    if 'abs_G0' not in df_w.columns and 'phot_g_mean_mag' in df_w.columns and 'parallax' in df_w.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df_w['abs_G0'] = df_w['phot_g_mean_mag'] - 5*np.log10(1000.0/df_w['parallax']) + 5
    
    frames.append(df_w)
    print(f"   -> [White Dwarf] {len(df_w):,} stars ready.")


# --- E. EXECUTE MERGING & SMART DEDUPLICATION ---
if frames:
    df_gabungan = pd.concat(frames, ignore_index=True)
    
    # Deduplication Priority (Lower number = higher priority)
    # 1. Scientific Paper -> Most Valid
    # 2. White Dwarf (Specialized Catalog)
    # 3. Giant Query (Specialized Catalog)
    # 4. Main Dataset (General)
    source_priority = {
        'Scientific_Paper_SG': 0, 
        'Vizier_Pantaleoni2021': 0,
        'Vizier_Hohle2010': 0,
        'GentileFusillo2021': 1, 
        'Gaia_DR3_Giant_Query': 2,
        'Gaia_DR3_Main': 3
    }
    
    df_gabungan['prio'] = df_gabungan['dataset_source'].map(source_priority).fillna(99)
    df_gabungan.sort_values('prio', inplace=True)
    
    # Remove Duplicate Source IDs
    before_dedup = len(df_gabungan)
    df_gabungan.drop_duplicates(subset=['source_id'], keep='first', inplace=True)
    after_dedup = len(df_gabungan)
    
    print(f"\nINFO: Deduplication Finished: {before_dedup - after_dedup} duplicates removed.")

    # Final Quality Filter
    df_gabungan = df_gabungan[
        (df_gabungan['parallax'] > 0) &
        (df_gabungan['bp_rp0'].notnull()) &
        (df_gabungan['abs_G0'].notnull())
    ].copy()
    
    # Shuffle
    df_gabungan = df_gabungan.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nSUCCESS: FINAL MASTER DATASET READY!")
    print(f"INFO: Total Rows: {len(df_gabungan):,}")
    print("-" * 30)
    print("INFO: Data Source Distribution:")
    print(df_gabungan['dataset_source'].value_counts())
    print("-" * 30)
    
    # Check Final Monsters (> 10 M_sun)
    monsters = len(df_gabungan[df_gabungan['mass_flame'] > 10.0])
    print(f"INFO: Number of True Monsters (> 10 M_sun): {monsters}")
    
    # Save to Master CSV
    df_gabungan.to_csv("FINAL_MASTER_DATASET.csv", index=False)

else:
    print("ERROR: Failed to create df_gabungan.")
