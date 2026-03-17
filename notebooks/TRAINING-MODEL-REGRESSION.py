# 14: MLP MODEL TRAINING & EVALUATION (REVISED: PRE-NORM + WARMUP SCHEDULER)
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os

# --- 1. DATA SETUP ---
print("Preparing Data for MLP Regressor...")
# Load Data
try:
    df_reg = pd.read_parquet("df_reg_flame_final.parquet")
    print(f"   -> Data loaded: {len(df_reg):,} rows.")
except:
    print("Warning: Parquet file not found. Ensure preprocessing (Step 8) has been run!")
    df_reg = pd.DataFrame()

if not df_reg.empty:
    # --- INPUT FEATURES CONFIGURATION ---
    # Ensuring teff_gspphot is included for physical grounding
    input_cols = [
        'bp_rp0', 'bp_g', 'g_rp', 
        'abs_G0', 
        'parallax', 'ruwe',
        'l_norm',       
        'teff_gspphot'  # Important: Effective Temperature feature
    ]
    
    # Safety Check for missing columns
    missing_cols = [c for c in input_cols if c not in df_reg.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing: {missing_cols}")
        input_cols = [c for c in input_cols if c in df_reg.columns]
    
    print(f"   -> Using {len(input_cols)} Input Features.")

    X = df_reg[input_cols].values
    y = df_reg[['log_mass', 'log_age']].values

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    # QuantileTransformer is excellent for MLP to normalize feature distributions
    scaler_X = QuantileTransformer(output_distribution='normal', random_state=42) 
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler() 
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # --- HYPERPARAMETERS & SCHEDULER ---
    BATCH_SIZE = 2048
    EPOCHS = 50 
    
    # Warmup Scheduler for better training stability
    steps_per_epoch = len(X_train_scaled) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,      # Start small (Warmup start)
        decay_steps=total_steps,
        alpha=0.01,                      # Decay to 1% of max LR
        warmup_target=1e-3,              # Max LR for MLP
        warmup_steps=int(0.1 * total_steps) # 10% Warmup period
    )

    # --- 3. BUILD & COMPILE ---
    print(f"\nBuilding & Compiling MLP Model (Pre-Norm)...")
    
    if 'build_mlp_regressor' in locals():
        model_mlp = build_mlp_regressor(
            input_dim=X_train.shape[1],
            width=512,             
            depth=6,              
            dropout_rate=0.1
        )
    else:
        raise ValueError("Function 'build_mlp_regressor' is not defined.")

    # Manual Compile with AdamW and Huber Loss
    optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    
    model_mlp.compile(
        optimizer=optimizer,
        loss=losses.Huber(delta=1.0), # Huber loss is robust against outliers
        metrics=['mae', 'mse', r2_keras],
        jit_compile=True 
    )

    # --- 4. TRAINING LOOP ---
    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]

    history_mlp = model_mlp.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )

    # --- 5. PHYSICAL ERROR ANALYSIS ---
    print("\nEvaluating Results in Physical Units...")
    
    # 1. Predictions (Scaled Log Normal)
    preds_scaled = model_mlp.predict(X_test_scaled, verbose=0)
    
    # 2. Revert to Original Log Scale (Inverse Standard Scaler)
    preds_log = scaler_y.inverse_transform(preds_scaled) 
    y_true_log = scaler_y.inverse_transform(y_test_scaled)
    
    # 3. CONVERSION TO PHYSICAL UNITS (10^x)
    pred_mass_phys = 10 ** preds_log[:, 0]
    true_mass_phys = 10 ** y_true_log[:, 0]
    
    pred_age_phys  = 10 ** preds_log[:, 1]
    true_age_phys  = 10 ** y_true_log[:, 1]
    
    # 4. Calculate Physical Errors
    mae_mass = mean_absolute_error(true_mass_phys, pred_mass_phys)
    mae_age = mean_absolute_error(true_age_phys, pred_age_phys)
    
    r2_mass = r2_score(true_mass_phys, pred_mass_phys)
    r2_age = r2_score(true_age_phys, pred_age_phys)
    
    print("-" * 40)
    print(f"PHYSICAL EVALUATION RESULTS (Pre-Norm MLP):")
    print(f"   -> Mass ($M_\odot$): MAE = {mae_mass:.3f} $M_\odot$ | R2 = {r2_mass:.4f}")
    print(f"   -> Age (Gyr)    : MAE = {mae_age:.3f} Gyr | R2 = {r2_age:.4f}")
    print("-" * 40)

    # --- VISUALIZATION (PHYSICAL UNITS) ---
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    plt.suptitle("MLP Evaluation: Physical Units ($M_{\odot}$ & Gyr)", fontsize=16, fontweight='bold')

    # Scatter Plot: Mass
    ax[0,0].scatter(true_mass_phys, pred_mass_phys, s=2, alpha=0.3, color='tab:blue')
    min_m, max_m = true_mass_phys.min(), true_mass_phys.max()
    ax[0,0].plot([min_m, max_m], [min_m, max_m], 'r--', lw=2)
    ax[0,0].set_title(f"Mass: Prediction vs Actual ($R^2={r2_mass:.3f}$)")
    ax[0,0].set_xlabel("Actual Mass ($M_{\odot}$)")
    ax[0,0].set_ylabel("Predicted Mass ($M_{\odot}$)")
    ax[0,0].set_xscale('log'); ax[0,0].set_yscale('log')

    # Scatter Plot: Age
    ax[0,1].scatter(true_age_phys, pred_age_phys, s=2, alpha=0.3, color='tab:orange')
    min_a, max_a = true_age_phys.min(), true_age_phys.max()
    ax[0,1].plot([min_a, max_a], [min_a, max_a], 'r--', lw=2)
    ax[0,1].set_title(f"Age: Prediction vs Actual ($R^2={r2_age:.3f}$)")
    ax[0,1].set_xlabel("Actual Age (Gyr)")
    ax[0,1].set_ylabel("Predicted Age (Gyr)")
    ax[0,1].set_xscale('log'); ax[0,1].set_yscale('log')

    # Residual Plot: Mass
    res_mass = pred_mass_phys - true_mass_phys
    ax[1,0].scatter(true_mass_phys, res_mass, s=2, alpha=0.3, color='purple')
    ax[1,0].axhline(0, color='black', linestyle='--')
    ax[1,0].set_title(f"Mass Residual (MAE: {mae_mass:.3f} $M_\odot$)")
    ax[1,0].set_xlabel("Actual Mass ($M_{\odot}$)")
    ax[1,0].set_ylabel("Error ($Pred - True$)")
    ax[1,0].set_xscale('log') 

    # Residual Plot: Age
    res_age = pred_age_phys - true_age_phys
    ax[1,1].scatter(true_age_phys, res_age, s=2, alpha=0.3, color='green')
    ax[1,1].axhline(0, color='black', linestyle='--')
    ax[1,1].set_title(f"Age Residual (MAE: {mae_age:.3f} Gyr)")
    ax[1,1].set_xlabel("Actual Age (Gyr)")
    ax[1,1].set_ylabel("Error ($Pred - True$)")
    ax[1,1].set_xscale('log')

    plt.tight_layout()
    plt.savefig('mlp_physical_evaluation.png', dpi=300) 
    plt.show()
    print("Plot saved: mlp_physical_evaluation.png")
    
    # Save Model
    model_mlp.save('best_mlp_regressor_full_physics.keras')
    print("Model saved: best_mlp_regressor_full_physics.keras")

