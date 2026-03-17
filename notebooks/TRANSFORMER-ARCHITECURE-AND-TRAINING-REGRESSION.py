# 15: NEW TRANSFORMER ARCHITECTURE DEFINITION
# ==============================================================================
# ARCHITECTURE REVISION: PRE-NORM TRANSFORMER (MORE STABLE)
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses

# R2 Function (Ensuring it remains available for compilation)
def r2_keras(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

def build_transformer_regressor_v2(input_dim, embed_dim=64, num_heads=8, num_blocks=4, dropout=0.1):
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    
    # 1. Feature Tokenizer
    x = layers.Reshape((input_dim, 1))(inputs)
    x = layers.Conv1D(filters=embed_dim, kernel_size=1, activation=None)(x)
    
    # 2. Transformer Blocks (Switched to PRE-NORM)
    # Pre-Norm: Normalize first, then Attention. This is the modern standard (GPT-3, PaLM, etc).
    for i in range(num_blocks):
        # --- Sub-layer 1: Multi-Head Attention ---
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x) # Pre-normalization
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )(x_norm, x_norm)
        x = layers.Add()([x, attn_output]) # Skip connection
        
        # --- Sub-layer 2: Feed Forward ---
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x) # Pre-normalization
        ffn = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'), # Expanded FFN 2x
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        ffn_output = ffn(x_norm)
        x = layers.Add()([x, ffn_output]) # Skip connection

    # 3. Output Head (Revision: Flatten)
    # Instead of pooling (averaging), we Flatten so every feature has its own influence
    x = layers.LayerNormalization(epsilon=1e-6)(x) # Final norm
    x = layers.Flatten()(x) 
    
    x = layers.Dense(128, activation='gelu')(x) # Larger head for regression
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='gelu')(x)
    
    outputs = layers.Dense(2, activation='linear', name='reg_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="PreNorm_Transformer")
    return model

print("Transformer V2 Architecture (Pre-Norm) Ready.")

# 16: TRANSFORMER MODEL TRAINING & EVALUATION
# ==============================================================================
# TRAINING TRANSFORMER V2 (REVISION: SAVE PLOTS & LEARNING CURVES)
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from sklearn.metrics import r2_score

# Ensure data is available in memory
if 'X_train_scaled' in locals():
    
    # --- 0. FEATURE CORRELATION MATRIX (Optional) ---
    # Plotting correlation if df_reg exists in memory
    if 'df_reg' in locals() and 'input_cols' in locals():
        print("Creating & Saving Feature Correlation Matrix...")
        plt.figure(figsize=(12, 10))
        # Combine input features and targets for correlation analysis
        cols_to_plot = input_cols + ['log_mass', 'log_age']
        # Filter for existing columns only
        cols_to_plot = [c for c in cols_to_plot if c in df_reg.columns]
        
        df_corr = df_reg[cols_to_plot].corr()
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("Feature & Target Correlation Matrix (Transformer Data)")
        plt.tight_layout()
        plt.savefig('transformer_v2_feature_correlation.png', dpi=300) 
        plt.show()
        print("Plot saved: transformer_v2_feature_correlation.png")

    # --- 1. CONFIGURATION ---
    BATCH_SIZE = 512
    EPOCHS = 50 
    
    # --- 2. LEARNING RATE SCHEDULER (Cosine Decay + Warmup) ---
    # Starts at initial_learning_rate, ramps up to warmup_target, then decays
    total_steps = (len(X_train_scaled) // BATCH_SIZE) * EPOCHS
    warmup_steps = int(0.1 * total_steps) # First 10% of steps for warmup
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,      # Start very small (Warmup start)
        decay_steps=total_steps,
        alpha=0.01,                      # Final LR = 1% of max LR
        warmup_target=1e-3,              # Target Max LR
        warmup_steps=warmup_steps
    )
    
    # Optimizer with Weight Decay (AdamW)
    optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    # --- 3. BUILD & COMPILE ---
    print("\nBuilding Transformer V2 (Pre-Norm)...")
    if 'build_transformer_regressor_v2' in locals():
        model_trans_v2 = build_transformer_regressor_v2(
            input_dim=X_train_scaled.shape[1],
            embed_dim=64,
            num_heads=8,
            num_blocks=4,
            dropout=0.1
        )
    else:
        raise ValueError("Function 'build_transformer_regressor_v2' is not defined.")
    
    model_trans_v2.compile(
        optimizer=optimizer,
        loss=losses.Huber(delta=1.0),
        metrics=['mae', 'mse', r2_keras],
        jit_compile=True
    )

    # --- 4. TRAINING ---
    print("Starting Training with Warmup Scheduler...")
    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]

    history_trans_v2 = model_trans_v2.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )

    # --- 5. EVALUATION & VISUALIZATION ---
    print("\nEvaluating Transformer V2...")
    
    # A. Learning Curves (Essential to visualize the Warmup effect)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_trans_v2.history['loss'], label='Train Loss')
    plt.plot(history_trans_v2.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve (Huber)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_trans_v2.history['r2_keras'], label='Train R2')
    plt.plot(history_trans_v2.history['val_r2_keras'], label='Val R2')
    plt.title('R2 Score Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('transformer_v2_learning_curves.png', dpi=300) 
    plt.show()
    print("Plot saved: transformer_v2_learning_curves.png")

    # B. Prediction Plots
    preds_scaled = model_trans_v2.predict(X_test_scaled, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled) 
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    r2_mass = r2_score(y_true[:,0], preds[:,0])
    r2_age = r2_score(y_true[:,1], preds[:,1])
    
    print(f"   -> R2 Score (Mass): {r2_mass:.4f}")
    print(f"   -> R2 Score (Age) : {r2_age:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle(f"Transformer V2 (Pre-Norm + Warmup) Results", fontsize=16)

    # Log Mass Analysis
    ax[0].scatter(y_true[:,0], preds[:,0], s=1, alpha=0.3, color='tab:blue')
    ax[0].plot([y_true[:,0].min(), y_true[:,0].max()], [y_true[:,0].min(), y_true[:,0].max()], 'r--')
    ax[0].set_title(f"Log Mass ($R^2={r2_mass:.3f}$)")
    ax[0].set_xlabel("True Mass"); ax[0].set_ylabel("Predicted Mass")

    # Log Age Analysis
    ax[1].scatter(y_true[:,1], preds[:,1], s=1, alpha=0.3, color='tab:orange')
    ax[1].plot([y_true[:,1].min(), y_true[:,1].max()], [y_true[:,1].min(), y_true[:,1].max()], 'r--')
    ax[1].set_title(f"Log Age ($R^2={r2_age:.3f}$)")
    ax[1].set_xlabel("True Age"); ax[1].set_ylabel("Predicted Age")
    
    plt.tight_layout()
    plt.savefig('transformer_v2_prediction_analysis.png', dpi=300) 
    plt.show()
    print("Plot saved: transformer_v2_prediction_analysis.png")
    
    # Save Final Model
    model_trans_v2.save('best_transformer_v2_regressor.keras')
    print("Model saved: best_transformer_v2_regressor.keras")

else:
    print("Error: Please run Part 2 first to load data.")
