# 13: REGRESSION MODEL ARCHITECTURE DEFINITION (REVISED: PRE-NORM & EXTERNAL COMPILE)
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, models
import numpy as np

# --- CUSTOM METRIC: R-SQUARE (R2) ---
def r2_keras(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

# ------------------------------------------------------------------------------
# MODEL A: PRE-NORM RESNET-MLP REGRESSOR
# Revision: Post-Norm -> Pre-Norm, Removed Compile
# ------------------------------------------------------------------------------
def build_mlp_regressor(input_dim, width=512, depth=6, dropout_rate=0.1):
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    
    # 1. Linear Projection (To match dimensions for the blocks)
    x = layers.Dense(width, activation='linear')(inputs)
    
    # 2. Residual Blocks (Pre-Norm Style)
    for i in range(depth):
        shortcut = x
        
        # Normalize FIRST
        x_norm = layers.LayerNormalization()(x)
        
        # The Shop (Dense -> Dropout -> Dense)
        x = layers.Dense(width, activation='gelu')(x_norm)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(width, activation='linear')(x)
        
        # Merge (Add)
        x = layers.Add()([shortcut, x]) 
        
    # 3. Output Head
    x = layers.LayerNormalization()(x) # Final Norm is mandatory in Pre-Norm
    x = layers.Dense(64, activation='gelu')(x)
    
    outputs = layers.Dense(2, activation='linear', name='reg_output')(x)
    
    # REMOVE model.compile to allow external Scheduler injection
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet_MLP_Regressor")
    return model

# ------------------------------------------------------------------------------
# MODEL B: PRE-NORM FT-TRANSFORMER REGRESSOR
# Revision: Pooling -> Flatten, Post-Norm -> Pre-Norm
# ------------------------------------------------------------------------------
def build_transformer_regressor(input_dim, embed_dim=64, num_heads=4, num_blocks=3, dropout=0.1):
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    
    # 1. Feature Tokenizer
    x = layers.Reshape((input_dim, 1))(inputs)
    x = layers.Conv1D(filters=embed_dim, kernel_size=1, activation=None)(x)
    
    # 2. Transformer Blocks (Pre-Norm Style)
    for i in range(num_blocks):
        # --- Attention Sub-layer ---
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x) # Pre-Norm
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )(x_norm, x_norm)
        x = layers.Add()([x, attn_output])
        
        # --- Feed Forward Sub-layer ---
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x) # Pre-Norm
        ffn = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        ffn_output = ffn(x_norm)
        x = layers.Add()([x, ffn_output])
        
    # 3. Head (Flatten to preserve specific features)
    x = layers.LayerNormalization(epsilon=1e-6)(x) # Final Norm
    x = layers.Flatten()(x) # Replace GlobalAvgPooling with Flatten
    
    x = layers.Dense(64, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(2, activation='linear', name='reg_output')(x)
    
    # REMOVE model.compile
    model = keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Regressor")
    return model

print("Regression Architectures Ready (Pre-Norm Version).")
