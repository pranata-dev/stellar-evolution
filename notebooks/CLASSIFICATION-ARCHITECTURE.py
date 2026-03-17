# 10. CLASSIFICATION MODEL ARCHITECTURE DEFINITION (REVISED: PRE-NORM, CAPACITY DIET, & REGULARIZATION)
# ==============================================================================
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

print("Building Neural Network Model Architectures (Mode: Classification)...")

# ------------------------------------------------------------------------------
# MODEL 1: ResNet-MLP (Deep Residual Network) - CLASSIFICATION
# ------------------------------------------------------------------------------
# REVISION: Default capacity reduced (width=128, depth=3) for inputs with < 10 features
def build_mlp_classifier(input_dim, output_dim=5, width=128, depth=3, dropout_rate=0.2):
    inputs = layers.Input(shape=(input_dim,), name='input_features')

    x = layers.Dense(width, activation='linear')(inputs) 
    
    # --- Residual Blocks Loop (Pre-Norm Style) ---
    for i in range(depth):
        shortcut = x 
        
        x_norm = layers.LayerNormalization()(x) 
        
        branch = layers.Dense(width, activation='gelu')(x_norm)
        branch = layers.Dropout(dropout_rate)(branch)
        branch = layers.Dense(width, activation='linear')(branch)
        
        x = layers.Add()([shortcut, branch]) 

    x = layers.LayerNormalization()(x)

    # REVISION: Added L2 Regularization to prevent weight overfitting
    x = layers.Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)

    outputs = layers.Dense(output_dim, activation='softmax', name='class_output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_MLP_Classifier")
    return model

# ------------------------------------------------------------------------------
# MODEL 2: TABULAR TRANSFORMER - CLASSIFICATION
# ------------------------------------------------------------------------------
# REVISION: embed_dim reduced to 32 because the number of features is very small (5 features)
def build_transformer_classifier(input_dim, output_dim=5, embed_dim=32, num_heads=4, num_blocks=3, dropout=0.1):
    inputs = layers.Input(shape=(input_dim,), name='input_features')

    # 1. Feature Tokenizer (Shared Weights via Conv1D)
    x = layers.Reshape((input_dim, 1))(inputs)
    x = layers.Conv1D(embed_dim, kernel_size=1, activation=None)(x) 

    # 2. Transformer Blocks (Pre-Norm Style)
    for _ in range(num_blocks):
        # Attention
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads, 
            dropout=dropout
        )(x_norm, x_norm)
        x = layers.Add()([x, attn_output]) 

        # Feed Forward
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = models.Sequential([
            layers.Dense(embed_dim * 2, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])
        x = layers.Add()([x, ffn(x_norm)])

    # 3. Head (Flatten)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x) 
    
    # REVISION: Added L2 Regularization
    x = layers.Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim, activation='softmax', name='class_output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Transformer_Classifier")
    return model

print("Classification Architectures Ready (Pre-Norm, Slimmed Features, L2 Regularizer).")
