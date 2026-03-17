# 11. CLASSIFICATION TRAINING: PRE-NORM RESNET-MLP (REVISED: CAPACITY DIET & REGULARIZATION)
# ==============================================================================
try:
    import keras_tuner as kt
except ImportError:
    !pip install keras_tuner -q
    import keras_tuner as kt

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os, shutil

print("Starting Pre-Norm ResNet Tuning Pipeline (Mode: Stable & Warmup)...")

# --- 1. LOAD DATA ---
if 'df_classification' in locals():
    df_use = df_classification.copy()
else:
    try:
        df_use = pd.read_parquet("df_cls_final.parquet")
        print("   -> Loaded from Parquet.")
    except:
        raise ValueError("Classification data (df_classification) not found!")

# Feature Selection
input_cols = [
    'bp_rp0', 'bp_g', 
    'abs_G0', 'ruwe', 
    'teff_gspphot'
]

available_cols = [c for c in input_cols if c in df_use.columns]
target_col = 'label_code'

X = df_use[available_cols].values
y = df_use[target_col].values

# --- FEATURE CORRELATION MATRIX ---
print("Creating Feature Correlation Matrix...")
plt.figure(figsize=(10, 8))
corr_matrix = df_use[available_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig('feature_correlation_matrix.png', dpi=300)
plt.show()

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling using QuantileTransformer (Maps data to a normal distribution)
print("   -> Scaling Data...")
scaler = QuantileTransformer(output_distribution='normal')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class Weights (Handles imbalanced stellar populations)
from sklearn.utils import class_weight
classes = np.unique(y_train)
weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print("Class Weights Applied:", class_weight_dict)

# --- SCHEDULER CONFIGURATION ---
BATCH_SIZE = 512
EPOCHS_SEARCH = 20
EPOCHS_FINAL = 50 

# --- PRE-NORM MODEL DEFINITION (REVISED: REDUCED CAPACITY FOR SMALL FEATURE SET) ---
def build_resnet_prenorm_tuner(hp):
    inputs = layers.Input(shape=(X_train.shape[1],))
    
    # Tuning Width (32 - 128) - Kept slim because input features < 10
    width = hp.Int('width', min_value=32, max_value=128, step=32) 
    dropout_rate = hp.Float('dropout_rate', 0.2, 0.5, step=0.1) 
    
    # Initial Projection
    x = layers.Dense(width, activation='linear')(inputs) 
    
    # --- RESIDUAL BLOCKS (PRE-NORM STYLE) ---
    for i in range(hp.Int('num_blocks', 1, 3)): 
        shortcut = x
        
        x_norm = layers.LayerNormalization()(x) 
        
        branch = layers.Dense(width, activation='gelu')(x_norm)
        branch = layers.Dropout(dropout_rate)(branch)
        branch = layers.Dense(width, activation='linear')(branch)
        
        x = layers.Add()([shortcut, branch]) 

    # --- OUTPUT HEAD ---
    x = layers.LayerNormalization()(x) 
    
    # L2 Regularizer added to the final dense layer to prevent overfitting
    x = layers.Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
    outputs = layers.Dense(5, activation='softmax')(x)

    # --- OPTIMIZER WITH WARMUP ---
    steps_per_epoch = len(X_train_scaled) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS_SEARCH 
    
    lr_max = hp.Choice('learning_rate', [1e-3, 5e-4])
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,      
        decay_steps=total_steps,
        alpha=0.01,
        warmup_target=lr_max,            
        warmup_steps=int(0.1 * total_steps)
    )

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# --- HYPERPARAMETER TUNING ---
if os.path.exists('tuning_dir/gaia_resnet_v2'): 
    shutil.rmtree('tuning_dir/gaia_resnet_v2')

tuner = kt.Hyperband(
    build_resnet_prenorm_tuner, 
    objective='val_accuracy', 
    max_epochs=EPOCHS_SEARCH, 
    factor=3, 
    directory='tuning_dir', 
    project_name='gaia_resnet_v2'
)

print("\nSearching for Best Hyperparameters...")
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(
    X_train_scaled, y_train, 
    epochs=EPOCHS_SEARCH, 
    validation_split=0.2, 
    callbacks=[stop_early], 
    batch_size=BATCH_SIZE, 
    class_weight=class_weight_dict, 
    verbose=1
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best HPs Found: Width={best_hps.get('width')}, Blocks={best_hps.get('num_blocks')}, LR={best_hps.get('learning_rate')}")

# --- RETRAINING (FULL SCHEDULER) ---
print("\nRetraining Best Model (Pre-Norm ResNet)...")

model_final = build_resnet_prenorm_tuner(best_hps)

total_steps_final = (len(X_train_scaled) // BATCH_SIZE) * EPOCHS_FINAL
lr_final = best_hps.get('learning_rate')

new_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-6, 
    decay_steps=total_steps_final,
    alpha=0.01,
    warmup_target=lr_final,
    warmup_steps=int(0.1 * total_steps_final)
)

model_final.compile(
    optimizer=optimizers.AdamW(learning_rate=new_schedule, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

history = model_final.fit(
    X_train_scaled, y_train, 
    epochs=EPOCHS_FINAL, 
    validation_split=0.2, 
    batch_size=BATCH_SIZE, 
    class_weight=class_weight_dict, 
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], 
    verbose=1
)

# --- EVALUATION ---
# A. Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve (Pre-Norm)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curves_prenorm.png', dpi=300)
plt.show()

# B. Confusion Matrix & Report
y_pred_probs = model_final.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)
class_labels = ['Main Sequence', 'Sub-Giant', 'Giant', 'Supergiant', 'White Dwarf']

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Normalized Confusion Matrix (Pre-Norm ResNet)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_prenorm.png', dpi=300)
plt.show()

model_final.save('best_resnet_prenorm_classifier.keras')
print("Model saved successfully: best_resnet_prenorm_classifier.keras")

# 12. FT-TRANSFORMER CLASSIFICATION TRAINING (FINAL REVISION: L2 REGULARIZER & PRE-NORM)
# ==============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import os, shutil

try:
    import keras_tuner as kt
except ImportError:
    !pip install keras_tuner -q
    import keras_tuner as kt

print("Starting FT-Transformer Pipeline (Mode: Pre-Norm, Warmup, & L2 Regularized)...")

# --- DATA PREPARATION ---
if 'df_classification' in locals():
    df_use = df_classification.copy()
else:
    try:
        df_use = pd.read_parquet("df_cls_final.parquet")
    except:
        raise ValueError("Data not found! Please run Preprocessing first.")

# Golden Features (Intrinsic Physics)
input_cols = ['bp_rp0', 'bp_g', 'abs_G0', 'ruwe', 'teff_gspphot']
X = df_use[input_cols].values
y = df_use['label_code'].values 

# --- FEATURE CORRELATION (TRANSFORMER) ---
print("Creating Feature Correlation Matrix (Transformer)...")
plt.figure(figsize=(10, 8))
corr_matrix = df_use[input_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Feature Correlation Matrix (Transformer)")
plt.tight_layout()
plt.savefig('transformer_feature_correlation.png', dpi=300)
plt.show()

# Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = QuantileTransformer(output_distribution='normal')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

NUM_FEATURES = X_train_scaled.shape[1]
NUM_CLASSES = 5

# Imbalanced Class Handling
classes = np.unique(y_train)
weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print("Class Weights Applied:", class_weight_dict)

# --- SCHEDULER CONFIGURATION ---
BATCH_SIZE = 512
EPOCHS_SEARCH = 15
EPOCHS_FINAL = 50

# --- FT-TRANSFORMER ARCHITECTURE DEFINITION (REVISED: PRE-NORM & L2 REGULARIZER) ---
def build_ft_transformer_prenorm(hp):
    inputs = layers.Input(shape=(NUM_FEATURES,))
    
    # Feature Tokenizer
    x = layers.Reshape((NUM_FEATURES, 1))(inputs)
    embed_dim = hp.Int('embed_dim', min_value=32, max_value=64, step=32)
    x = layers.Conv1D(filters=embed_dim, kernel_size=1, activation=None)(x) 
    
    # --- Transformer Blocks (Pre-Norm Style) ---
    for i in range(hp.Int('num_blocks', 1, 3)):
        # 1. Attention Block
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x) # Pre-normalization
        
        num_heads = hp.Int(f'num_heads_{i}', 2, 4, step=2)
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads, 
            dropout=hp.Float(f'attn_dropout_{i}', 0.0, 0.2, step=0.1)
        )(x_norm, x_norm)
        
        x = layers.Add()([x, attn_output]) # Residual connection
        
        # 2. Feed Forward Block
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x) # Pre-normalization
        
        ff_dim = embed_dim * 2 
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"), 
            layers.Dropout(hp.Float(f'ffn_dropout_{i}', 0.0, 0.2, step=0.1)), 
            layers.Dense(embed_dim)
        ])
        ffn_output = ffn(x_norm)
        
        x = layers.Add()([x, ffn_output]) # Residual connection

    # --- Head (Flatten & Final Norm) ---
    x = layers.LayerNormalization(epsilon=1e-6)(x) # Crucial Final Norm
    x = layers.Flatten()(x) 
    
    # L2 Regularization added to the final Dense layer
    x = layers.Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    # --- Optimizer with Warmup ---
    steps_per_epoch = len(X_train_scaled) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS_SEARCH
    
    lr_max = hp.Choice('learning_rate', values=[1e-3, 5e-4])
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=total_steps,
        alpha=0.01,
        warmup_target=lr_max,
        warmup_steps=int(0.1 * total_steps)
    )
    
    weight_decay = hp.Choice('weight_decay', values=[1e-4, 1e-5])
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# --- HYPERPARAMETER TUNING ---
if os.path.exists('tuning_dir/gaia_transformer_v2'): 
    shutil.rmtree('tuning_dir/gaia_transformer_v2')

tuner = kt.Hyperband(
    build_ft_transformer_prenorm, 
    objective='val_accuracy', 
    max_epochs=EPOCHS_SEARCH, 
    factor=3, 
    directory='tuning_dir', 
    project_name='gaia_transformer_v2', 
    overwrite=True
)

print("\nTuning FT-Transformer (Pre-Norm & L2 Reg)...")
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(
    X_train_scaled, y_train, 
    epochs=EPOCHS_SEARCH, 
    batch_size=BATCH_SIZE, 
    validation_split=0.2, 
    callbacks=[stop_early], 
    class_weight=class_weight_dict, 
    verbose=1
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best HPs Found: Embed Dim={best_hps.get('embed_dim')}, Blocks={best_hps.get('num_blocks')}, LR={best_hps.get('learning_rate')}")

# --- FINAL TRAINING (FULL SCHEDULER) ---
print("\nTraining Final Model (Pre-Norm + Full Warmup + L2 Reg)...")

# Build best model manually
best_transformer = build_ft_transformer_prenorm(best_hps)

# Update Scheduler for 50 epoch duration
total_steps_final = (len(X_train_scaled) // BATCH_SIZE) * EPOCHS_FINAL
lr_final_val = best_hps.get('learning_rate')

new_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=total_steps_final,
    alpha=0.01,
    warmup_target=lr_final_val,
    warmup_steps=int(0.1 * total_steps_final)
)

weight_decay_final = best_hps.get('weight_decay')

# Re-compile
best_transformer.compile(
    optimizer=optimizers.AdamW(learning_rate=new_schedule, weight_decay=weight_decay_final), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

history = best_transformer.fit(
    X_train_scaled, y_train, 
    epochs=EPOCHS_FINAL, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test_scaled, y_test), 
    class_weight=class_weight_dict,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], 
    verbose=1
)

# --- EVALUATION & VISUALIZATION ---
print("\nTransformer Evaluation Results:")
loss, acc = best_transformer.evaluate(X_test_scaled, y_test, batch_size=512)
print(f"   Accuracy: {acc:.2%}")

# A. Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve (Pre-Norm & L2)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig('transformer_learning_curves_prenorm.png', dpi=300)
plt.show()

# B. Confusion Matrix
y_pred = np.argmax(best_transformer.predict(X_test_scaled, batch_size=512), axis=1)
labels = ['Main Sequence', 'Sub-Giant', 'Giant', 'Supergiant', 'White Dwarf']

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Normalized Confusion Matrix - Pre-Norm Transformer')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('transformer_confusion_matrix_prenorm.png', dpi=300)
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))
best_transformer.save("best_transformer_classifier_prenorm.keras")
print("Model saved successfully: best_transformer_classifier_prenorm.keras")
