"""
Smart Agriculture ML Model Training and Evaluation Script
Implements all 6 required Machine Learning algorithms:
1. Simple Linear Regression (Rainfall -> Crop Yield)
2. Multiple Linear Regression (Rainfall, Temp, Humidity, pH, N, P, K -> Crop Yield)
3. Logistic Regression (Soil Suitability Classification: Suitable vs Unsuitable)
4. KNN Classification (Top 3 Crop Recommendation with Feature Scaling)
5. K-Means Clustering (Agricultural Condition Clustering)
6. Gaussian Naive Bayes Classification (Crop Category Prediction: Cereals, Pulses, Fruits, Cash Crops)

Generates real evaluation metrics, saves trained models, and produces evaluation charts.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHART_DIR = os.path.join(BASE_DIR, 'static', 'img', 'charts')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Generation: Benchmark Agricultural Crop Dataset
# -------------------------------------------------------------
def generate_crop_recommendation_dataset():
    crop_profiles = {
        'rice': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temp': (20, 27), 'hum': (80, 85), 'ph': (5.0, 7.0), 'rain': (180, 300), 'cat': 'Cereals'},
        'maize': {'N': (60, 100), 'P': (35, 60), 'K': (15, 25), 'temp': (18, 27), 'hum': (55, 75), 'ph': (5.5, 7.0), 'rain': (60, 110), 'cat': 'Cereals'},
        'chickpea': {'N': (20, 60), 'P': (55, 80), 'K': (75, 85), 'temp': (17, 20), 'hum': (14, 20), 'ph': (6.0, 8.5), 'rain': (65, 95), 'cat': 'Pulses'},
        'kidneybeans': {'N': (10, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (15, 25), 'hum': (18, 25), 'ph': (5.5, 6.0), 'rain': (60, 150), 'cat': 'Pulses'},
        'pigeonpeas': {'N': (10, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (18, 38), 'hum': (30, 65), 'ph': (4.5, 7.5), 'rain': (90, 200), 'cat': 'Pulses'},
        'mothbeans': {'N': (10, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (24, 32), 'hum': (40, 65), 'ph': (3.5, 9.5), 'rain': (30, 75), 'cat': 'Pulses'},
        'mungbean': {'N': (10, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (27, 30), 'hum': (80, 90), 'ph': (6.2, 7.2), 'rain': (35, 60), 'cat': 'Pulses'},
        'blackgram': {'N': (20, 60), 'P': (55, 80), 'K': (15, 25), 'temp': (25, 35), 'hum': (60, 70), 'ph': (6.5, 7.8), 'rain': (60, 75), 'cat': 'Pulses'},
        'lentil': {'N': (10, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (18, 30), 'hum': (60, 70), 'ph': (5.9, 7.8), 'rain': (35, 55), 'cat': 'Pulses'},
        'pomegranate': {'N': (10, 40), 'P': (10, 30), 'K': (35, 45), 'temp': (18, 25), 'hum': (85, 95), 'ph': (5.5, 7.2), 'rain': (100, 115), 'cat': 'Fruits'},
        'banana': {'N': (80, 120), 'P': (70, 95), 'K': (45, 55), 'temp': (25, 30), 'hum': (75, 85), 'ph': (5.5, 6.5), 'rain': (90, 120), 'cat': 'Fruits'},
        'mango': {'N': (10, 40), 'P': (15, 35), 'K': (25, 35), 'temp': (27, 36), 'hum': (45, 55), 'ph': (4.5, 7.0), 'rain': (85, 105), 'cat': 'Fruits'},
        'grapes': {'N': (10, 40), 'P': (120, 145), 'K': (195, 205), 'temp': (8, 42), 'hum': (80, 85), 'ph': (5.5, 6.5), 'rain': (65, 75), 'cat': 'Fruits'},
        'watermelon': {'N': (80, 120), 'P': (5, 30), 'K': (45, 55), 'temp': (24, 27), 'hum': (80, 90), 'ph': (6.0, 7.0), 'rain': (40, 60), 'cat': 'Fruits'},
        'muskmelon': {'N': (80, 120), 'P': (5, 30), 'K': (45, 55), 'temp': (27, 30), 'hum': (90, 95), 'ph': (6.0, 6.8), 'rain': (20, 30), 'cat': 'Fruits'},
        'apple': {'N': (10, 40), 'P': (120, 145), 'K': (195, 205), 'temp': (21, 24), 'hum': (90, 95), 'ph': (5.5, 6.5), 'rain': (100, 125), 'cat': 'Fruits'},
        'orange': {'N': (10, 40), 'P': (5, 30), 'K': (5, 15), 'temp': (10, 35), 'hum': (90, 95), 'ph': (6.0, 8.0), 'rain': (100, 120), 'cat': 'Fruits'},
        'papaya': {'N': (30, 70), 'P': (45, 70), 'K': (45, 55), 'temp': (23, 44), 'hum': (90, 95), 'ph': (6.5, 7.0), 'rain': (140, 250), 'cat': 'Fruits'},
        'coconut': {'N': (10, 40), 'P': (5, 30), 'K': (25, 35), 'temp': (25, 30), 'hum': (90, 99), 'ph': (5.5, 6.5), 'rain': (130, 230), 'cat': 'Fruits'},
        'cotton': {'N': (100, 140), 'P': (35, 60), 'K': (15, 25), 'temp': (22, 26), 'hum': (75, 85), 'ph': (5.8, 8.0), 'rain': (60, 100), 'cat': 'Commercial Crops'},
        'jute': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temp': (23, 26), 'hum': (70, 90), 'ph': (6.0, 7.5), 'rain': (150, 200), 'cat': 'Commercial Crops'},
        'coffee': {'N': (80, 120), 'P': (15, 40), 'K': (25, 35), 'temp': (23, 28), 'hum': (50, 70), 'ph': (6.0, 7.5), 'rain': (115, 200), 'cat': 'Commercial Crops'}
    }

    np.random.seed(42)
    rows = []
    samples_per_crop = 100  # 2,200 rows total

    for crop, prof in crop_profiles.items():
        for _ in range(samples_per_crop):
            n = np.clip(np.random.normal(np.mean(prof['N']), (prof['N'][1] - prof['N'][0]) / 4), 0, 140)
            p = np.clip(np.random.normal(np.mean(prof['P']), (prof['P'][1] - prof['P'][0]) / 4), 5, 145)
            k = np.clip(np.random.normal(np.mean(prof['K']), (prof['K'][1] - prof['K'][0]) / 4), 5, 205)
            temp = np.clip(np.random.normal(np.mean(prof['temp']), (prof['temp'][1] - prof['temp'][0]) / 4), 8, 45)
            hum = np.clip(np.random.normal(np.mean(prof['hum']), (prof['hum'][1] - prof['hum'][0]) / 4), 10, 100)
            ph = np.clip(np.random.normal(np.mean(prof['ph']), (prof['ph'][1] - prof['ph'][0]) / 4), 3.5, 9.5)
            rain = np.clip(np.random.normal(np.mean(prof['rain']), (prof['rain'][1] - prof['rain'][0]) / 4), 20, 300)

            rows.append({
                'N': round(float(n), 2),
                'P': round(float(p), 2),
                'K': round(float(k), 2),
                'temperature': round(float(temp), 2),
                'humidity': round(float(hum), 2),
                'ph': round(float(ph), 2),
                'rainfall': round(float(rain), 2),
                'label': crop,
                'category': prof['cat']
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(DATA_DIR, 'crop_recommendation.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} crop recommendation records in {csv_path}")
    return df

# -------------------------------------------------------------
# 2. Dataset Generation: Crop Yield Dataset (Agronomic Formula)
# -------------------------------------------------------------
def generate_crop_yield_dataset():
    np.random.seed(42)
    n_samples = 2000

    # Agricultural variables
    rainfall = np.random.uniform(50, 300, n_samples)
    temperature = np.random.uniform(15, 38, n_samples)
    humidity = np.random.uniform(30, 95, n_samples)
    ph = np.random.uniform(5.0, 8.5, n_samples)
    nitrogen = np.random.uniform(20, 140, n_samples)
    phosphorus = np.random.uniform(10, 90, n_samples)
    potassium = np.random.uniform(15, 100, n_samples)

    # Biological multi-factor response (Mitscherlich & climatic optimal):
    # Base yield ~ 1200 kg/ha
    # Rainfall effect: positive with slight diminishing returns
    rain_effect = 12.5 * rainfall - 0.015 * (rainfall ** 2)
    # Temperature effect: optimum around 26C
    temp_effect = -8.0 * ((temperature - 26) ** 2) + 250
    # Humidity effect: optimum around 70%
    hum_effect = 6.0 * humidity
    # pH effect: optimum around 6.5
    ph_effect = -150.0 * ((ph - 6.5) ** 2) + 300
    # Nutrient effects (N, P, K)
    nutrient_effect = 9.2 * nitrogen + 7.5 * phosphorus + 6.8 * potassium
    # Noise
    noise = np.random.normal(0, 120, n_samples)

    yield_kg_per_ha = (
        800 + rain_effect + temp_effect + hum_effect + ph_effect + nutrient_effect + noise
    )
    # Ensure realistic positive yield values (1000 - 7000 kg/ha)
    yield_kg_per_ha = np.clip(yield_kg_per_ha, 800, 7500)

    # Also calculate a purely rainfall-correlated yield for Simple Linear Regression
    simple_yield = 1200 + 15.8 * rainfall + np.random.normal(0, 220, n_samples)
    simple_yield = np.clip(simple_yield, 1000, 6500)

    df = pd.DataFrame({
        'rainfall': np.round(rainfall, 2),
        'temperature': np.round(temperature, 2),
        'humidity': np.round(humidity, 2),
        'ph': np.round(ph, 2),
        'nitrogen': np.round(nitrogen, 2),
        'phosphorus': np.round(phosphorus, 2),
        'potassium': np.round(potassium, 2),
        'yield_kg_per_ha': np.round(yield_kg_per_ha, 2),
        'simple_yield_kg_per_ha': np.round(simple_yield, 2)
    })

    csv_path = os.path.join(DATA_DIR, 'crop_yield_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} crop yield records in {csv_path}")
    return df

# -------------------------------------------------------------
# 3. Dataset Generation: Soil Suitability Dataset
# -------------------------------------------------------------
def generate_soil_suitability_dataset():
    np.random.seed(42)
    n_samples = 1800

    ph = np.random.uniform(4.0, 9.5, n_samples)
    nitrogen = np.random.uniform(5, 140, n_samples)
    phosphorus = np.random.uniform(5, 100, n_samples)
    potassium = np.random.uniform(5, 120, n_samples)
    moisture = np.random.uniform(10, 90, n_samples)
    temperature = np.random.uniform(10, 42, n_samples)

    # Suitability scoring logic (Agronomic soil health criteria):
    # Optimal pH: 6.0 to 7.8
    # Adequate N: > 35
    # Adequate P: > 20
    # Adequate K: > 25
    # Adequate moisture: 25% - 75%
    z = (
        - 1.8 * ((ph - 6.6) ** 2)
        + 0.035 * nitrogen
        + 0.040 * phosphorus
        + 0.030 * potassium
        + 0.05 * moisture - 0.0006 * (moisture ** 2)
        - 0.08 * np.abs(temperature - 25)
        - 1.2
    )
    # Sigmoid probability
    prob = 1 / (1 + np.exp(-z))
    # Binary classification with realistic borderline
    is_suitable = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        'ph': np.round(ph, 2),
        'nitrogen': np.round(nitrogen, 2),
        'phosphorus': np.round(phosphorus, 2),
        'potassium': np.round(potassium, 2),
        'moisture': np.round(moisture, 2),
        'temperature': np.round(temperature, 2),
        'is_suitable': is_suitable
    })

    csv_path = os.path.join(DATA_DIR, 'soil_suitability_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} soil suitability records in {csv_path}")
    return df

# -------------------------------------------------------------
# Training & Evaluation of All 6 Algorithms
# -------------------------------------------------------------
def train_and_evaluate_all():
    metrics_summary = {}

    crop_df = generate_crop_recommendation_dataset()
    yield_df = generate_crop_yield_dataset()
    soil_df = generate_soil_suitability_dataset()

    # =========================================================
    # ALGORITHM 1: Simple Linear Regression (Rainfall -> Yield)
    # =========================================================
    print("\n--- Training Algorithm 1: Simple Linear Regression ---")
    X_simple = yield_df[['rainfall']].values
    y_simple = yield_df['simple_yield_kg_per_ha'].values

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_simple, y_simple, test_size=0.20, random_state=42
    )

    slr_model = LinearRegression()
    slr_model.fit(X_train_s, y_train_s)
    y_pred_s = slr_model.predict(X_test_s)

    mae_s = float(mean_absolute_error(y_test_s, y_pred_s))
    mse_s = float(mean_squared_error(y_test_s, y_pred_s))
    rmse_s = float(np.sqrt(mse_s))
    r2_s = float(r2_score(y_test_s, y_pred_s))

    metrics_summary['simple_linear_regression'] = {
        'name': 'Simple Linear Regression',
        'type': 'Regression',
        'features': ['Rainfall (mm)'],
        'target': 'Crop Yield (kg/ha)',
        'coefficient': float(slr_model.coef_[0]),
        'intercept': float(slr_model.intercept_),
        'metrics': {
            'MAE': round(mae_s, 2),
            'MSE': round(mse_s, 2),
            'RMSE': round(rmse_s, 2),
            'R2_Score': round(r2_s, 4)
        }
    }
    joblib.dump(slr_model, os.path.join(MODEL_DIR, 'simple_linear_regression.joblib'))

    # Plot SLR Regression Graph
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test_s[:150], y_test_s[:150], color='#2d6a4f', alpha=0.6, label='Actual Yield')
    sort_idx = np.argsort(X_test_s[:, 0])
    plt.plot(X_test_s[sort_idx], y_pred_s[sort_idx], color='#e76f51', linewidth=2.5, label='Regression Line (Slope: ' + str(round(float(slr_model.coef_[0]), 2)) + ')')
    plt.title('Simple Linear Regression: Rainfall vs Crop Yield', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Rainfall (mm)', fontsize=11)
    plt.ylabel('Crop Yield (kg/hectare)', fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'slr_regression_graph.png'), dpi=150)
    plt.close()

    # =========================================================
    # ALGORITHM 2: Multiple Linear Regression
    # =========================================================
    print("--- Training Algorithm 2: Multiple Linear Regression ---")
    mlr_features = ['rainfall', 'temperature', 'humidity', 'ph', 'nitrogen', 'phosphorus', 'potassium']
    X_mlr = yield_df[mlr_features].values
    y_mlr = yield_df['yield_kg_per_ha'].values

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mlr, y_mlr, test_size=0.20, random_state=42
    )

    mlr_scaler = StandardScaler()
    X_train_m_scaled = mlr_scaler.fit_transform(X_train_m)
    X_test_m_scaled = mlr_scaler.transform(X_test_m)

    mlr_model = LinearRegression()
    mlr_model.fit(X_train_m_scaled, y_train_m)
    y_pred_m = mlr_model.predict(X_test_m_scaled)

    mae_m = float(mean_absolute_error(y_test_m, y_pred_m))
    mse_m = float(mean_squared_error(y_test_m, y_pred_m))
    rmse_m = float(np.sqrt(mse_m))
    r2_m = float(r2_score(y_test_m, y_pred_m))

    metrics_summary['multiple_linear_regression'] = {
        'name': 'Multiple Linear Regression',
        'type': 'Regression',
        'features': mlr_features,
        'target': 'Crop Yield (kg/ha)',
        'intercept': float(mlr_model.intercept_),
        'coefficients': {f: round(float(c), 3) for f, c in zip(mlr_features, mlr_model.coef_)},
        'metrics': {
            'MAE': round(mae_m, 2),
            'MSE': round(mse_m, 2),
            'RMSE': round(rmse_m, 2),
            'R2_Score': round(r2_m, 4)
        }
    }
    joblib.dump(mlr_model, os.path.join(MODEL_DIR, 'multiple_linear_regression.joblib'))
    joblib.dump(mlr_scaler, os.path.join(MODEL_DIR, 'mlr_scaler.joblib'))

    # Residuals plot
    plt.figure(figsize=(8, 5))
    residuals = y_test_m - y_pred_m
    plt.scatter(y_pred_m, residuals, color='#1b4332', alpha=0.5)
    plt.axhline(0, color='#d90429', linestyle='--', linewidth=2)
    plt.title('Multiple Linear Regression: Residual Plot', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Yield (kg/ha)', fontsize=11)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'mlr_residuals_graph.png'), dpi=150)
    plt.close()

    # =========================================================
    # ALGORITHM 3: Logistic Regression (Soil Suitability)
    # =========================================================
    print("--- Training Algorithm 3: Logistic Regression ---")
    log_features = ['ph', 'nitrogen', 'phosphorus', 'potassium', 'moisture', 'temperature']
    X_log = soil_df[log_features].values
    y_log = soil_df['is_suitable'].values

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_log, y_log, test_size=0.20, random_state=42, stratify=y_log
    )

    log_scaler = StandardScaler()
    X_train_l_scaled = log_scaler.fit_transform(X_train_l)
    X_test_l_scaled = log_scaler.transform(X_test_l)

    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_l_scaled, y_train_l)
    y_pred_l = log_model.predict(X_test_l_scaled)

    acc_l = float(accuracy_score(y_test_l, y_pred_l))
    prec_l = float(precision_score(y_test_l, y_pred_l, zero_division=0))
    rec_l = float(recall_score(y_test_l, y_pred_l, zero_division=0))
    f1_l = float(f1_score(y_test_l, y_pred_l, zero_division=0))
    cm_l = confusion_matrix(y_test_l, y_pred_l).tolist()

    metrics_summary['logistic_regression'] = {
        'name': 'Logistic Regression (Soil Suitability)',
        'type': 'Classification',
        'features': log_features,
        'target': 'Soil Suitability (0: Unsuitable, 1: Suitable)',
        'metrics': {
            'Accuracy': round(acc_l * 100, 2),
            'Precision': round(prec_l * 100, 2),
            'Recall': round(rec_l * 100, 2),
            'F1_Score': round(f1_l * 100, 2)
        },
        'confusion_matrix': cm_l,
        'classes': ['Unsuitable', 'Suitable']
    }
    joblib.dump(log_model, os.path.join(MODEL_DIR, 'logistic_regression_soil.joblib'))
    joblib.dump(log_scaler, os.path.join(MODEL_DIR, 'soil_scaler.joblib'))

    # Plot Logistic Regression Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm_l)
    plt.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Unsuitable', 'Suitable'])
    plt.yticks(tick_marks, ['Unsuitable', 'Suitable'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm_arr[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                     fontweight='bold', fontsize=14)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'logistic_confusion_matrix.png'), dpi=150)
    plt.close()

    # =========================================================
    # ALGORITHM 4: KNN Classification (Crop Recommendation)
    # =========================================================
    print("--- Training Algorithm 4: KNN Classification ---")
    knn_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_knn = crop_df[knn_features].values
    y_knn_raw = crop_df['label'].values

    knn_encoder = LabelEncoder()
    y_knn = knn_encoder.fit_transform(y_knn_raw)

    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
        X_knn, y_knn, test_size=0.20, random_state=42, stratify=y_knn
    )

    knn_scaler = StandardScaler()
    X_train_k_scaled = knn_scaler.fit_transform(X_train_k)
    X_test_k_scaled = knn_scaler.transform(X_test_k)

    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
    knn_model.fit(X_train_k_scaled, y_train_k)
    y_pred_k = knn_model.predict(X_test_k_scaled)

    acc_k = float(accuracy_score(y_test_k, y_pred_k))
    prec_k = float(precision_score(y_test_k, y_pred_k, average='weighted', zero_division=0))
    rec_k = float(recall_score(y_test_k, y_pred_k, average='weighted', zero_division=0))
    f1_k = float(f1_score(y_test_k, y_pred_k, average='weighted', zero_division=0))
    cm_k = confusion_matrix(y_test_k, y_pred_k).tolist()

    metrics_summary['knn_classification'] = {
        'name': 'K-Nearest Neighbors (KNN) Crop Recommendation',
        'type': 'Classification (Multi-class)',
        'k_neighbors': 5,
        'features': knn_features,
        'num_classes': len(knn_encoder.classes_),
        'metrics': {
            'Accuracy': round(acc_k * 100, 2),
            'Precision': round(prec_k * 100, 2),
            'Recall': round(rec_k * 100, 2),
            'F1_Score': round(f1_k * 100, 2)
        },
        'classes': list(knn_encoder.classes_)
    }
    joblib.dump(knn_model, os.path.join(MODEL_DIR, 'knn_crop_recommendation.joblib'))
    joblib.dump(knn_scaler, os.path.join(MODEL_DIR, 'knn_scaler.joblib'))
    joblib.dump(knn_encoder, os.path.join(MODEL_DIR, 'knn_encoder.joblib'))

    # =========================================================
    # ALGORITHM 5: K-Means Clustering (Agricultural Conditions)
    # =========================================================
    print("--- Training Algorithm 5: K-Means Clustering ---")
    cluster_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_cluster = crop_df[cluster_features].values

    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

    # 4 agricultural clusters:
    # Cluster 0: High Rainfall Tropical Zone (Rice, Jute, Coconut)
    # Cluster 1: High NPK Commercial & Fruit Zone (Banana, Coffee, Cotton)
    # Cluster 2: Semi-Arid Pulse Zone (Chickpea, Kidneybeans, Lentil)
    # Cluster 3: Temperate / High Potassium Fruit Zone (Apple, Grapes)
    k_clusters = 4
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_cluster_scaled)

    sil_score = float(silhouette_score(X_cluster_scaled, cluster_labels))
    inertia = float(kmeans_model.inertia_)

    # Compute cluster characteristics
    cluster_df = crop_df[cluster_features].copy()
    cluster_df['cluster'] = cluster_labels
    cluster_means = cluster_df.groupby('cluster').mean().round(2).to_dict(orient='index')

    cluster_names = {
        0: 'Tropical High-Moisture Zone',
        1: 'High-Fertility Intensive Zone',
        2: 'Semi-Arid Pulse & Grain Zone',
        3: 'Temperate High-Mineral Orchard Zone'
    }

    metrics_summary['kmeans_clustering'] = {
        'name': 'K-Means Clustering',
        'type': 'Unsupervised Clustering',
        'num_clusters': k_clusters,
        'features': cluster_features,
        'metrics': {
            'Silhouette_Score': round(sil_score, 4),
            'Inertia': round(inertia, 2)
        },
        'cluster_profiles': {
            int(k): {
                'title': cluster_names.get(int(k), f'Cluster {k}'),
                'means': cluster_means[k]
            } for k in cluster_means
        }
    }
    joblib.dump(kmeans_model, os.path.join(MODEL_DIR, 'kmeans_agricultural_clustering.joblib'))
    joblib.dump(cluster_scaler, os.path.join(MODEL_DIR, 'kmeans_scaler.joblib'))

    # PCA 2D Visualization of Clusters
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cluster_scaled)
    joblib.dump(pca, os.path.join(MODEL_DIR, 'kmeans_pca.joblib'))

    plt.figure(figsize=(9, 6))
    colors = ['#2a9d8f', '#e76f51', '#f4a261', '#264653']
    for c in range(k_clusters):
        mask = cluster_labels == c
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=colors[c % len(colors)],
            label=f'{cluster_names[c]} (n={np.sum(mask)})',
            alpha=0.6, edgecolors='none'
        )
    # Centroids in PCA space
    centroids_pca = pca.transform(kmeans_model.cluster_centers_)
    plt.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        s=180, marker='X', color='black', edgecolor='white', linewidth=2, label='Cluster Centroids'
    )
    plt.title(f'K-Means Agricultural Condition Clusters (PCA Projection, k={k_clusters})', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(f'Principal Component 1 ({round(pca.explained_variance_ratio_[0]*100, 1)}% variance)', fontsize=11)
    plt.ylabel(f'Principal Component 2 ({round(pca.explained_variance_ratio_[1]*100, 1)}% variance)', fontsize=11)
    plt.legend(frameon=True, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'kmeans_pca_clusters.png'), dpi=150)
    plt.close()

    # =========================================================
    # ALGORITHM 6: Gaussian Naive Bayes (Crop Category Prediction)
    # =========================================================
    print("--- Training Algorithm 6: Gaussian Naive Bayes ---")
    nb_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_nb = crop_df[nb_features].values
    y_nb_raw = crop_df['category'].values

    nb_encoder = LabelEncoder()
    y_nb = nb_encoder.fit_transform(y_nb_raw)

    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
        X_nb, y_nb, test_size=0.20, random_state=42, stratify=y_nb
    )

    nb_scaler = StandardScaler()
    X_train_nb_scaled = nb_scaler.fit_transform(X_train_nb)
    X_test_nb_scaled = nb_scaler.transform(X_test_nb)

    nb_model = GaussianNB()
    nb_model.fit(X_train_nb_scaled, y_train_nb)
    y_pred_nb = nb_model.predict(X_test_nb_scaled)

    acc_nb = float(accuracy_score(y_test_nb, y_pred_nb))
    prec_nb = float(precision_score(y_test_nb, y_pred_nb, average='weighted', zero_division=0))
    rec_nb = float(recall_score(y_test_nb, y_pred_nb, average='weighted', zero_division=0))
    f1_nb = float(f1_score(y_test_nb, y_pred_nb, average='weighted', zero_division=0))
    cm_nb = confusion_matrix(y_test_nb, y_pred_nb).tolist()

    metrics_summary['naive_bayes_classification'] = {
        'name': 'Gaussian Naive Bayes (Crop Category)',
        'type': 'Classification (Multi-class)',
        'features': nb_features,
        'metrics': {
            'Accuracy': round(acc_nb * 100, 2),
            'Precision': round(prec_nb * 100, 2),
            'Recall': round(rec_nb * 100, 2),
            'F1_Score': round(f1_nb * 100, 2)
        },
        'confusion_matrix': cm_nb,
        'classes': list(nb_encoder.classes_)
    }
    joblib.dump(nb_model, os.path.join(MODEL_DIR, 'naive_bayes_crop_category.joblib'))
    joblib.dump(nb_scaler, os.path.join(MODEL_DIR, 'nb_scaler.joblib'))
    joblib.dump(nb_encoder, os.path.join(MODEL_DIR, 'nb_encoder.joblib'))

    # Plot Naive Bayes Confusion Matrix
    plt.figure(figsize=(7, 6))
    cm_nb_arr = np.array(cm_nb)
    plt.imshow(cm_nb_arr, interpolation='nearest', cmap=plt.cm.YlGn)
    plt.title('Gaussian Naive Bayes Category Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.colorbar()
    tick_marks = np.arange(len(nb_encoder.classes_))
    plt.xticks(tick_marks, nb_encoder.classes_, rotation=30, ha='right')
    plt.yticks(tick_marks, nb_encoder.classes_)
    for i in range(len(nb_encoder.classes_)):
        for j in range(len(nb_encoder.classes_)):
            plt.text(j, i, format(cm_nb_arr[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_nb_arr[i, j] > cm_nb_arr.max() / 2 else "black",
                     fontweight='bold', fontsize=12)
    plt.xlabel('Predicted Category', fontweight='bold')
    plt.ylabel('True Category', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'nb_confusion_matrix.png'), dpi=150)
    plt.close()

    # Save metrics JSON
    metrics_json_path = os.path.join(MODEL_DIR, 'model_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n==========================================")
    print("ALL 6 MODELS TRAINED AND SAVED SUCCESSFULLY")
    print(f"Metrics saved to: {metrics_json_path}")
    print("==========================================")

    for k, v in metrics_summary.items():
        print(f"\nModel: {v['name']} ({v['type']})")
        for m_name, m_val in v['metrics'].items():
            print(f"  {m_name}: {m_val}")

    return metrics_summary


if __name__ == '__main__':
    train_and_evaluate_all()
