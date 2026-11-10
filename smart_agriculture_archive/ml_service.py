import os
import json
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load cached artifacts
_MODELS = {}

def load_all_models():
    """Loads all 6 ML model pipelines and artifacts."""
    global _MODELS
    if _MODELS:
        return _MODELS

    models_to_load = {
        'slr': 'simple_linear_regression.joblib',
        'mlr': 'multiple_linear_regression.joblib',
        'mlr_scaler': 'mlr_scaler.joblib',
        'logistic': 'logistic_regression_soil.joblib',
        'soil_scaler': 'soil_scaler.joblib',
        'knn': 'knn_crop_recommendation.joblib',
        'knn_scaler': 'knn_scaler.joblib',
        'knn_encoder': 'knn_encoder.joblib',
        'kmeans': 'kmeans_agricultural_clustering.joblib',
        'kmeans_scaler': 'kmeans_scaler.joblib',
        'kmeans_pca': 'kmeans_pca.joblib',
        'nb': 'naive_bayes_crop_category.joblib',
        'nb_scaler': 'nb_scaler.joblib',
        'nb_encoder': 'nb_encoder.joblib',
    }

    for key, filename in models_to_load.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            _MODELS[key] = joblib.load(filepath)
        else:
            print(f"Warning: Model file {filename} not found.")

    # Load metrics json
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            _MODELS['metrics'] = json.load(f)
    else:
        _MODELS['metrics'] = {}

    return _MODELS

# Ensure models are loaded on import
load_all_models()

# Reference crop details
CROP_METADATA = {
    'rice': {'name': 'Rice (Paddy)', 'category': 'Cereals', 'icon': 'fa-bowl-rice', 'season': 'Kharif (Monsoon)', 'description': 'Thrives in high moisture, standing water and warm sub-tropical climate.'},
    'maize': {'name': 'Maize (Corn)', 'category': 'Cereals', 'icon': 'fa-seedling', 'season': 'Kharif / Rabi', 'description': 'Versatile cereal requiring well-drained loamy soil and moderate rainfall.'},
    'chickpea': {'name': 'Chickpea (Gram)', 'category': 'Pulses', 'icon': 'fa-cookie-bite', 'season': 'Rabi (Winter)', 'description': 'Cold hardy leguminous pulse that fixes atmospheric nitrogen in the soil.'},
    'kidneybeans': {'name': 'Kidney Beans (Rajma)', 'category': 'Pulses', 'icon': 'fa-bone', 'season': 'Rabi / Spring', 'description': 'Nutrient-rich legume suitable for temperate and sub-tropical elevations.'},
    'pigeonpeas': {'name': 'Pigeon Peas (Arhar / Toor)', 'category': 'Pulses', 'icon': 'fa-cubes-stacked', 'season': 'Kharif', 'description': 'Deep-rooted drought-resilient pulse crop ideal for dryland agriculture.'},
    'mothbeans': {'name': 'Moth Beans', 'category': 'Pulses', 'icon': 'fa-shapes', 'season': 'Kharif', 'description': 'Extremely drought-hardy legume suitable for arid and semi-arid sandy regions.'},
    'mungbean': {'name': 'Mung Bean (Green Gram)', 'category': 'Pulses', 'icon': 'fa-certificate', 'season': 'Kharif / Summer', 'description': 'Short-duration pulse crop ideal for crop rotation and soil enrichment.'},
    'blackgram': {'name': 'Black Gram (Urad)', 'category': 'Pulses', 'icon': 'fa-circle-dot', 'season': 'Kharif / Summer', 'description': 'Warm-season pulse demanding warm humid temperatures and loamy soil.'},
    'lentil': {'name': 'Lentil (Masoor)', 'category': 'Pulses', 'icon': 'fa-gem', 'season': 'Rabi', 'description': 'Cool-season legume with high protein content and moderate water requirements.'},
    'pomegranate': {'name': 'Pomegranate', 'category': 'Fruits', 'icon': 'fa-apple-whole', 'season': 'Perennial (Year-round)', 'description': 'Drought-tolerant fruit crop suited to semi-arid climates with mild winters.'},
    'banana': {'name': 'Banana', 'category': 'Fruits', 'icon': 'fa-lemon', 'season': 'Tropical Year-round', 'description': 'Heavy feeder requiring abundant water, high potassium, and tropical warmth.'},
    'mango': {'name': 'Mango', 'category': 'Fruits', 'icon': 'fa-fan', 'season': 'Summer Harvest', 'description': 'King of fruits, favors warm tropical climate and distinct dry flowering period.'},
    'grapes': {'name': 'Grapes', 'category': 'Fruits', 'icon': 'fa-wine-glass', 'season': 'Spring / Summer', 'description': 'Demands high potassium, well-drained sandy loam, and dry harvest conditions.'},
    'watermelon': {'name': 'Watermelon', 'category': 'Fruits', 'icon': 'fa-sun', 'season': 'Zaid (Summer)', 'description': 'Warm season vine crop thriving in sandy river beds and warm sunny days.'},
    'muskmelon': {'name': 'Muskmelon', 'category': 'Fruits', 'icon': 'fa-circle', 'season': 'Zaid (Summer)', 'description': 'Warm-season dessert fruit thriving in warm, dry weather and sandy loam.'},
    'apple': {'name': 'Apple', 'category': 'Fruits', 'icon': 'fa-apple-whole', 'season': 'Temperate / Autumn', 'description': 'Requires chilling hours, temperate climate, and rich potassium soils.'},
    'orange': {'name': 'Orange (Citrus)', 'category': 'Fruits', 'icon': 'fa-lemon', 'season': 'Subtropical / Winter', 'description': 'Citrus tree preferring subtropical warmth and well-aerated light soils.'},
    'papaya': {'name': 'Papaya', 'category': 'Fruits', 'icon': 'fa-egg', 'season': 'Tropical Year-round', 'description': 'Fast growing fruit tree that thrives in warm frost-free sunny environments.'},
    'coconut': {'name': 'Coconut', 'category': 'Fruits', 'icon': 'fa-tree', 'season': 'Coastal Tropical', 'description': 'High moisture coastal palm requiring humid climate and saline tolerance.'},
    'cotton': {'name': 'Cotton', 'category': 'Commercial Crops', 'icon': 'fa-shirt', 'season': 'Kharif', 'description': 'Major commercial cash crop requiring deep black soil and high nitrogen.'},
    'jute': {'name': 'Jute (Golden Fiber)', 'category': 'Commercial Crops', 'icon': 'fa-scroll', 'season': 'Kharif', 'description': 'Natural fiber crop requiring humid tropical conditions and high rainfall.'},
    'coffee': {'name': 'Coffee', 'category': 'Commercial Crops', 'icon': 'fa-mug-hot', 'season': 'Hill Plantation', 'description': 'Shade-loving plantation crop thriving on tropical hills with high organic humus.'}
}


def predict_simple_linear_regression(rainfall):
    """Algorithm 1: Simple Linear Regression (Rainfall -> Crop Yield)"""
    models = load_all_models()
    model = models.get('slr')
    if not model:
        raise ValueError("Simple Linear Regression model not loaded.")

    rainfall_val = float(rainfall)
    pred_yield = float(model.predict([[rainfall_val]])[0])
    pred_yield = max(round(pred_yield, 2), 0.0)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    return {
        'rainfall': rainfall_val,
        'predicted_yield_kg_per_ha': pred_yield,
        'predicted_yield_tons_per_ha': round(pred_yield / 1000.0, 2),
        'slope': round(slope, 3),
        'intercept': round(intercept, 2),
        'equation': f"Yield = {round(intercept, 2)} + ({round(slope, 2)} × Rainfall)"
    }


def predict_multiple_linear_regression(rainfall, temperature, humidity, ph, nitrogen, phosphorus, potassium):
    """Algorithm 2: Multiple Linear Regression (7 variables -> Crop Yield)"""
    models = load_all_models()
    model = models.get('mlr')
    scaler = models.get('mlr_scaler')
    if not model or not scaler:
        raise ValueError("Multiple Linear Regression model or scaler not loaded.")

    features = [
        float(rainfall),
        float(temperature),
        float(humidity),
        float(ph),
        float(nitrogen),
        float(phosphorus),
        float(potassium)
    ]

    features_scaled = scaler.transform([features])
    pred_yield = float(model.predict(features_scaled)[0])
    pred_yield = max(round(pred_yield, 2), 0.0)

    # Determine yield category
    if pred_yield >= 4500:
        yield_category = 'High Potential Yield'
        category_badge = 'success'
    elif pred_yield >= 2800:
        yield_category = 'Moderate Yield'
        category_badge = 'primary'
    else:
        yield_category = 'Low / Sub-optimal Yield'
        category_badge = 'warning'

    return {
        'inputs': {
            'rainfall': features[0],
            'temperature': features[1],
            'humidity': features[2],
            'ph': features[3],
            'nitrogen': features[4],
            'phosphorus': features[5],
            'potassium': features[6]
        },
        'predicted_yield_kg_per_ha': pred_yield,
        'predicted_yield_tons_per_ha': round(pred_yield / 1000.0, 2),
        'yield_category': yield_category,
        'category_badge': category_badge
    }


def predict_soil_suitability(ph, nitrogen, phosphorus, potassium, moisture, temperature):
    """Algorithm 3: Logistic Regression (Soil Suitability Classification)"""
    models = load_all_models()
    model = models.get('logistic')
    scaler = models.get('soil_scaler')
    if not model or not scaler:
        raise ValueError("Logistic Regression model or scaler not loaded.")

    inputs = [
        float(ph),
        float(nitrogen),
        float(phosphorus),
        float(potassium),
        float(moisture),
        float(temperature)
    ]

    inputs_scaled = scaler.transform([inputs])
    pred_class = int(model.predict(inputs_scaled)[0])
    pred_proba = model.predict_proba(inputs_scaled)[0]

    prob_suitable = float(pred_proba[1])
    prob_unsuitable = float(pred_proba[0])

    is_suitable = (pred_class == 1)

    # Agronomic diagnostics
    recommendations = []
    if inputs[0] < 6.0:
        recommendations.append("Soil is acidic (pH < 6.0). Apply agricultural lime or dolomite to balance pH.")
    elif inputs[0] > 7.8:
        recommendations.append("Soil is alkaline (pH > 7.8). Apply gypsum or sulfur to lower alkalinity.")
    else:
        recommendations.append("Soil pH is in the optimal range (6.0 - 7.8) for nutrient uptake.")

    if inputs[1] < 35:
        recommendations.append("Nitrogen level is low. Supplement with organic manure, urea, or legume intercropping.")
    if inputs[2] < 20:
        recommendations.append("Phosphorus level is deficient. Apply rock phosphate or single superphosphate (SSP).")
    if inputs[3] < 25:
        recommendations.append("Potassium is below threshold. Apply Muriate of Potash (MOP) to enhance disease resistance.")
    if inputs[4] < 25:
        recommendations.append("Soil moisture is insufficient. Implement drip irrigation or mulching.")
    elif inputs[4] > 80:
        recommendations.append("Excessive moisture / waterlogging detected. Ensure proper field drainage channels.")

    if not recommendations or len(recommendations) == 1:
        recommendations.append("Overall balanced agronomic parameters. Maintain regular soil organic matter addition.")

    return {
        'is_suitable': is_suitable,
        'status_text': 'Suitable for Cultivation' if is_suitable else 'Unsuitable / Needs Soil Amendment',
        'status_badge': 'success' if is_suitable else 'danger',
        'suitability_percentage': round(prob_suitable * 100, 1),
        'unsuitability_percentage': round(prob_unsuitable * 100, 1),
        'probability': round(prob_suitable, 4),
        'recommendations': recommendations
    }


def recommend_crops_knn(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, top_k=3):
    """Algorithm 4: KNN Classification (Top 3 Crop Recommendation with Feature Scaling)"""
    models = load_all_models()
    model = models.get('knn')
    scaler = models.get('knn_scaler')
    encoder = models.get('knn_encoder')
    if not model or not scaler or not encoder:
        raise ValueError("KNN model, scaler, or encoder not loaded.")

    features = [
        float(nitrogen),
        float(phosphorus),
        float(potassium),
        float(temperature),
        float(humidity),
        float(ph),
        float(rainfall)
    ]

    features_scaled = scaler.transform([features])
    probabilities = model.predict_proba(features_scaled)[0]

    # Get top_k indices sorted by probability descending
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        crop_code = encoder.inverse_transform([idx])[0]
        prob = float(probabilities[idx])
        meta = CROP_METADATA.get(crop_code, {
            'name': crop_code.capitalize(),
            'category': 'General Crop',
            'icon': 'fa-seedling',
            'season': 'All Season',
            'description': 'Suitable under matching agronomic conditions.'
        })

        results.append({
            'rank': rank,
            'crop_code': crop_code,
            'crop_name': meta['name'],
            'category': meta['category'],
            'icon': meta['icon'],
            'season': meta['season'],
            'description': meta['description'],
            'confidence_percent': round(max(prob * 100, 5.0), 1),  # minimum visible floor
            'confidence_score': round(prob, 4)
        })

    return {
        'inputs': {
            'N': features[0],
            'P': features[1],
            'K': features[2],
            'temperature': features[3],
            'humidity': features[4],
            'ph': features[5],
            'rainfall': features[6]
        },
        'top_recommendations': results
    }


def cluster_agricultural_condition(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """Algorithm 5: K-Means Clustering (Agricultural Conditions)"""
    models = load_all_models()
    model = models.get('kmeans')
    scaler = models.get('kmeans_scaler')
    pca = models.get('kmeans_pca')
    metrics = models.get('metrics', {}).get('kmeans_clustering', {})
    if not model or not scaler:
        raise ValueError("K-Means model or scaler not loaded.")

    features = [
        float(nitrogen),
        float(phosphorus),
        float(potassium),
        float(temperature),
        float(humidity),
        float(ph),
        float(rainfall)
    ]

    features_scaled = scaler.transform([features])
    cluster_id = int(model.predict(features_scaled)[0])

    cluster_names = {
        0: 'Tropical High-Moisture Zone (Humid & Wet)',
        1: 'High-Fertility Intensive Zone (Commercial Feeder)',
        2: 'Semi-Arid Pulse & Grain Zone (Drought-Resilient)',
        3: 'Temperate High-Mineral Orchard Zone (Specialty Horticulture)'
    }

    cluster_descriptions = {
        0: 'Characterized by high precipitation (> 150mm) and high relative humidity. Ideal for flood-tolerant and tropical crops such as Rice, Jute, Coconut, and Papaya.',
        1: 'Characterized by rich macronutrient demand (N > 80, P > 40). Best suited for high-yielding cash crops and heavy feeders like Banana, Cotton, and Coffee.',
        2: 'Characterized by moderate to low rainfall and lower nitrogen requirements. Perfectly suited for leguminous pulses (Chickpea, Lentil, Beans) and millets.',
        3: 'Characterized by high potassium and phosphorus mineral concentrations with cooler temperature bands. Tailored for orchards like Apples, Grapes, and temperate fruits.'
    }

    # PCA 2D coordinates for visual pinpointing
    pca_coords = [0.0, 0.0]
    if pca:
        coords = pca.transform(features_scaled)[0]
        pca_coords = [round(float(coords[0]), 2), round(float(coords[1]), 2)]

    profile = metrics.get('cluster_profiles', {}).get(str(cluster_id), {})

    return {
        'cluster_id': cluster_id,
        'cluster_name': cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
        'description': cluster_descriptions.get(cluster_id, 'Agronomic cluster zone.'),
        'pca_coords': pca_coords,
        'centroid_means': profile.get('means', {}),
        'all_profiles': metrics.get('cluster_profiles', {})
    }


def predict_crop_category_nb(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """Algorithm 6: Gaussian Naive Bayes Classification (Crop Category Prediction)"""
    models = load_all_models()
    model = models.get('nb')
    scaler = models.get('nb_scaler')
    encoder = models.get('nb_encoder')
    if not model or not scaler or not encoder:
        raise ValueError("Naive Bayes model, scaler, or encoder not loaded.")

    features = [
        float(nitrogen),
        float(phosphorus),
        float(potassium),
        float(temperature),
        float(humidity),
        float(ph),
        float(rainfall)
    ]

    features_scaled = scaler.transform([features])
    pred_idx = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]

    category = encoder.inverse_transform([pred_idx])[0]

    breakdown = []
    for idx, cat_name in enumerate(encoder.classes_):
        breakdown.append({
            'category': cat_name,
            'probability': round(float(probabilities[idx]) * 100, 1)
        })

    # Sort breakdown by descending probability
    breakdown.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'predicted_category': category,
        'confidence_percentage': breakdown[0]['probability'],
        'probability_breakdown': breakdown
    }


def get_all_model_metrics():
    """Returns the pre-calculated metrics for all 6 models."""
    models = load_all_models()
    return models.get('metrics', {})
