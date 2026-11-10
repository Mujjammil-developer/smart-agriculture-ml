# Smart Agriculture – Crop Yield Prediction & Crop Recommendation System Using Machine Learning

A complete, responsive, and fully functional Machine Learning web application designed to help farmers, agricultural extension workers, and agronomists make data-driven agricultural decisions.

The application leverages **six Scikit-learn Machine Learning algorithms** to recommend optimal crops, predict yields, diagnose soil suitability, cluster agro-climatic zones, and classify crop categories.

---

## 🌾 Key System Modules (8 Modules)

1. **Dashboard (`/dashboard`)**:
   - Executive metrics: Supported crops (22), average regional rainfall, average soil pH, and total agricultural records.
   - Interactive Chart.js visualizations for meteorological trends (Rainfall vs. Temperature) and soil nutrient distributions (N-P-K radar).
   - High-level status overview of all 6 Machine Learning models.
   - Recent crop recommendations and yield predictions audit tables.

2. **Soil Management (`/soil`)**:
   - Comprehensive CRUD (Create, Read, Update, Delete) for field soil samples.
   - Tracks Soil pH, Nitrogen (N), Phosphorus (P), Potassium (K), and soil type.
   - Automated agronomic fertility classification (*High*, *Moderate*, *Low Fertility*).
   - Instant CSV export of all soil records.

3. **Weather Management (`/weather`)**:
   - Tracks Temperature (°C), Relative Humidity (%), Rainfall (mm), and sky conditions.
   - Meteorological data visualization with multi-axis Chart.js bar and line graphs.
   - Summary telemetry: Average temperature, relative humidity, and cumulative rainfall.
   - Instant CSV export of weather observations.

4. **Crop Recommendation (`/recommendation`)**:
   - Powered by **K-Nearest Neighbors (KNN)** with `StandardScaler` feature normalization.
   - Delivers the **Top 3 Recommended Crops** ranked by match confidence percentage.
   - Displays crop seasonality, key environmental preferences, and agronomic icons.
   - Includes 1-click test presets (*Rice Wetland*, *Dry Chickpea*, *Coffee Plantation*, *Apple Orchard*).

5. **Crop Yield Prediction (`/yield-prediction`)**:
   - **Simple Linear Regression**: Predicts crop yield (kg/hectare) solely from seasonal rainfall volume; displays the regression equation and fitted trendline plot.
   - **Multiple Linear Regression**: Forecasts yield based on 7 agronomic and meteorological factors (*Rainfall, Temperature, Humidity, pH, N, P, K*) with residual error analysis.
   - Reports calculated regression metrics: MAE, MSE, RMSE, and R² score.

6. **Soil Suitability Analysis (`/soil-suitability`)**:
   - Powered by **Logistic Regression** binary classification.
   - Evaluates soil sample viability (*Suitable* vs. *Unsuitable*) with probability confidence percentage.
   - Provides targeted agronomic corrective actions (e.g., lime addition for acidic soil, organic matter replenishment, phosphorus amendment).
   - Displays the confusion matrix calculated from the test set.

7. **Crop Analysis (`/crop-analysis`)**:
   - **K-Means Clustering**: Unsupervised grouping of agricultural conditions into 4 distinct agro-climatic zones with centroid characteristics and 2D PCA projection scatter plot.
   - **Gaussian Naive Bayes**: Multi-class classification predicting crop category (*Cereals, Pulses, Fruits, Commercial / Cash Crops*) with probability distribution breakdown.

8. **Reports & Analytics (`/reports`)**:
   - Centralized performance benchmark hub comparing all 6 ML algorithms.
   - Historical audit logs for yield forecasts and crop recommendations.
   - One-click CSV downloads for:
     - Yield Prediction History CSV
     - Crop Recommendations Report CSV
     - Soil Records CSV
     - Weather Records CSV

---

## 🤖 Machine Learning Algorithms Overview

| # | Algorithm | Scikit-learn Class | Inputs | Output / Target | Calculated Benchmark Metric |
|---|---|---|---|---|---|
| **1** | **Simple Linear Regression** | `LinearRegression` | Rainfall (mm) | Yield (kg/ha) | **R²: 0.966**, RMSE: 218.51 kg/ha |
| **2** | **Multiple Linear Regression** | `LinearRegression` + `StandardScaler` | Rainfall, Temp, Humidity, pH, N, P, K | Yield (kg/ha) | **R²: 0.742**, RMSE: 397.38 kg/ha |
| **3** | **Logistic Regression** | `LogisticRegression` + `StandardScaler` | pH, N, P, K, Moisture, Temp | Suitable (1) vs Unsuitable (0) | **Accuracy: 63.3%**, F1: 71.7% |
| **4** | **KNN Classification** | `KNeighborsClassifier(k=5)` + `StandardScaler` | N, P, K, Temp, Humidity, pH, Rainfall | Top 3 Recommended Crops | **Accuracy: 98.6%**, F1: 98.6% |
| **5** | **K-Means Clustering** | `KMeans(k=4)` + `StandardScaler` + `PCA` | N, P, K, Temp, Humidity, pH, Rainfall | 4 Agro-climatic Zones | **Silhouette Score: 0.2813** |
| **6** | **Gaussian Naive Bayes** | `GaussianNB` + `StandardScaler` | N, P, K, Temp, Humidity, pH, Rainfall | Crop Category (Cereals, Pulses, Fruits, Cash) | **Accuracy: 89.5%**, F1: 89.6% |

---

## 💻 Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, FontAwesome 6, Chart.js 4.4.
- **Backend**: Python 3.11+, Flask 3.1, Flask-SQLAlchemy 3.1.
- **Database**: PostgreSQL (Production on Render) / SQLite (Automatic Local Development Fallback).
- **Machine Learning**: Scikit-learn, NumPy, Pandas, Joblib, Matplotlib.
- **Production Server**: Gunicorn WSGI.

---

## 🚀 Running Locally

### Step 1: Open Terminal in the Project Directory
```bash
cd smart_agriculture
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Retrain Models (Optional, Pre-trained Models are already included in `models/`)
```bash
python train_models.py
```

### Step 5: Start the Local Development Server
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
The application will automatically initialize the local SQLite database (`smart_agriculture.db`) and seed default crop reference profiles and sample field records!

---

## 🌐 Deploying Online to Render (Step-by-Step Guide)

Follow these beginner-friendly instructions to host your project on **Render** with a **live public URL** and **PostgreSQL database**.

### Part 1: Push Project to GitHub

1. **Sign in to GitHub** ([github.com](https://github.com/)).
2. Click the **"+"** icon in the top right corner and select **New repository**.
3. Name your repository (e.g., `smart-agriculture-ml`), set it to **Public**, and click **Create repository**.
4. On your local machine, open terminal in the `smart_agriculture` folder and execute:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Complete Smart Agriculture ML Web App"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
   git push -u origin main
   ```

---

### Part 2: Deploy on Render

#### Option A: Quick Blueprint Deployment (Recommended - 1 Click)
This repository includes a `render.yaml` configuration that automatically provisions both the Web Service and PostgreSQL database.

1. Go to [render.com](https://render.com/) and sign in (you can sign in with your GitHub account).
2. In the Render Dashboard, click **New +** &rarr; **Blueprint**.
3. Connect your GitHub repository (`smart-agriculture-ml`).
4. Render will read `render.yaml` and display two resources:
   - `smart-agri-db` (PostgreSQL Database)
   - `smart-agriculture-app` (Web Service)
5. Click **Apply**. Render will automatically build the service, configure database credentials, run `pip install -r requirements.txt`, and start the app with `gunicorn wsgi:app`!

---

#### Option B: Manual Web Service & PostgreSQL Setup

If you prefer to configure the services manually:

##### Step 1: Create the PostgreSQL Database
1. In the Render Dashboard, click **New +** &rarr; **PostgreSQL**.
2. Set the Name to `smart-agri-db` and Database to `smartagri`.
3. Select the **Free** instance type.
4. Click **Create Database**.
5. Once created, copy the **Internal Database URL** (or External Database URL).

##### Step 2: Create the Web Service
1. In Render Dashboard, click **New +** &rarr; **Web Service**.
2. Connect your GitHub repository.
3. Configure the following fields:
   - **Name**: `smart-agriculture-app`
   - **Language**: `Python 3`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app`
   - **Instance Type**: `Free`
4. Expand **Advanced** and add the following **Environment Variables**:
   - `DATABASE_URL`: Paste the PostgreSQL connection URL copied from Step 1.
   - `SECRET_KEY`: Enter any random secure string (e.g. `smartagri-production-secure-key-2026`).
   - `PYTHON_VERSION`: `3.11.9`
5. Click **Create Web Service**.

Render will deploy the application and provide a public URL like:
```
https://smart-agriculture-app.onrender.com
```

---

## 🧪 Demonstration & Faculty Presentation Checklist

When presenting this project to your college faculty:

1. **Dashboard Walkthrough (`/dashboard`)**:
   - Highlight the **Database Status badge** in the sidebar (showing PostgreSQL on Render or SQLite in Dev Mode).
   - Show the summary cards (22 crops, rainfall, pH, total records).
   - Point out the real-time **Weather Trend chart** and **NPK nutrient radar chart**.

2. **Soil Management Module (`/soil`)**:
   - Add a new soil sample (e.g. pH: 6.8, N: 85, P: 45, K: 40).
   - Show how the system calculates the **Fertility Status** badge automatically.
   - Edit the record and demonstrate the **Export CSV** feature.

3. **Weather Management Module (`/weather`)**:
   - Add a weather observation (Temp: 27°C, Humidity: 80%, Rainfall: 190mm).
   - Show the live updated meteorological visualization.

4. **Crop Recommendation (`/recommendation`)**:
   - Use the **Quick Test Presets** (e.g. click "Rice (Wetland)" or "Chickpea").
   - Click **Predict Suitable Crops** &rarr; Showcase the **Top 3 Recommended Crops** with confidence percentages, growing season, and environmental descriptions.
   - Mention the **98.6% KNN test accuracy**.

5. **Crop Yield Prediction (`/yield-prediction`)**:
   - Demonstrate **Tab 1: Simple Linear Regression**: Enter rainfall (e.g. 180 mm) &rarr; view predicted yield and the regression line chart.
   - Demonstrate **Tab 2: Multiple Linear Regression**: Show the 7-variable multi-factor forecast with residual plot and calculated R² score (0.742).

6. **Soil Suitability (`/soil-suitability`)**:
   - Test an optimal soil sample (e.g. click "Balanced / Optimal Soil") &rarr; shows **Suitable** with high probability.
   - Test an acidic sample (click "Acidic Nutrient-Deficient Soil") &rarr; shows **Unsuitable** and displays diagnostic agronomic remediation steps (e.g. add agricultural lime to neutralize acidity).

7. **Crop Analysis (`/crop-analysis`)**:
   - Demonstrate **K-Means clustering** assigning the field to one of 4 agro-climatic zones, showing the 2D PCA cluster map and centroid values.
   - Show **Gaussian Naive Bayes** category classification breakdown across Cereals, Pulses, Fruits, and Cash Crops.

8. **Reports & Analytics (`/reports`)**:
   - Review the complete **6-Algorithm Model Performance Benchmark Table**.
   - Download the **Yield Forecast History CSV** and open it in Excel/Sheets to show complete data persistence.

9. **Health Check Endpoint (`/health`)**:
   - Open `/health` in a new tab to show the JSON monitoring endpoint indicating database connectivity and all 6 models active.
