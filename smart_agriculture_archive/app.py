import os
import io
import csv
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, Response, send_file
)
from dotenv import load_dotenv

load_dotenv()

from models import (
    db, SoilRecord, WeatherRecord, CropInfo,
    PredictionHistory, RecommendationHistory, SoilSuitabilityHistory
)
import ml_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Secret key configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'smart-agriculture-deepmind-secure-key-2026')

# PostgreSQL database URI for Render production with SQLite fallback for local development
raw_db_url = os.environ.get('DATABASE_URL')
if raw_db_url:
    # Render's PostgreSQL URL starts with postgres://, which SQLAlchemy 1.4+ expects as postgresql://
    if raw_db_url.startswith("postgres://"):
        raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = raw_db_url
else:
    local_db_path = os.path.join(BASE_DIR, 'smart_agriculture.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{local_db_path}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Helper function to seed initial records if empty
def seed_initial_data():
    with app.app_context():
        db.create_all()

        # Seed CropInfo if empty
        if CropInfo.query.count() == 0:
            crop_defaults = [
                ('rice', 'Cereals', 80, 47, 40, 23.6, 82.2, 6.4, 236.1, 'Thrives in warm, humid weather with water-retaining clayey soil.', 'Kharif', 'fa-bowl-rice'),
                ('maize', 'Cereals', 77, 48, 20, 22.3, 65.0, 6.2, 84.7, 'Requires well-drained fertile loam and moderate rainfall.', 'Kharif / Rabi', 'fa-seedling'),
                ('chickpea', 'Pulses', 40, 67, 79, 18.8, 16.8, 7.3, 80.0, 'Cool-season crop that enriches soil through nitrogen fixation.', 'Rabi', 'fa-cookie-bite'),
                ('kidneybeans', 'Pulses', 20, 67, 20, 20.1, 21.6, 5.7, 105.9, 'Sensitive to frost; thrives in fertile, well-aerated soil.', 'Rabi', 'fa-bone'),
                ('pigeonpeas', 'Pulses', 20, 67, 20, 27.7, 48.0, 5.7, 149.4, 'Deep taproots make it drought-resilient for semi-arid zones.', 'Kharif', 'fa-cubes-stacked'),
                ('mothbeans', 'Pulses', 21, 48, 20, 28.1, 53.1, 6.8, 51.1, 'Exceptional drought tolerance; suited to arid sandy soils.', 'Kharif', 'fa-shapes'),
                ('mungbean', 'Pulses', 20, 47, 20, 28.5, 85.9, 6.7, 48.4, 'Short-duration leguminous crop great for soil rotation.', 'Kharif / Summer', 'fa-certificate'),
                ('blackgram', 'Pulses', 40, 67, 20, 29.9, 65.1, 7.1, 67.8, 'Warm weather pulse crop thriving on heavier loamy soils.', 'Kharif', 'fa-circle-dot'),
                ('lentil', 'Pulses', 18, 68, 19, 24.5, 64.8, 6.9, 45.6, 'Hardy pulse with low water needs and high dietary protein.', 'Rabi', 'fa-gem'),
                ('pomegranate', 'Fruits', 18, 18, 40, 21.8, 90.1, 6.4, 107.5, 'Subtropical fruit tree tolerant of dry spells and limestone soils.', 'Year-round', 'fa-apple-whole'),
                ('banana', 'Fruits', 100, 82, 50, 27.3, 80.3, 5.9, 104.6, 'Heavy potassium feeder requiring rich moist alluvium.', 'Tropical Year-round', 'fa-lemon'),
                ('mango', 'Fruits', 20, 27, 29, 31.2, 49.9, 5.7, 94.7, 'Requires warm tropical weather with a dry season during blooming.', 'Summer', 'fa-fan'),
                ('grapes', 'Fruits', 23, 132, 200, 23.8, 81.8, 6.0, 69.6, 'Demands high potassium, good drainage, and sunny dry ripening.', 'Spring / Summer', 'fa-wine-glass'),
                ('watermelon', 'Fruits', 99, 17, 50, 25.5, 85.1, 6.4, 50.7, 'Warm-season vine requiring sandy loam and abundant sunlight.', 'Summer / Zaid', 'fa-sun'),
                ('muskmelon', 'Fruits', 100, 17, 50, 28.6, 92.3, 6.3, 24.6, 'Sweet melon requiring high temperatures and dry sunny harvest.', 'Summer / Zaid', 'fa-circle'),
                ('apple', 'Fruits', 20, 134, 199, 22.6, 92.3, 5.9, 112.6, 'Requires chilling hours, temperate climate, and rich potassium soils.', 'Autumn', 'fa-apple-whole'),
                ('orange', 'Fruits', 19, 16, 10, 22.7, 92.1, 7.0, 110.4, 'Subtropical citrus requiring well-aerated light loams and sun.', 'Winter', 'fa-lemon'),
                ('papaya', 'Fruits', 49, 59, 50, 33.7, 92.4, 6.7, 142.6, 'Fast-growing tropical tree sensitive to frost and waterlogging.', 'Year-round', 'fa-egg'),
                ('coconut', 'Fruits', 21, 16, 30, 27.4, 94.8, 5.9, 175.6, 'Coastal palm requiring abundant sunlight and maritime humidity.', 'Coastal Year-round', 'fa-tree'),
                ('cotton', 'Commercial Crops', 117, 46, 19, 23.9, 79.8, 6.9, 80.3, 'Major commercial cash crop thriving in black cotton soils.', 'Kharif', 'fa-shirt'),
                ('jute', 'Commercial Crops', 78, 46, 39, 24.9, 79.6, 6.7, 174.7, 'Golden bast fiber requiring monsoon rains and humid warmth.', 'Kharif', 'fa-scroll'),
                ('coffee', 'Commercial Crops', 101, 28, 29, 25.5, 58.8, 6.7, 158.0, 'Shade-loving hill plantation crop requiring well-drained humus.', 'Plantation', 'fa-mug-hot')
            ]
            for row in crop_defaults:
                db.session.add(CropInfo(
                    name=row[0], category=row[1], ideal_n=row[2], ideal_p=row[3], ideal_k=row[4],
                    ideal_temp=row[5], ideal_humidity=row[6], ideal_ph=row[7], ideal_rainfall=row[8],
                    description=row[9], season=row[10], icon=row[11]
                ))

        # Seed sample soil records if empty
        if SoilRecord.query.count() == 0:
            sample_soils = [
                ('North Field Plot 1', 6.5, 85.0, 48.0, 42.0, 'Loamy', 'Rich in organic compost, ideal for cereal crops.'),
                ('Valley Basin Plot B', 5.8, 32.0, 65.0, 22.0, 'Clay Loam', 'Slightly acidic; recommended for pulse rotation.'),
                ('Hillside Terrace 3', 7.2, 22.0, 130.0, 195.0, 'Sandy Loam', 'High mineral content; optimal for orchard cultivation.'),
                ('East Riverbed Plot 4', 6.4, 95.0, 20.0, 48.0, 'Alluvial', 'High nitrogen alluvial soil with high drainage capacity.')
            ]
            for loc, ph, n, p, k, stype, notes in sample_soils:
                db.session.add(SoilRecord(
                    location=loc, ph=ph, nitrogen=n, phosphorus=p, potassium=k, soil_type=stype, notes=notes
                ))

        # Seed sample weather records if empty
        if WeatherRecord.query.count() == 0:
            sample_weathers = [
                ('Station Alpha', 26.5, 78.0, 185.0, 'Rainy'),
                ('Station Beta', 24.2, 82.0, 210.0, 'Overcast'),
                ('Station Gamma', 28.0, 65.0, 95.0, 'Sunny'),
                ('Station Delta', 22.5, 70.0, 140.0, 'Humid')
            ]
            for loc, temp, hum, rain, cond in sample_weathers:
                db.session.add(WeatherRecord(
                    location=loc, temperature=temp, humidity=hum, rainfall=rain, condition=cond
                ))

        # Seed sample yield predictions if empty
        if PredictionHistory.query.count() == 0:
            db.session.add(PredictionHistory(
                model_type='Simple Linear Regression', rainfall=180.0, predicted_yield=4053.72
            ))
            db.session.add(PredictionHistory(
                model_type='Multiple Linear Regression', rainfall=175.0, temperature=25.0,
                humidity=80.0, ph=6.5, nitrogen=80.0, phosphorus=45.0, potassium=40.0, predicted_yield=4320.5
            ))

        # Seed sample recommendation history if empty
        if RecommendationHistory.query.count() == 0:
            db.session.add(RecommendationHistory(
                nitrogen=80.0, phosphorus=45.0, potassium=40.0, temperature=25.0, humidity=80.0, ph=6.5, rainfall=200.0,
                top1_crop='Jute (Golden Fiber)', top1_prob=74.7,
                top2_crop='Rice (Paddy)', top2_prob=25.3,
                top3_crop='Pomegranate', top3_prob=5.0
            ))

        db.session.commit()

# Run database setup
with app.app_context():
    seed_initial_data()

# Inject common context (active database type, total crops count)
@app.context_processor
def inject_global_template_vars():
    is_postgres = 'postgresql' in app.config['SQLALCHEMY_DATABASE_URI']
    return dict(
        db_type='PostgreSQL' if is_postgres else 'SQLite (Dev Mode)',
        current_year=datetime.utcnow().year
    )

# =========================================================================
# MODULE 1: DASHBOARD
# =========================================================================
@app.route('/')
@app.route('/dashboard')
def dashboard():
    total_crops = CropInfo.query.count()
    total_soil = SoilRecord.query.count()
    total_weather = WeatherRecord.query.count()
    total_predictions = PredictionHistory.query.count()
    total_recommendations = RecommendationHistory.query.count()

    # Calculate averages
    avg_rainfall = 145.0
    avg_ph = 6.5
    weather_records = WeatherRecord.query.all()
    if weather_records:
        avg_rainfall = round(sum(w.rainfall for w in weather_records) / len(weather_records), 1)

    soil_records = SoilRecord.query.all()
    if soil_records:
        avg_ph = round(sum(s.ph for s in soil_records) / len(soil_records), 2)

    recent_predictions = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(5).all()
    recent_recommendations = RecommendationHistory.query.order_by(RecommendationHistory.created_at.desc()).limit(5).all()

    # Model metrics summary for dashboard widgets
    metrics = ml_service.get_all_model_metrics()

    return render_template(
        'dashboard.html',
        total_crops=total_crops,
        total_soil=total_soil,
        total_weather=total_weather,
        total_predictions=total_predictions,
        total_recommendations=total_recommendations,
        avg_rainfall=avg_rainfall,
        avg_ph=avg_ph,
        recent_predictions=recent_predictions,
        recent_recommendations=recent_recommendations,
        metrics=metrics
    )

# =========================================================================
# MODULE 2: SOIL MANAGEMENT (CRUD + CSV)
# =========================================================================
@app.route('/soil')
def soil_management():
    records = SoilRecord.query.order_by(SoilRecord.created_at.desc()).all()
    return render_template('soil_management.html', records=records)

@app.route('/soil/add', methods=['POST'])
def add_soil_record():
    try:
        location = request.form.get('location', 'Field Block A')
        ph = float(request.form.get('ph', 6.5))
        nitrogen = float(request.form.get('nitrogen', 40.0))
        phosphorus = float(request.form.get('phosphorus', 30.0))
        potassium = float(request.form.get('potassium', 30.0))
        soil_type = request.form.get('soil_type', 'Loamy')
        notes = request.form.get('notes', '')

        # Server-side validations
        if not (0 <= ph <= 14):
            flash('Soil pH must be between 0 and 14.', 'danger')
            return redirect(url_for('soil_management'))
        if nitrogen < 0 or phosphorus < 0 or potassium < 0:
            flash('Nutrient values (N, P, K) cannot be negative.', 'danger')
            return redirect(url_for('soil_management'))

        new_record = SoilRecord(
            location=location, ph=ph, nitrogen=nitrogen,
            phosphorus=phosphorus, potassium=potassium,
            soil_type=soil_type, notes=notes
        )
        db.session.add(new_record)
        db.session.commit()
        flash('Soil record successfully added to database!', 'success')
    except Exception as e:
        flash(f'Error adding soil record: {str(e)}', 'danger')

    return redirect(url_for('soil_management'))

@app.route('/soil/edit/<int:record_id>', methods=['POST'])
def edit_soil_record(record_id):
    record = SoilRecord.query.get_or_404(record_id)
    try:
        record.location = request.form.get('location', record.location)
        record.ph = float(request.form.get('ph', record.ph))
        record.nitrogen = float(request.form.get('nitrogen', record.nitrogen))
        record.phosphorus = float(request.form.get('phosphorus', record.phosphorus))
        record.potassium = float(request.form.get('potassium', record.potassium))
        record.soil_type = request.form.get('soil_type', record.soil_type)
        record.notes = request.form.get('notes', record.notes)

        db.session.commit()
        flash('Soil record updated successfully!', 'success')
    except Exception as e:
        flash(f'Error updating soil record: {str(e)}', 'danger')

    return redirect(url_for('soil_management'))

@app.route('/soil/delete/<int:record_id>', methods=['POST'])
def delete_soil_record(record_id):
    record = SoilRecord.query.get_or_404(record_id)
    try:
        db.session.delete(record)
        db.session.commit()
        flash('Soil record deleted successfully.', 'info')
    except Exception as e:
        flash(f'Error deleting record: {str(e)}', 'danger')

    return redirect(url_for('soil_management'))

@app.route('/soil/export-csv')
def export_soil_csv():
    records = SoilRecord.query.order_by(SoilRecord.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Location', 'Soil pH', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Soil Type', 'Notes', 'Created At'])

    for r in records:
        writer.writerow([r.id, r.location, r.ph, r.nitrogen, r.phosphorus, r.potassium, r.soil_type, r.notes or '', r.created_at.strftime('%Y-%m-%d %H:%M:%S')])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=soil_records.csv"}
    )

# =========================================================================
# MODULE 3: WEATHER MANAGEMENT (CRUD + Visualization + CSV)
# =========================================================================
@app.route('/weather')
def weather_management():
    records = WeatherRecord.query.order_by(WeatherRecord.recorded_date.desc(), WeatherRecord.created_at.desc()).all()

    # Summary metrics
    avg_temp = round(sum(r.temperature for r in records) / len(records), 1) if records else 25.0
    avg_hum = round(sum(r.humidity for r in records) / len(records), 1) if records else 70.0
    total_rain = round(sum(r.rainfall for r in records), 1) if records else 0.0

    return render_template(
        'weather_management.html',
        records=records,
        avg_temp=avg_temp,
        avg_hum=avg_hum,
        total_rain=total_rain
    )

@app.route('/weather/add', methods=['POST'])
def add_weather_record():
    try:
        location = request.form.get('location', 'Station Alpha')
        temperature = float(request.form.get('temperature', 25.0))
        humidity = float(request.form.get('humidity', 70.0))
        rainfall = float(request.form.get('rainfall', 100.0))
        condition = request.form.get('condition', 'Sunny')

        if not (-20 <= temperature <= 60):
            flash('Temperature must be within realistic ranges (-20°C to 60°C).', 'danger')
            return redirect(url_for('weather_management'))
        if not (0 <= humidity <= 100):
            flash('Humidity must be between 0% and 100%.', 'danger')
            return redirect(url_for('weather_management'))
        if rainfall < 0:
            flash('Rainfall cannot be negative.', 'danger')
            return redirect(url_for('weather_management'))

        new_weather = WeatherRecord(
            location=location, temperature=temperature,
            humidity=humidity, rainfall=rainfall, condition=condition
        )
        db.session.add(new_weather)
        db.session.commit()
        flash('Weather observation recorded successfully!', 'success')
    except Exception as e:
        flash(f'Error saving weather record: {str(e)}', 'danger')

    return redirect(url_for('weather_management'))

@app.route('/weather/delete/<int:record_id>', methods=['POST'])
def delete_weather_record(record_id):
    record = WeatherRecord.query.get_or_404(record_id)
    try:
        db.session.delete(record)
        db.session.commit()
        flash('Weather record deleted successfully.', 'info')
    except Exception as e:
        flash(f'Error deleting weather record: {str(e)}', 'danger')

    return redirect(url_for('weather_management'))

@app.route('/weather/export-csv')
def export_weather_csv():
    records = WeatherRecord.query.order_by(WeatherRecord.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Location', 'Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Condition', 'Recorded Date', 'Created At'])

    for r in records:
        writer.writerow([r.id, r.location, r.temperature, r.humidity, r.rainfall, r.condition, str(r.recorded_date), r.created_at.strftime('%Y-%m-%d %H:%M:%S')])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=weather_records.csv"}
    )

# =========================================================================
# MODULE 4: CROP RECOMMENDATION (Algorithm 4: KNN Classification)
# =========================================================================
@app.route('/recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    result = None
    inputs = {
        'N': 80.0, 'P': 40.0, 'K': 40.0,
        'temperature': 25.0, 'humidity': 80.0,
        'ph': 6.5, 'rainfall': 200.0
    }

    if request.method == 'POST':
        try:
            inputs['N'] = float(request.form.get('nitrogen', 80.0))
            inputs['P'] = float(request.form.get('phosphorus', 40.0))
            inputs['K'] = float(request.form.get('potassium', 40.0))
            inputs['temperature'] = float(request.form.get('temperature', 25.0))
            inputs['humidity'] = float(request.form.get('humidity', 80.0))
            inputs['ph'] = float(request.form.get('ph', 6.5))
            inputs['rainfall'] = float(request.form.get('rainfall', 200.0))

            # Run KNN Model Prediction
            result = ml_service.recommend_crops_knn(
                inputs['N'], inputs['P'], inputs['K'],
                inputs['temperature'], inputs['humidity'],
                inputs['ph'], inputs['rainfall']
            )

            # Save recommendation to DB
            recs = result['top_recommendations']
            history_entry = RecommendationHistory(
                nitrogen=inputs['N'], phosphorus=inputs['P'], potassium=inputs['K'],
                temperature=inputs['temperature'], humidity=inputs['humidity'],
                ph=inputs['ph'], rainfall=inputs['rainfall'],
                top1_crop=recs[0]['crop_name'], top1_prob=recs[0]['confidence_percent'],
                top2_crop=recs[1]['crop_name'], top2_prob=recs[1]['confidence_percent'],
                top3_crop=recs[2]['crop_name'], top3_prob=recs[2]['confidence_percent']
            )
            db.session.add(history_entry)
            db.session.commit()
            flash('Crop recommendation generated using KNN algorithm and saved to history!', 'success')

        except Exception as e:
            flash(f'Error generating recommendation: {str(e)}', 'danger')

    knn_metrics = ml_service.get_all_model_metrics().get('knn_classification', {})
    return render_template(
        'crop_recommendation.html',
        result=result,
        inputs=inputs,
        metrics=knn_metrics
    )

# =========================================================================
# MODULE 5: CROP YIELD PREDICTION (Algorithms 1 & 2: Simple & Multiple Linear Regression)
# =========================================================================
@app.route('/yield-prediction', methods=['GET', 'POST'])
def yield_prediction():
    active_tab = request.args.get('tab', 'simple')
    slr_result = None
    mlr_result = None

    slr_input = {'rainfall': 180.0}
    mlr_input = {
        'rainfall': 180.0, 'temperature': 26.0, 'humidity': 75.0,
        'ph': 6.5, 'nitrogen': 80.0, 'phosphorus': 45.0, 'potassium': 40.0
    }

    if request.method == 'POST':
        model_mode = request.form.get('model_mode', 'simple')
        active_tab = model_mode

        if model_mode == 'simple':
            try:
                rainfall = float(request.form.get('rainfall', 150.0))
                slr_input['rainfall'] = rainfall
                slr_result = ml_service.predict_simple_linear_regression(rainfall)

                # Save to database
                db_record = PredictionHistory(
                    model_type='Simple Linear Regression',
                    rainfall=rainfall,
                    predicted_yield=slr_result['predicted_yield_kg_per_ha']
                )
                db.session.add(db_record)
                db.session.commit()
                flash('Simple Linear Regression prediction completed successfully!', 'success')
            except Exception as e:
                flash(f'Error in Simple Linear Regression: {str(e)}', 'danger')

        elif model_mode == 'multiple':
            try:
                mlr_input['rainfall'] = float(request.form.get('rainfall', 180.0))
                mlr_input['temperature'] = float(request.form.get('temperature', 26.0))
                mlr_input['humidity'] = float(request.form.get('humidity', 75.0))
                mlr_input['ph'] = float(request.form.get('ph', 6.5))
                mlr_input['nitrogen'] = float(request.form.get('nitrogen', 80.0))
                mlr_input['phosphorus'] = float(request.form.get('phosphorus', 45.0))
                mlr_input['potassium'] = float(request.form.get('potassium', 40.0))

                mlr_result = ml_service.predict_multiple_linear_regression(
                    mlr_input['rainfall'], mlr_input['temperature'],
                    mlr_input['humidity'], mlr_input['ph'],
                    mlr_input['nitrogen'], mlr_input['phosphorus'],
                    mlr_input['potassium']
                )

                # Save to database
                db_record = PredictionHistory(
                    model_type='Multiple Linear Regression',
                    rainfall=mlr_input['rainfall'],
                    temperature=mlr_input['temperature'],
                    humidity=mlr_input['humidity'],
                    ph=mlr_input['ph'],
                    nitrogen=mlr_input['nitrogen'],
                    phosphorus=mlr_input['phosphorus'],
                    potassium=mlr_input['potassium'],
                    predicted_yield=mlr_result['predicted_yield_kg_per_ha']
                )
                db.session.add(db_record)
                db.session.commit()
                flash('Multiple Linear Regression prediction completed successfully!', 'success')
            except Exception as e:
                flash(f'Error in Multiple Linear Regression: {str(e)}', 'danger')

    all_metrics = ml_service.get_all_model_metrics()
    slr_metrics = all_metrics.get('simple_linear_regression', {})
    mlr_metrics = all_metrics.get('multiple_linear_regression', {})

    return render_template(
        'yield_prediction.html',
        active_tab=active_tab,
        slr_result=slr_result,
        mlr_result=mlr_result,
        slr_input=slr_input,
        mlr_input=mlr_input,
        slr_metrics=slr_metrics,
        mlr_metrics=mlr_metrics
    )

# =========================================================================
# MODULE 6: SOIL SUITABILITY (Algorithm 3: Logistic Regression)
# =========================================================================
@app.route('/soil-suitability', methods=['GET', 'POST'])
def soil_suitability():
    result = None
    inputs = {
        'ph': 6.5, 'nitrogen': 65.0, 'phosphorus': 40.0,
        'potassium': 45.0, 'moisture': 55.0, 'temperature': 26.0
    }

    if request.method == 'POST':
        try:
            inputs['ph'] = float(request.form.get('ph', 6.5))
            inputs['nitrogen'] = float(request.form.get('nitrogen', 65.0))
            inputs['phosphorus'] = float(request.form.get('phosphorus', 40.0))
            inputs['potassium'] = float(request.form.get('potassium', 45.0))
            inputs['moisture'] = float(request.form.get('moisture', 55.0))
            inputs['temperature'] = float(request.form.get('temperature', 26.0))

            result = ml_service.predict_soil_suitability(
                inputs['ph'], inputs['nitrogen'], inputs['phosphorus'],
                inputs['potassium'], inputs['moisture'], inputs['temperature']
            )

            # Save suitability prediction
            history_rec = SoilSuitabilityHistory(
                ph=inputs['ph'], nitrogen=inputs['nitrogen'], phosphorus=inputs['phosphorus'],
                potassium=inputs['potassium'], moisture=inputs['moisture'],
                temperature=inputs['temperature'],
                is_suitable=result['is_suitable'],
                probability=result['probability'],
                diagnosis="; ".join(result['recommendations'])
            )
            db.session.add(history_rec)
            db.session.commit()
            flash('Soil suitability evaluated using Logistic Regression model!', 'success')

        except Exception as e:
            flash(f'Error evaluating soil suitability: {str(e)}', 'danger')

    logistic_metrics = ml_service.get_all_model_metrics().get('logistic_regression', {})

    return render_template(
        'soil_suitability.html',
        result=result,
        inputs=inputs,
        metrics=logistic_metrics
    )

# =========================================================================
# MODULE 7: CROP ANALYSIS (Algorithms 5 & 6: K-Means & Naive Bayes)
# =========================================================================
@app.route('/crop-analysis', methods=['GET', 'POST'])
def crop_analysis():
    cluster_result = None
    nb_result = None

    inputs = {
        'N': 75.0, 'P': 45.0, 'K': 40.0,
        'temperature': 26.0, 'humidity': 80.0,
        'ph': 6.5, 'rainfall': 180.0
    }

    if request.method == 'POST':
        try:
            inputs['N'] = float(request.form.get('nitrogen', 75.0))
            inputs['P'] = float(request.form.get('phosphorus', 45.0))
            inputs['K'] = float(request.form.get('potassium', 40.0))
            inputs['temperature'] = float(request.form.get('temperature', 26.0))
            inputs['humidity'] = float(request.form.get('humidity', 80.0))
            inputs['ph'] = float(request.form.get('ph', 6.5))
            inputs['rainfall'] = float(request.form.get('rainfall', 180.0))

            # Run K-Means Clustering
            cluster_result = ml_service.cluster_agricultural_condition(
                inputs['N'], inputs['P'], inputs['K'],
                inputs['temperature'], inputs['humidity'],
                inputs['ph'], inputs['rainfall']
            )

            # Run Naive Bayes Category Classification
            nb_result = ml_service.predict_crop_category_nb(
                inputs['N'], inputs['P'], inputs['K'],
                inputs['temperature'], inputs['humidity'],
                inputs['ph'], inputs['rainfall']
            )

            flash('Crop Analysis executed: K-Means zone assigned & Naive Bayes category predicted!', 'success')

        except Exception as e:
            flash(f'Error running crop analysis: {str(e)}', 'danger')

    all_metrics = ml_service.get_all_model_metrics()
    kmeans_metrics = all_metrics.get('kmeans_clustering', {})
    nb_metrics = all_metrics.get('naive_bayes_classification', {})

    return render_template(
        'crop_analysis.html',
        inputs=inputs,
        cluster_result=cluster_result,
        nb_result=nb_result,
        kmeans_metrics=kmeans_metrics,
        nb_metrics=nb_metrics
    )

# =========================================================================
# MODULE 8: REPORTS AND ANALYTICS (History, Metric Hub & CSV Downloads)
# =========================================================================
@app.route('/reports')
def reports_and_analytics():
    prediction_records = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(50).all()
    recommendation_records = RecommendationHistory.query.order_by(RecommendationHistory.created_at.desc()).limit(50).all()
    suitability_records = SoilSuitabilityHistory.query.order_by(SoilSuitabilityHistory.created_at.desc()).limit(50).all()

    all_metrics = ml_service.get_all_model_metrics()

    return render_template(
        'reports.html',
        predictions=prediction_records,
        recommendations=recommendation_records,
        suitabilities=suitability_records,
        metrics=all_metrics
    )

@app.route('/reports/export/yield')
def export_yield_predictions_csv():
    records = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Algorithm Model', 'Rainfall (mm)', 'Temp (°C)', 'Humidity (%)', 'Soil pH', 'N', 'P', 'K', 'Predicted Yield (kg/ha)', 'Predicted Yield (t/ha)', 'Created At'])

    for r in records:
        writer.writerow([
            r.id, r.model_type, r.rainfall, r.temperature or '', r.humidity or '',
            r.ph or '', r.nitrogen or '', r.phosphorus or '', r.potassium or '',
            r.predicted_yield, round(r.predicted_yield / 1000.0, 2),
            r.created_at.strftime('%Y-%m-%d %H:%M:%S')
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=yield_predictions_report.csv"}
    )

@app.route('/reports/export/recommendations')
def export_recommendations_csv():
    records = RecommendationHistory.query.order_by(RecommendationHistory.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temp (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)', 'Top 1 Crop', 'Top 1 Prob %', 'Top 2 Crop', 'Top 2 Prob %', 'Top 3 Crop', 'Top 3 Prob %', 'Created At'])

    for r in records:
        writer.writerow([
            r.id, r.nitrogen, r.phosphorus, r.potassium, r.temperature, r.humidity, r.ph, r.rainfall,
            r.top1_crop, r.top1_prob, r.top2_crop, r.top2_prob, r.top3_crop, r.top3_prob,
            r.created_at.strftime('%Y-%m-%d %H:%M:%S')
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=crop_recommendations_report.csv"}
    )

# =========================================================================
# SYSTEM, API & HEALTH CHECK
# =========================================================================
@app.route('/health')
def health_check():
    """Health check endpoint for Render monitoring and uptime checks."""
    db_status = "connected"
    try:
        SoilRecord.query.count()
    except Exception as e:
        db_status = f"error: {str(e)}"

    models_loaded = len(ml_service.load_all_models())

    return jsonify({
        'status': 'healthy',
        'service': 'Smart Agriculture ML Web Application',
        'database': db_status,
        'database_type': 'PostgreSQL' if 'postgresql' in app.config['SQLALCHEMY_DATABASE_URI'] else 'SQLite',
        'models_available': models_loaded >= 6,
        'total_models_and_artifacts': models_loaded,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/api/stats')
def api_stats():
    """JSON API for dashboard and dynamic charts."""
    weather_records = WeatherRecord.query.order_by(WeatherRecord.created_at.asc()).limit(12).all()
    soil_records = SoilRecord.query.all()

    weather_data = {
        'labels': [w.recorded_date.strftime('%b %d') if w.recorded_date else f'W#{w.id}' for w in weather_records],
        'temperatures': [w.temperature for w in weather_records],
        'humidities': [w.humidity for w in weather_records],
        'rainfalls': [w.rainfall for w in weather_records]
    }

    avg_n = round(sum(s.nitrogen for s in soil_records) / len(soil_records), 1) if soil_records else 60.0
    avg_p = round(sum(s.phosphorus for s in soil_records) / len(soil_records), 1) if soil_records else 45.0
    avg_k = round(sum(s.potassium for s in soil_records) / len(soil_records), 1) if soil_records else 45.0

    return jsonify({
        'weather': weather_data,
        'nutrients': {'N': avg_n, 'P': avg_p, 'K': avg_k}
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
