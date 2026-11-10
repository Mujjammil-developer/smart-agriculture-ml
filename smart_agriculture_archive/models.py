from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class SoilRecord(db.Model):
    __tablename__ = 'soil_records'

    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(120), nullable=False, default='Field Block A')
    ph = db.Column(db.Float, nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.String(60), default='Loamy')
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def fertility_status(self):
        # Quick agronomic status
        score = 0
        if 6.0 <= self.ph <= 7.5:
            score += 1
        if self.nitrogen >= 40:
            score += 1
        if self.phosphorus >= 25:
            score += 1
        if self.potassium >= 25:
            score += 1

        if score >= 3:
            return ('High Fertility', 'success')
        elif score == 2:
            return ('Moderate Fertility', 'warning')
        else:
            return ('Low Fertility', 'danger')


class WeatherRecord(db.Model):
    __tablename__ = 'weather_records'

    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(120), nullable=False, default='Central Agro Station')
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    condition = db.Column(db.String(60), default='Sunny')
    recorded_date = db.Column(db.Date, default=datetime.utcnow().date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CropInfo(db.Model):
    __tablename__ = 'crop_information'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    category = db.Column(db.String(60), nullable=False)
    ideal_n = db.Column(db.Float, nullable=False)
    ideal_p = db.Column(db.Float, nullable=False)
    ideal_k = db.Column(db.Float, nullable=False)
    ideal_temp = db.Column(db.Float, nullable=False)
    ideal_humidity = db.Column(db.Float, nullable=False)
    ideal_ph = db.Column(db.Float, nullable=False)
    ideal_rainfall = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=True)
    season = db.Column(db.String(60), default='Kharif / Rabi')
    icon = db.Column(db.String(60), default='fa-leaf')


class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)  # 'Simple Linear Regression' or 'Multiple Linear Regression'
    rainfall = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    ph = db.Column(db.Float, nullable=True)
    nitrogen = db.Column(db.Float, nullable=True)
    phosphorus = db.Column(db.Float, nullable=True)
    potassium = db.Column(db.Float, nullable=True)
    predicted_yield = db.Column(db.Float, nullable=False)  # kg/hectare
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class RecommendationHistory(db.Model):
    __tablename__ = 'recommendation_history'

    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    top1_crop = db.Column(db.String(60), nullable=False)
    top1_prob = db.Column(db.Float, nullable=False)
    top2_crop = db.Column(db.String(60), nullable=False)
    top2_prob = db.Column(db.Float, nullable=False)
    top3_crop = db.Column(db.String(60), nullable=False)
    top3_prob = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class SoilSuitabilityHistory(db.Model):
    __tablename__ = 'soil_suitability_records'

    id = db.Column(db.Integer, primary_key=True)
    ph = db.Column(db.Float, nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    moisture = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    is_suitable = db.Column(db.Boolean, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    diagnosis = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
