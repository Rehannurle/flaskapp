from flask import Flask, request, render_template
import joblib
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Any
import requests
from datetime import datetime, timedelta
import json
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from CropYieldModel import CropYieldModel
import warnings
from sklearn.exceptions import InconsistentVersionWarning
# Adding these imports to your for backend 
from flask import redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import re
#try something new 
from flask import send_file, abort

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'd0b5f9ce92a00bdfd55390bdd649806d2b1ae251d08fc64212738ec1b3de443a'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Suppress specific warnings
warnings.filterwarnings('ignore', message=r'.*XGBoost.*', category=UserWarning)
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', message=r'.*X does not have valid feature names.*')

#changes 
# Load the saved XGBoost model and scaler
MODEL_PATH = 'xgboost_rain_prediction_model.pkl'
SCALER_PATH = 'scaler.pkl'

xgb_model = joblib.load(MODEL_PATH)
print("XGBoost Model Loaded Successfully")
scaler = joblib.load(SCALER_PATH)
print("Scaler Loaded Successfully")

# OpenWeatherMap API key (replace 'your_openweathermap_api_key' with your actual API key)
API_KEY = 'ae8ecb91693c1556fb3d12f4867c9e76'

# Fetch weather data by latitude and longitude
def fetch_weather_data_by_coordinates(lat: float, lon: float):
    """Fetch weather data using latitude and longitude."""
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    

# Caching decorator
@lru_cache(maxsize=100)
def cached_api_call(url):
    response = requests.get(url)
    if response.status_code == 200:
        return json.dumps(response.json())
    return None

 #Function to fetch both current and 5-day forecast weather data
def fetch_weather_data(lat: float, lon: float):
    url = f'http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current = data['list'][0]
        forecast = data['list'][::8]  # Get data for every 24 hours (3-hour steps * 8 = 24 hours)
        return {
            'current': current,
            'forecast': forecast[:5]  # Limit to 5 days
        }
    return None

# Function to predict rain based on weather data
def predict_rain(features):
    input_data = np.array([features])
    input_data_scaled = scaler.transform(input_data)
    rain_probability = xgb_model.predict_proba(input_data_scaled)[0][1]
    return rain_probability


# Function to handle rain prediction based on coordinates
# Function to handle rain prediction based on coordinates
def predict_rain_by_coordinates(lat: float, lon: float):
    weather_data = fetch_weather_data(lat, lon)

    if weather_data:
        current = weather_data['current']
        current_features = [
            current['main']['temp'], 
            current['main']['humidity'], 
            current['main']['pressure'],
            current['wind']['speed'], 
            current['main']['temp'] - ((100 - current['main']['humidity']) / 5),
            current['main']['temp'] + 0.5 * current['main']['humidity'], 
            lat, 
            lon
        ]
        current_rain_probability = predict_rain(current_features)

        forecast_predictions = []
        for day in weather_data['forecast']:
            day_features = [
                day['main']['temp'], 
                day['main']['humidity'], 
                day['main']['pressure'], 
                day['wind']['speed'],
                day['main']['temp'] - ((100 - day['main']['humidity']) / 5),
                day['main']['temp'] + 0.5 * day['main']['humidity'], 
                lat, 
                lon
            ]
            day_rain_probability = predict_rain(day_features)
            forecast_predictions.append({
                'date': datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d'),
                'rain_probability': day_rain_probability,
                'temp': day['main']['temp'],
                'humidity': day['main']['humidity'],
                'description': day['weather'][0]['description'],
                'wind_speed': day['wind']['speed'],
                'uv_index': day.get('uvi', 'N/A'),  # UV index might not be available in all API responses
                'icon': day['weather'][0]['icon']
            })

        return {
            'current': {
                'rain_probability': current_rain_probability,
                'temp': current['main']['temp'],
                'humidity': current['main']['humidity'],
                'description': current['weather'][0]['description'],
                'wind_speed': current['wind']['speed'],
                'uv_index': current.get('uvi', 'N/A'),
                'icon': current['weather'][0]['icon']
            },
            'forecast': forecast_predictions
        }
    
    return None



API_KEY = 'ae8ecb91693c1556fb3d12f4867c9e76'  # Replace with your OpenWeatherMap API key
# Load the saved model using joblib
MODEL_PATH ='LogisticRegression.joblib' 

def load_model(model_path: str):
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except (OSError, IOError, joblib.JoblibException) as e:
            app.logger.error(f"Error loading model: {e}")
    else:
        app.logger.error(f"Model file not found: {model_path}")
    return None

LogReg = load_model(MODEL_PATH)

# Crop information dictionary
crop_info: Dict[str, Dict[str, str]] = {
    'rice': {
        'time': '3-6 months',
        'fertilizer': 'NPK (Nitrogen, Phosphorus, Potassium) fertilizer',
        'practices': 'Maintain proper water management, use quality seeds, and practice crop rotation.',
        'image':'static/images/rice.jpg'
    },
    'maize':{
        'time': '3-6 months',
        'fertilizer': 'NPK (Nitrogen, Phosphorus, Potassium) fertilizer',
        'practices': 'Maintain proper water management, use quality seeds, and practice crop rotation.',
        'image':'static/images/maize.jpg'
     },
    'chickpea': {
        'time': '90-120 days',
        'fertilizer': 'Phosphorus-rich fertilizer',
        'practices': 'Plant in well-drained soil, practice intercropping, and manage pests organically.',
        'image': 'static/images/chickpea.jpg'
    },
    # Add information for other crops here
    'kidneybeans': {
        'time_to_grow': '3 to 4 months',
        'fertilizer': 'NPK fertilizers and compost',
        'practice': 'Use trellising and regular irrigation for better growth.',
        'image': 'static/images/kidneybean.jpg'
    },
    'pigeonpeas': {
        'time_to_grow': '5 to 6 months',
        'fertilizer': 'Phosphate and Potassium',
        'practice': 'Proper spacing and crop rotation are recommended.',
        'image': 'static/images/peigionbean.jpg'
    },
    'mothbeans': {
        'time_to_grow': '3 to 4 months',
        'fertilizer': 'Organic manure and Phosphate fertilizers',
        'practice': 'Intercropping and minimal irrigation for best results.',
        'image': 'static/images/mothbean.jpg'
    },
    'mungbean': {
        'time_to_grow': '2 to 3 months',
        'fertilizer': 'Phosphorus and Nitrogen',
        'practice': 'Timely irrigation and pest control improve yields.',
        'image': 'static/images/mungbean.jpg'
    },
    'blackgram': {
        'time_to_grow': '3 to 4 months',
        'fertilizer': 'Potassium and Phosphorus',
        'practice': 'Ensure good drainage and weed control.',
        'image': 'static/images/Black_gram.jpg'
    },
    'lentil': {
        'time_to_grow': '3 to 4 months',
        'fertilizer': 'Phosphorus and Potassium',
        'practice': 'Apply fertilizers during sowing and avoid waterlogging.',
        'image': 'image.jpg'
    },
    'pomegranate': {
        'time_to_grow': '5 to 6 months',
        'fertilizer': 'Potassium and organic compost',
        'practice': 'Regular pruning and pest control improve fruit quality.',
        'image': 'static/images/pomegranate.jpeg'
    },
    'banana': {
        'time_to_grow': '9 to 12 months',
        'fertilizer': 'Nitrogen, Phosphorus, and Potassium',
        'practice': 'Mulching and irrigation are essential for healthy growth.',
        'image': 'static/images/banana.jpeg'
    },
    'mango': {
        'time_to_grow': '6 to 9 months',
        'fertilizer': 'Urea, Potassium, and Phosphorus',
        'practice': 'Pruning and pest control improve yields.',
        'image': 'static/images/mango.jpeg'
    },
    'grapes': {
        'time_to_grow': '6 to 8 months',
        'fertilizer': 'Phosphorus and Potassium',
        'practice': 'Trellising and pruning help improve grape quality.',
        'image': 'static/images/grapes.jpeg'
    },
    'watermelon': {
        'time_to_grow': '3 to 4 months',
        'fertilizer': 'Nitrogen and organic compost',
        'practice': 'Ensure proper irrigation and weed control.',
        'image': 'static/images/watermelon.jpeg'
    },
    'muskmelon': {
        'time_to_grow': '3 to 4 months',
        'fertilizer': 'Nitrogen, Phosphorus, and Potassium',
        'practice': 'Timely irrigation and mulching improve yields.',
        'image': 'static/images/muskmelon.jpeg'
    },
    'apple': {
        'time_to_grow': '6 to 10 months',
        'fertilizer': 'Phosphorus, Potassium, and Nitrogen',
        'practice': 'Pruning and pest management improve production.',
        'image': 'static/images/apple.jpeg'
    },
    'orange': {
        'time_to_grow': '8 to 10 months',
        'fertilizer': 'Nitrogen, Phosphorus, and Potassium',
        'practice': 'Water management and timely pruning are essential.',
        'image': 'static/images/orange.jpeg'
    },
    'papaya': {
        'time_to_grow': '6 to 9 months',
        'fertilizer': 'Organic manure and Potassium',
        'practice': 'Proper irrigation and pest control enhance fruit quality.',
        'image': 'static/images/papaya.jpeg'
    },
    'coconut': {
        'time_to_grow': '10 to 12 months',
        'fertilizer': 'Organic manure and Potassium',
        'practice': 'Regular mulching and irrigation improve coconut yield.',
        'image': 'static/images/coconut.jpeg'
    },
    'cotton': {
        'time_to_grow': '6 to 7 months',
        'fertilizer': 'Nitrogen, Phosphorus, and Potassium',
        'practice': 'Weed control and regular irrigation are important.',
        'image': 'static/images/contact.jpg'
    },
    'jute': {
        'time_to_grow': '4 to 6 months',
        'fertilizer': 'Organic manure and Nitrogen',
        'practice': 'Timely sowing and weed management improve quality.',
        'image': 'static/images/jute.jpeg'
    },
    
    'coffee': {
        'time': '3-4 years for first harvest, then annual',
        'fertilizer': 'Balanced NPK fertilizer with micronutrients',
        'practices': 'Provide shade, prune regularly, and implement integrated pest management.',
        'image': 'static/images/coffee.jpeg'
    }
}




@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_url = None
    
    if LogReg is None:
        return render_template('crop_recommendation.html', 
                             prediction_text="Model is not available. Please try again later.")

    try:
        # Gather form data
        data = request.form.to_dict()
        features = pd.DataFrame({
            'N': [float(data['N'])],
            'P': [float(data['P'])],
            'K': [float(data['K'])],
            'temperature': [float(data['temperature'])],
            'humidity': [float(data['humidity'])],
            'ph': [float(data['ph'])],
            'rainfall': [float(data['rainfall'])]
        })

        # Scale the features
        scaler = LogReg['scaler']
        scaled_features = scaler.transform(features)

        # Make prediction using the model and decode it
        model = LogReg['model']
        le = LogReg['label_encoder']
        
        # Get prediction probabilities
        proba = model.predict_proba(scaled_features)
        prediction_encoded = model.predict(scaled_features)
        prediction = le.inverse_transform(prediction_encoded)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(proba[0])[-3:][::-1]
        top_3_crops = [(le.inverse_transform([idx])[0], proba[0][idx]) for idx in top_3_indices]
        
        # Format the result
        if prediction.lower() in crop_info:
            info = crop_info[prediction.lower()]
            result = f"""
            Top Recommended Crop: {prediction.capitalize()} ({top_3_crops[0][1]:.1%} confidence)
            
            Alternative Options:
            2. {top_3_crops[1][0].capitalize()} ({top_3_crops[1][1]:.1%} confidence)
            3. {top_3_crops[2][0].capitalize()} ({top_3_crops[2][1]:.1%} confidence)
            
            For {prediction.capitalize()}:
            Growing Time: {info['time']}
            Recommended Fertilizer: {info['fertilizer']}
            Best Practices: {info['practices']}
            """
            image_url = info.get('image')
        else:
            result = f"""
            Top Recommended Crop: {prediction.capitalize()} ({top_3_crops[0][1]:.1%} confidence)
            
            Alternative Options:
            2. {top_3_crops[1][0].capitalize()} ({top_3_crops[1][1]:.1%} confidence)
            3. {top_3_crops[2][0].capitalize()} ({top_3_crops[2][1]:.1%} confidence)
            
            Detailed information not available for this crop.
            """
            image_url = None

        return render_template('crop_recommendation.html', 
                             prediction_text=result, 
                             image_url=image_url)

    except ValueError as e:
        app.logger.error(f"Input error: {e}")
        return render_template('crop_recommendation.html', 
                             prediction_text="Input error: Please ensure all fields contain valid numbers.")
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return render_template('crop_recommendation.html', 
                             prediction_text="An unexpected error occurred. Please try again later.")



@app.route('/get_weather_by_location')
def get_weather_by_location():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if lat and lon:
        url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed']
            }
            return jsonify(weather_info)
        else:
            return jsonify({'error': 'Unable to fetch weather data'}), response.status_code
    else:
        return jsonify({'error': 'Latitude and longitude not provided'}), 400
    
#                                 <<<<<<<<<<<<<<<<<  SECTION FOR RAINFALL PREDICTION   >>>>>>>>>>>>>>>>>>>>>>
# /changes
@app.route('/rainfall_prediction')
def rainfall_prediction():
    return render_template('rainfall_prediction.html')

@app.route('/predict_rain', methods=['GET'])
def rain_prediction():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    if lat is not None and lon is not None:
        rain_data = predict_rain_by_coordinates(lat, lon)
        if rain_data is not None:
            current = rain_data['current']
            current_rain_probability = current['rain_probability']
            
            def get_rain_description(prob):
                if prob > 0.9:
                    return 'Severe weather warning: Heavy rain expected. Stay safe!'
                elif prob > 0.75:
                    return 'High chance of rain. Don\'t forget your umbrella!'
                elif prob > 0.6:
                    return 'Overcast with a good chance of showers.'
                elif prob > 0.45:
                    return 'Slight chance of rain, maybe some drizzles.'
                elif prob > 0.3:
                    return 'Mostly cloudy, slight chance of a light shower.'
                elif prob > 0.1:
                    return 'Low chance of rain, but skies may be overcast.'
                else:
                    return 'Very low chance of rain, clear skies expected.'

            current_result = {
                'prediction': get_rain_description(current_rain_probability),
                'probability': f"{current_rain_probability:.2f}",
                'temperature': current['temp'],
                'humidity': current['humidity'],
                'description': current['description'],
                'wind_speed': current['wind_speed'],
                'uv_index': current['uv_index'],
                'icon': current['icon']
            }

            forecast_results = []
            for day in rain_data['forecast']:
                forecast_results.append({
                    'date': day['date'],
                    'prediction': get_rain_description(day['rain_probability']),
                    'probability': f"{day['rain_probability']:.2f}",
                    'temperature': day['temp'],
                    'humidity': day['humidity'],
                    'description': day['description'],
                    'wind_speed': day['wind_speed'],
                    'uv_index': day['uv_index'],
                    'icon': day['icon']
                })

            return jsonify({
                'current': current_result,
                'forecast': forecast_results
            })
        else:
            return jsonify({'error': 'Unable to fetch weather data'})
    else:
        return jsonify({'error': 'Location not provided'}), 400


#                                 <<<<<<<<<<<<<<<<<  SECTION FOR YIELD PREDICTION   >>>>>>>>>>>>>>>>>>>>>>

from flask import Flask, request, jsonify, render_template
import pandas as pd
from CropYieldModel import CropYieldModel



# Initialize the model
try:
    crop_yield_model = CropYieldModel.load_model('crop_yield_model.joblib')
    print("Crop Yield Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    crop_yield_model = None

@app.route('/predict_yield', methods=['GET'])
def yield_form():
    """Render the prediction form webpage"""
    return render_template('predict_yield.html')

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    """
    Handle POST requests for crop yield predictions
    
    Expected JSON input format:
    {
        "crop": string,
        "area": float,
        "rainfall": float,
        "pesticide": float,
        "soil_ph": float,
        "fertilizer": float,
        "temperature": float
    }
    """
    if crop_yield_model is None:
        return jsonify({
            'error': 'Model not initialized. Please try again later.'
        }), 500

    try:
        data = request.get_json()
        
        # Validate that all required fields are present
        required_fields = ['crop', 'area', 'rainfall', 'pesticide', 'soil_ph', 'fertilizer', 'temperature']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields. Please provide all required parameters.'
            }), 400

        # Create DataFrame with input data
        input_data = pd.DataFrame({
            'crop': [str(data['crop'])],
            'area': [float(data['area'])],
            'rainfall': [float(data['rainfall'])],
            'pesticide': [float(data['pesticide'])],
            'soil_ph': [float(data['soil_ph'])],
            'fertilizer': [float(data['fertilizer'])],
            'temperature': [float(data['temperature'])]
        })

        # Validate numeric values
        numeric_fields = {
            'area': 'Area',
            'rainfall': 'Rainfall',
            'pesticide': 'Pesticide',
            'fertilizer': 'Fertilizer',
            'temperature': 'Temperature'
        }
        
        for field, display_name in numeric_fields.items():
            if input_data[field].iloc[0] <= 0:
                return jsonify({
                    'error': f'{display_name} must be a positive value.'
                }), 400

        # Validate soil pH range
        if not (0 <= input_data['soil_ph'].iloc[0] <= 14):
            return jsonify({
                'error': 'Soil pH must be between 0 and 14.'
            }), 400

        # Make prediction
        prediction = crop_yield_model.predict(input_data)
        yield_prediction = float(prediction[0])

        return jsonify({
            'yield_prediction': yield_prediction,
            'units': 'tonnes/hectare'
        })

    except ValueError as e:
        print(f"Value error: {str(e)}")
        return jsonify({
            'error': 'Invalid input: Please check that all inputs are valid numbers.'
        }), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred while processing your request.'
        }), 500

#                    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<      contact page section     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/contact.html')
def contact():
    return render_template('contact.html')
#                    <<<<<<<<<<<<<<<<<<<<<<<<<<<<< to get to home           >>>>>>>>>>>>>>>>>>>>>>>>>>>>>›››
@app.route('/index.html')
def index_page():
    return render_template('index.html')

#                   <<<<<<<<<<<<<<<<<<<<<<<<<<<<< to get to about          >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/about.html')
def about():
    return render_template('about.html')



from flask import Flask, send_from_directory

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static/favicon_io', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# @app.route('/plant_disease_detection')
# def plant_disease_detection():
#     return render_template('plant_disease_detection.html')

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Add additional fields for user profile if needed
    predictions_count = db.Column(db.Integer, default=0)
    recommendations_count = db.Column(db.Integer, default=0)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    


@login_manager.user_loader
def load_user(id):
    return db.session.get(User, int(id))

# Routes
@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to index
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            # Get the next page from the URL parameters, or default to index
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))

        if len(username) < 3 or len(username) > 20:
            flash('Username must be between 3 and 20 characters')
            return redirect(url_for('register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long')
            return redirect(url_for('register'))

        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/profile')
@login_required
def profile():
    from datetime import UTC
    if current_user.created_at.tzinfo is None:
        created_at = current_user.created_at.replace(tzinfo=UTC)
    else:
        created_at = current_user.created_at
        
    days_member = (datetime.now(UTC) - created_at).days
    return render_template('profile.html',
                         predictions_count=current_user.predictions_count,
                         recommendations_count=current_user.recommendations_count,
                         days_member=days_member)
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500



# Create database tables
def init_db():
    with app.app_context():
        db.create_all()



# Add this route to your app.py file
@app.route('/download_db_temporary/<secret_key>')
def download_db(secret_key):
    # Use a strong, random secret key as protection
    if secret_key != 'f7a8b9c1d2e3f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8':
        return abort(404)  # Return 404 to hide that this endpoint exists
    
    db_path = 'users.db'  # or 'instance/users.db' if it's in the instance folder
    if os.path.exists(db_path):
        return send_file(db_path, 
                       as_attachment=True,
                       download_name='users_backup.db')
    else:
        return f"Database file not found at {db_path}", 404

if __name__ == '__main__':
    init_db()  # Initialize database tables
    app.run(debug=True)


