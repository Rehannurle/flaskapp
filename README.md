# Agricultural Prediction System 🌾

A comprehensive web application that helps farmers make data-driven decisions through various predictive models for crop recommendation, rainfall prediction, and crop yield estimation.

## Live Demo 🌐
Check out the live application: [FarmSphere](https://farmsphere-nugx.onrender.com)

## Features 🌟

- **Crop Recommendation System**: Get personalized crop suggestions based on soil composition and environmental factors
- **Rainfall Prediction**: Advanced weather forecasting with 5-day predictions
- **Crop Yield Prediction**: Estimate crop yields based on various agricultural parameters
- **User Authentication**: Secure login and registration system
- **User Profile**: Track prediction history and account statistics
- **Weather Integration**: Real-time weather data using OpenWeatherMap API

## Tech Stack 💻

- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy
- **Authentication**: Flask-Login
- **Machine Learning**: XGBoost, Scikit-learn
- **API Integration**: OpenWeatherMap API
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render

## Installation 🚀

1. Clone the repository
```bash
git clone https://github.com/yourusername/agricultural-prediction-system.git
cd agricultural-prediction-system
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
# Create a .env file and add:
OPENWEATHERMAP_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

5. Initialize the database
```bash
flask db init
flask db migrate
flask db upgrade
```

6. Run the application
```bash
python app.py
```

## Project Structure 📁

```
agricultural-prediction-system/
├── app.py                 # Main application file
├── models/               # Machine learning models
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
├── CropYieldModel.py    # Crop yield prediction model
└── requirements.txt     # Project dependencies
```

## Machine Learning Models 🤖

- **Crop Recommendation**: Logistic Regression model trained on soil and climate data
- **Rainfall Prediction**: XGBoost model for weather forecasting
- **Yield Prediction**: Custom model considering multiple agricultural factors

## API Reference 📚

### Crop Recommendation
```http
POST /predict
Content-Type: application/x-www-form-urlencoded

Parameters:
- N: Nitrogen content
- P: Phosphorus content
- K: Potassium content
- temperature: Temperature in celsius
- humidity: Humidity percentage
- ph: Soil pH
- rainfall: Rainfall in mm
```

### Rainfall Prediction
```http
GET /predict_rain?lat={latitude}&lon={longitude}

Parameters:
- lat: Latitude
- lon: Longitude
```

### Yield Prediction
```http
POST /predict_yield
Content-Type: application/json

{
    "crop": "crop_name",
    "area": float,
    "rainfall": float,
    "pesticide": float,
    "soil_ph": float,
    "fertilizer": float,
    "temperature": float
}
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- OpenWeatherMap API for weather data
- Scikit-learn and XGBoost communities
- Flask and its extension maintainers
- Render for hosting the live application

## Contact 📧

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/agricultural-prediction-system](https://github.com/yourusername/agricultural-prediction-system)  
Live Demo: [https://farmsphere-nugx.onrender.com](https://farmsphere-nugx.onrender.com)
