import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class CropYieldModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_importance = None
        
    def prepare_data(self, data):
        """Prepare data for training/prediction"""
        # Create a copy of the data
        df = data.copy()
        
        # Encode categorical variables
        df['crop_encoded'] = self.label_encoder.transform(df['crop'])
        
        # Scale numerical features
        numerical_features = ['area', 'rainfall', 'pesticide', 'soil_ph', 'fertilizer', 'temperature']
        df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df
    
    def predict(self, input_data):
        """Make predictions on new data"""
        # Prepare input data
        df = self.prepare_data(input_data)
        
        # Extract features
        X = df[['area', 'rainfall', 'pesticide', 'soil_ph', 'fertilizer', 'temperature', 'crop_encoded']]
        
        # Make prediction
        prediction = self.best_model.predict(X)
        
        return prediction
    
    @classmethod
    def load_model(cls, path='crop_yield_model.joblib'):
        """Load a trained model"""
        model_data = joblib.load(path)
        instance = cls()
        instance.best_model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_importance = model_data['feature_importance']
        return instance