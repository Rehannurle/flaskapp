import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def verify_model_file(model_path):
    """Verify if the model file contains all required components"""
    try:
        # Load the model file
        print(f"Attempting to load model from: {model_path}")
        model_data = joblib.load(model_path)
        
        # Check if it's a dictionary
        if not isinstance(model_data, dict):
            print("❌ Error: Model file is not in the correct format (should be a dictionary)")
            return False
        
        # Check for required components
        required_components = ['model', 'scaler', 'label_encoder', 'feature_importance']
        for component in required_components:
            if component not in model_data:
                print(f"❌ Missing required component: {component}")
                return False
            print(f"✅ Found component: {component}")
        
        # Verify scaler
        if not isinstance(model_data['scaler'], StandardScaler):
            print("❌ Error: Scaler is not a StandardScaler instance")
            return False
        
        # Verify label encoder
        if not isinstance(model_data['label_encoder'], LabelEncoder):
            print("❌ Error: Label encoder is not a LabelEncoder instance")
            return False
        
        # Verify feature importance exists
        if not isinstance(model_data['feature_importance'], (np.ndarray, list)):
            print("❌ Error: Feature importance is not in the correct format")
            return False
        
        print("\n✅ All components verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model file: {str(e)}")
        return False

def save_model_components(model, scaler, label_encoder, feature_importance, output_path):
    """Save all model components in the correct format"""
    try:
        model_data = {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_importance': feature_importance
        }
        
        joblib.dump(model_data, output_path)
        print(f"✅ Model components saved successfully to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving model components: {str(e)}")
        return False

if __name__ == "__main__":
    # Replace with your model path
    MODEL_PATH = '/Users/rehannurle/Downloads/crop/crop_yield_model.joblib'
    
    # Verify existing model file
    print("Verifying existing model file...")
    print("-" * 50)
    verify_model_file(MODEL_PATH)
    
    # If you need to save a new model file, uncomment and modify this section:
    """
    # Example of saving a new model file
    from sklearn.ensemble import RandomForestRegressor  # or your actual model type
    
    # Initialize components
    model = RandomForestRegressor()  # Your trained model
    scaler = StandardScaler()  # Your fitted scaler
    label_encoder = LabelEncoder()  # Your fitted label encoder
    feature_importance = model.feature_importances_  # Your feature importance values
    
    # Save components
    save_model_components(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_importance=feature_importance,
        output_path='new_crop_yield_model.joblib'
    )
    """