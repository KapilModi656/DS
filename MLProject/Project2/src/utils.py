import joblib
import tensorflow.keras as keras
def save_model(model, model_path):
    
    try:
        model.save(model_path)
        print(f"Model saved successfully at {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
def save_object(obj, file_path):
    try:
        joblib.dump(obj, file_path)
        print(f"Object saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving object: {e}")
def load_object(file_path):
    try:
        obj = joblib.load(file_path)
        print(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        return None