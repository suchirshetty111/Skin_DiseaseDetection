import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the existing .h5 model
h5_model_path = os.path.join("model", "skin_disease_model.h5")
model = load_model(h5_model_path)

# Save as .keras format
keras_model_path = os.path.join("model", "skin_disease_model.keras")
model.save(keras_model_path)

# OR: Save as SavedModel format
saved_model_dir = os.path.join("model", "skin_disease_model_saved")
model.save(saved_model_dir, save_format="tf")

print("✅ Model conversion successful!")
