import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to trained model
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_model.h5")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# Loading trained model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Define class labels
class_labels = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous", "Candidiasis",
    "Drug Eruption", "Eczema", "Infestations/Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrheic Keratoses", "Skin Cancer",
    "Sun/Sunlight Damage", "Tinea", "Unknown/Normal", "Vascular Tumors",
    "Vasculitis", "Vitiligo", "Warts"
]

# Disease-specific medicines
DISEASE_MEDICINE = {
    "Acne": ["Benzoyl peroxide", "Antibiotics (Doxycycline)", "Salicylic acid"],
    "Actinic Keratosis": ["Fluorouracil cream", "Imiquimod cream"],
    "Benign Tumors": ["Observation or Surgical Removal if Suspicious"],
    "Bullous": ["Corticosteroids (Prednisone)", "Antihistamines"],
    "Candidiasis": ["Antifungal creams (Clotrimazole)", "Fluconazole"],
    "Drug Eruption": ["Antihistamines", "Withdrawal of causative medication"],
    "Eczema": ["Moisturizers", "Topical corticosteroids"],
    "Infestations/Bites": ["Antihistamines", "Calamine Lotion"],
    "Lichen": ["Corticosteroids", "Antihistamines"],
    "Lupus": ["Hydroxychloroquine", "NSAIDs"],
    "Moles": ["Observation or Surgical Removal if Suspicious"],
    "Psoriasis": ["Methotrexate", "Corticosteroids", "Calcipotriene"],
    "Rosacea": ["Antibiotics (Doxycycline)", "Brimonidine"],
    "Seborrheic Keratoses": ["Cryotherapy (Freezing)", "Curettage"],
    "Skin Cancer": ["Surgical Removal (Mohs)", "Radiotherapy"],
    "Sun/Sunlight Damage": ["Sunscreen (SPF 30+)", "Antioxidants (Vitamin C)"],
    "Tinea": ["Antifungal creams (Terbinafine)", "Antifungal pills (Fluconazole)"],
    "Unknown/Normal": [],
    "Vascular Tumors": ["Observation", "Surgical removal if growing"],
    "Vasculitis": ["Corticosteroids (Prednisone)", "Antihistamines"],
    "Vitiligo": ["Corticosteroids", "Calcineurin inhibitors"],
    "Warts": ["Salicylic acid", "Cryotherapy"]
}

# Disease-specific information
DISEASE_INFO = {
    "Acne": {
        "Description": "Acne occurs when hair follicles become plugged with oil and dead skin cells.",
        "Symptoms": "Whiteheads, Blackheads, Pimples, Cysts.",
        "Causes": "Hormonal imbalances, excess oil production, stress.",
        "Prevention": "Keep skin clean, avoid heavy cosmetics, drink plenty of water."
    },
    "Actinic Keratosis": {
        "Description": "Actinic Keratosis refers to rough, scaly patches due to excess sun exposure.",
        "Symptoms": "Small, raised, scaly lesion.",
        "Causes": "Prolonged UV radiation.",
        "Prevention": "Limit sun exposure, apply sunscreen, wear protective clothing."
    },
    "Benign Tumors": {
        "Description": "Benign Tumors are non-cancerous growths on the skin.",
        "Symptoms": "Small, firm lumps or growths.",
        "Causes": "Usually unknown, related to growing skin cells.",
        "Prevention": "Routine skin checks and protective measures."
    },
    "Bullous": {
        "Description": "Bullous disorders manifest as blisters on the skin.",
        "Symptoms": "Fluid-filled blisters, sores.",
        "Causes": "Autoimmune disorders, medication reactions, trauma.",
        "Prevention": "Avoid skin trauma and promptly see a dermatologist if blisters appear."
    },
    "Candidiasis": {
        "Description": "Candidiasis is a fungal skin infection, typically by Candida species.",
        "Symptoms": "Red rash, itching, soreness.",
        "Causes": "Warm, moist skin; diabetes; antibiotic usage.",
        "Prevention": "Keep skin clean and dry; manage diabetes; avoid antibiotic overuse."
    },
    "Drug Eruption": {
        "Description": "Drug Eruptions are skin reactions due to medication.",
        "Symptoms": "Rash, hives, redness.",
        "Causes": "Adverse reaction to medication.",
        "Prevention": "Avoid medications that previously triggered reactions; inform clinician."
    },
    "Eczema": {
        "Description": "Eczema is a chronic skin condition causing itchiness and redness.",
        "Symptoms": "Dry skin, redness, sores.",
        "Causes": "Allergy, stress, or skin barrier dysfunction.",
        "Prevention": "Moisturize frequently, avoid allergens, reduce stress."
    },
    "Infestations/Bites": {
        "Description": "Infestations or insect bites by mites, mosquitoes, or other pests.",
        "Symptoms": "Redness, itchiness, swelling.",
        "Causes": "Parasites or insect bites.",
        "Prevention": "Clean living spaces, insect repellent, avoid scratching."
    },
    "Lichen": {
        "Description": "Lichen Planus involves purple-colored, flat-topped bumps.",
        "Symptoms": "Itchy, purple-colored rash.",
        "Causes": "Immune-related disorders.",
        "Prevention": "Reduce stress, avoid skin injuries."
    },
    "Lupus": {
        "Description": "Lupus is a chronic autoimmune disease affecting skin and other body parts.",
        "Symptoms": "Rash (butterfly rash), sores, joint pain.",
        "Causes": "Autoimmune reaction.",
        "Prevention": "Avoid excess sun, stress; follow clinician’s treatment."
    },
    "Moles": {
        "Description": "Moles are clusters of pigmented skin cells.",
        "Symptoms": "Small, dark spots.",
        "Causes": "Sun exposure, genetics.",
        "Prevention": "Watch for ABCD symptoms (asymmetry, borders, color, diameter)."
    },
    "Psoriasis": {
        "Description": "Psoriasis involves the overproduction of skin cells.",
        "Symptoms": "Red, scaly patches.",
        "Causes": "Immune dysfunction, stress.",
        "Prevention": "Moisturize, avoid stress, follow clinician’s guidance."
    },
    "Rosacea": {
        "Description": "Rosacea is a condition causing redness and visible blood vessels.",
        "Symptoms": "Flushing, redness, small pimples.",
        "Causes": "Triggers include stress, alcohol, spicy food.",
        "Prevention": "Identify and avoid triggers, apply sunscreen."
    },
    "Seborrheic Keratoses": {
        "Description": "Seborrheic Keratoses are non-cancerous skin growths.",
        "Symptoms": "Brown or black wart-like growths.",
        "Causes": "Aging, genetics.",
        "Prevention": "Usually not preventable; see clinician if worried."
    },
    "Skin Cancer": {
        "Description": "Skin Cancer involves abnormal growth of skin cells.",
        "Symptoms": "Sore that doesn’t heal, new or changing lesion.",
        "Causes": "Prolonged UV exposure.",
        "Prevention": "Limit sun exposure, apply sunscreen, perform skin checks."
    },
    "Sun/Sunlight Damage": {
        "Description": "Damage due to UV radiation.",
        "Symptoms": "Sunburn, redness, peeling.",
        "Causes": "Prolonged UV exposure.",
        "Prevention": "Sunscreen, protective clothing, avoid tanning beds."
    },
    "Tinea": {
        "Description": "Ringworm — a fungal skin condition.",
        "Symptoms": "Ring-shaped rash, itching.",
        "Causes": "Close skin contact, damp skin.",
        "Prevention": "Keep skin clean and dry; avoid sharing towels."
    },
    "Unknown/Normal": {
        "Description": "Normal skin or undiagnosed condition.",
        "Symptoms": "No abnormalities.",
        "Causes": "Not applicable.",
        "Prevention": "General skin care routines."
    },
    "Vascular Tumors": {
        "Description": "Vascular Tumors are abnormal growths of blood vessels.",
        "Symptoms": "Red or purple-colored lesion.",
        "Causes": "Proliferation of blood vessels.",
        "Prevention": "Usually not preventable; observation recommended."
    },
    "Vasculitis": {
        "Description": "Vasculitis involves swelling of blood vessels.",
        "Symptoms": "Rash, sores, weakness.",
        "Causes": "Autoimmune disorders, medication side effects.",
        "Prevention": "Management by clinician, control underlying condition."
    },
    "Vitiligo": {
        "Description": "Vitiligo involves the destruction of pigmentation.",
        "Symptoms": "White patches on skin.",
        "Causes": "Autoimmune process.",
        "Prevention": "Sun protection, skin care routines, consultation with dermatologist."
    },
    "Warts": {
        "Description": "Warts are skin growths due to human papilloma virus.",
        "Symptoms": "Small, raised growths.",
        "Causes": "Viral Infection (HPV).",
        "Prevention": "Avoid touching or spreading warts, proper hygiene."
    },
}

# Set upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
def allowed_file(file):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return (
        "." in file.filename and
        file.filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS and
        file.mimetype.startswith("image/")
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded!", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected!", 400

    if not allowed_file(file):
        return "Invalid file format.", 400

    file.filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        # Resize to (224, 224) to match training
        image = load_img(file_path, target_size=(224, 224)).convert("RGB")
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        predicted_label = class_labels[predicted_class]

        return render_template("result.html", label=predicted_label, image_url=file.filename)

    except Exception as e:
        return f"❌ Prediction failed: {str(e)}", 500


@app.route('/disease/<disease_name>')
def disease(disease_name):
    medicines = DISEASE_MEDICINE.get(disease_name, [])
    info = DISEASE_INFO.get(disease_name, {})
    return render_template('disease.html',
                             disease_name=disease_name,
                             medicines=medicines,
                             info=info)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
