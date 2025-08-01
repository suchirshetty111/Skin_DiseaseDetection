
# 🧬 AI-Based Skin Disease Detection System

An intelligent web-based application that uses deep learning to classify various skin diseases from uploaded images. It provides predicted disease, precautions, and medical guidance with a professional and interactive interface.

## 🌐 Live Preview (optional)
> [🔗 Netlify/Vercel Link Here](#)

## 🚀 Features

- ✅ AI-powered image classification for **23 skin diseases**
- 📷 Image upload with real-time prediction
- 💊 Disease-specific **description, symptoms, causes, prevention, and medicines**
- 🩺 Integrated Google search for nearby dermatologists
- 💎 Modern 3D UI with glassmorphism and responsive layout
- 📁 Flask backend with TensorFlow model integration

## 🖼️ Supported Disease Classes
```
Acne, Actinic Keratosis, Benign Tumors, Bullous, Candidiasis,
Drug Eruption, Eczema, Infestations/Bites, Lichen, Lupus,
Moles, Psoriasis, Rosacea, Seborrheic Keratoses, Skin Cancer,
Sun/Sunlight Damage, Tinea, Unknown/Normal, Vascular Tumors,
Vasculitis, Vitiligo, Warts
```

## 📂 Project Structure

```
SkinDiseaseDetection/
│
├── backend/
│   ├── app.py                  # Flask App
│   ├── disease_data.py         # Disease info + medicines
│   ├── model/
│   │   └── skin_disease_model.h5
│   ├── static/
│   │   ├── css/
│   │   │   ├── styles.css
│   │   │   └── result.css
│   │   ├── images/
│   │   │   └── logo.png, background.jpg
│   │   └── uploads/
│   └── templates/
│       ├── index.html
│       ├── result.html
│       └── disease.html
```

## 🛠️ Requirements

- Python 3.7+
- Flask
- TensorFlow / Keras
- NumPy
- Pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

> _Note: You may need to set the env variable:_
```bash
set TF_ENABLE_ONEDNN_OPTS=0   # For Windows
```

## ⚙️ How to Run Locally

```bash
cd backend
python app.py
```

Then open: `http://127.0.0.1:5000/` in your browser.

## 📈 Model Training (Optional)

If you want to train your own model, use the `train.py` file. It expects:
- Labeled image dataset
- Preprocessing to resize images to 128x128 or 224x224

## 💡 Credits

- Disease info collected and formatted from **DermNet**, **WebMD**, and other verified sources.
- UI inspired by modern glassmorphism and gradient UI kits.
- Built with ❤️ using Python, Flask, HTML/CSS, TensorFlow.

## 📜 License

This project is for **educational purposes only**. Contact the author for commercial licensing.
