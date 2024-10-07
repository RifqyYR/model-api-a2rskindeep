import xgboost as xgb
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler
from django.core.files.storage import default_storage
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def extract_features(model, images):
    images_preprocessed = preprocess_input(images)
    features = model.predict(images_preprocessed)
    return features

def predict(classes, probabilities):
    labels = {
        0: "Jerawat",
        1: "Komedo",
        2: "Hiperpigmentasi",
        3: "Normal",
        4: "Berminyak",
        5: "Kemerahan"
    }
    predicted_label = labels.get(classes)
    predicted_probability = probabilities[classes]
    return predicted_label, predicted_probability

# Fungsi untuk klasifikasi gambar baru
def classify_image(image, feature_extractor, scaler, classifier):
    features = extract_features(feature_extractor, np.expand_dims(image, axis=0))
    features_scaled = scaler.transform(features)
    prediction_probabilities = classifier.predict_proba(features_scaled)[0]  # Mengambil probabilitas semua kelas

    # Mengambil kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(prediction_probabilities)

    # Menggunakan fungsi predict untuk mendapatkan label dan probabilitas kelas yang diprediksi
    label, probability = predict(predicted_class, prediction_probabilities)
    
    # Membuat dictionary untuk semua label dengan probabilitasnya
    all_predictions = {}
    labels = {
        0: "Jerawat",
        1: "Komedo",
        2: "Hiperpigmentasi",
        3: "Normal",
        4: "Berminyak",
        5: "Kemerahan"
    }
    
    for idx, prob in enumerate(prediction_probabilities):
        all_predictions[labels[idx]] = prob * 100  # Menyimpan probabilitas dalam persen

    return label, probability * 100, all_predictions

def classify(file):
    scaler_path = default_storage.path('model/' + 'scaler2.pkl')
    scaler = joblib.load(scaler_path)
    print(f"Classifying file: {file}")

    model_path = default_storage.path('model/' + 'best_model_mobilenetv3.keras')
    xgb_path = default_storage.path('model/' + 'xgb_model_mobilenetv3small-3.json')
    
    custom_model = load_model(model_path)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    layer_name = 'global_average_pooling2d'
    feature_extractor = Model(inputs=custom_model.input, outputs=custom_model.get_layer(layer_name).output)

    SIZE = 224
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Mengklasifikasikan gambar dan mendapatkan label dan probabilitas
    predicted_label, predicted_probability, all_probabilities = classify_image(img, feature_extractor, scaler, xgb_model)
    
    print(f"Prediksi: {predicted_label}, Probabilitas: {predicted_probability:.2f}%")
    
    # Menampilkan probabilitas semua kelas
    print("Probabilitas untuk setiap kelas:")
    for label, prob in all_probabilities.items():
        print(f"{label}: {prob:.2f}%")

    return predicted_label, predicted_probability, all_probabilities