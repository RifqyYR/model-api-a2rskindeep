import xgboost as xgb
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler
from django.core.files.storage import default_storage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def extract_features(model, images):
    images_preprocessed = preprocess_input(images)
    features = model.predict(images_preprocessed)
    return features

def predict(classes):
    if (classes == 0): return "Berminyak"
    elif (classes == 1): return "Hiperpigmentasi"
    elif (classes == 2): return "Jerawat"
    elif (classes == 3): return "Kemerahan"
    elif (classes == 4): return "Komedo"
    elif (classes == 5): return "Normal"

# Fungsi untuk klasifikasi gambar baru
def classify_image(image, feature_extractor, scaler, classifier):
    features = extract_features(feature_extractor, np.expand_dims(image, axis=0))
    features_scaled = scaler.transform(features)
    prediction = classifier.predict(features_scaled)
    return prediction[0]

def classify(file):
    scaler_path = default_storage.path('model/' + 'scaler.pkl')
    svm_path = default_storage.path('model/' + 'svm_model_mobilenetv2.pkl')
    svm = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    print(f"Classifying file: {file}")

    model_path = default_storage.path('model/' + 'mobile_net_v2_coba.keras')
    xgb_path = default_storage.path('model/' + 'xgb_model_mobilenetv2_coba.json')
    
    # # Load models from the same directory as the script
    custom_model = load_model(model_path)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    layer_name='average_pooling'
    # layer_name='my_dense_layer'
    feature_extractor = Model(inputs=custom_model.input,outputs=custom_model.get_layer(layer_name).output)

    SIZE = 224
    # SIZE = 128

    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prediction_xgb = classify_image(img, feature_extractor, scaler, xgb_model)
    prediction_svm = classify_image(img, feature_extractor, scaler, svm)
    print(f"Prediksi XGB: {prediction_xgb}")
    print(f"Prediksi SVM: {prediction_svm}")

    # feature_extractor = Model(inputs=custom_model.input, outputs=custom_model.layers[-4].output)

    # img = cv2.imread(file, cv2.IMREAD_COLOR)
    # img = cv2.resize(img, (SIZE, SIZE))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = np.expand_dims(img, axis=0)
    # X_test = feature_extractor.predict(image)
    # X_test_features = X_test.reshape(X_test.shape[0], -1)

    # predicted_probabilities = xgb_model.predict_proba(X_test_features)
    # print(predicted_probabilities)
    # predicted_classes = np.argmax(predicted_probabilities, axis=1)

    # return prediction_xgb
    return prediction_svm