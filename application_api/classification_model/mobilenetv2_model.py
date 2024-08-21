import xgboost as xgb
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model
import os

def predict(classes):
    if (classes == 0): return "Berminyak"
    elif (classes == 1): return "Hiperpigmentasi"
    elif (classes == 2): return "Jerawat"
    elif (classes == 3): return "Kemerahan"
    elif (classes == 4): return "Komedo"
    elif (classes == 5): return "Normal"

def classify(file):
    print(f"Classifying file: {file}")
    
    # Load models from the same directory as the script
    custom_model = load_model("mobile-net-v2.keras")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgb_model_mobilenetv2.json")

    SIZE = 128

    feature_extractor = Model(inputs=custom_model.input, outputs=custom_model.layers[-4].output)

    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(img, axis=0)
    X_test = feature_extractor.predict(image)
    X_test_features = X_test.reshape(X_test.shape[0], -1)

    predicted_probabilities = xgb_model.predict_proba(X_test_features)
    print(predicted_probabilities)
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    return predicted_classes