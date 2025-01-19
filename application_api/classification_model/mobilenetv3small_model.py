import xgboost as xgb
import cv2
import numpy as np
import joblib
from torchvision import transforms
import torch
from mtcnn import MTCNN
from torch import nn
from ultralytics import YOLO
from scipy.ndimage import gaussian_laplace
from tensorflow.keras.models import load_model, Model
from django.core.files.storage import default_storage
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Initialize MTCNN
detector = MTCNN()

def preprocess_image(file, size=224):
    # Read the image using OpenCV
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MTCNN

    # Detect faces using MTCNN
    detections = detector.detect_faces(img_rgb)
    
    if detections:
        # If faces are detected, use the first detected face
        x, y, width, height = detections[0]['box']  # Get the bounding box
        x, y = max(0, x), max(0, y)  # Ensure the coordinates are non-negative
        
        # Crop the face
        face = img_rgb[y:y+height, x:x+width]
        # Resize the cropped face to the target size
        face = cv2.resize(face, (size, size))
        img_processed = face
    else:
        # If no faces are detected, resize the entire image
        img_processed = cv2.resize(img_rgb, (size, size))
    
    return img_processed

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
    # features_scaled = scaler.transform(features)
    prediction_probabilities = classifier.predict_proba(features)[0]  # Mengambil probabilitas semua kelas

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

def classify_yolo(image, model):
    # Prediksi pada gambar
    results = model.predict(source=image)

    # Mengambil nama kelas dari model
    class_names = ['Berminyak', 'Hiperpigmentasi', 'Jerawat', 'Kemerahan', 'Komedo', 'Normal']

    # Membuat dictionary untuk menampung nama kelas dan probabilitasnya dalam bentuk persentase
    probability_dict = {}

    # Mengakses probabilitas dari hasil prediksi
    for result in results:
        probs = result.probs.data  # Mengakses tensor probabilitas

        # Mengonversi probabilitas ke dalam dictionary dengan format persentase
        for idx, prob in enumerate(probs):
            class_name = class_names[idx]
            probability_dict[class_name] = f"{prob.item() * 100:.2f}%"  # Mengonversi tensor ke nilai float dan bentuk persentase

    return probability_dict

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer Convolutional pertama
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer Convolutional kedua
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer Convolutional ketiga
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling dan Dropout
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        
        # Global Average Pooling untuk mengurangi ukuran tensor sebelum fully connected layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 6)
        
        # Activation Function
        self.relu = nn.ReLU()   

    def forward(self, x):
        # Forward pass untuk layer-layer convolutional dan batch normalization
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
class CNN2ExtractionFeatures(nn.Module):
    def __init__(self):
        super(CNN2ExtractionFeatures, self).__init__()
        # Layer Convolutional dan Fully Connected seperti sebelumnya
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128 + 10, 128)  # Tambahkan 9 dimensi untuk Color Moments
        self.fc2 = nn.Linear(128, 6)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Forward pass untuk layer-layer CNN
        x_cnn = self.relu(self.bn1(self.conv1(x)))
        x_cnn = self.maxpool(x_cnn)
        x_cnn = self.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = self.maxpool(x_cnn)
        x_cnn = self.relu(self.bn3(self.conv3(x_cnn)))
        x_cnn = self.maxpool(x_cnn)
        x_cnn = self.gap(x_cnn)
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        
        # Ekstraksi fitur manual (Color Moments)
        additional_features = self.extract_features(x)
        
        # Gabungkan fitur CNN dan fitur manual
        combined_features = torch.cat((x_cnn, additional_features), dim=1)
        
        # Fully connected layer
        x = self.dropout1(combined_features)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def extract_features(self, x):
        # Convert batch of images to numpy array
        images_np = x.cpu().numpy().transpose(0, 2, 3, 1)
        batch_size = images_np.shape[0]
        all_features = []
        
        for i in range(batch_size):
            image = images_np[i]
            
            # Ekstraksi Color Moments (R, G, B)
            color_moments = self.calculate_color_moments(image)
            
            # Ekstraksi LoG (Laplacian of Gaussian) seperti sebelumnya
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = (gray_image * 255).astype(np.uint8)
            log_feature = np.mean(gaussian_laplace(gray_image, sigma=1.5))
            
            # Gabungkan semua fitur (Color Moments + LoG)
            combined_features = np.concatenate([color_moments, [log_feature]])
            all_features.append(combined_features)
        
        return torch.tensor(all_features, dtype=torch.float32).to(x.device)

    def calculate_color_moments(self, image):
        """
        Menghitung Color Moments (Mean, Std, Skewness) untuk setiap channel (R, G, B)
        """
        # Channel R, G, B
        channels = cv2.split(image)
        
        moments = []
        for channel in channels:
            # Hitung Mean dan Std Dev
            mean = np.mean(channel)
            std = np.std(channel)
            
            # Hitung Skewness secara manual
            skewness_numerator = np.mean((channel - mean) ** 3)
            skewness_value = np.sign(skewness_numerator) * (abs(skewness_numerator) ** (1./3.))  # Mengelola tanda

            # Simpan momen ke list
            moments.extend([mean, std, skewness_value])
        
        return np.array(moments)
    
def predict_cnn(image, model):
    # Define preprocessing steps (adjust based on your model's requirements)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ])

    # Preprocess the image
    image_tensor = transform(image)

    # Add a batch dimension (unsqueeze) if the model expects a batch of images
    image_tensor = image_tensor.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)  # Output logit values

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get the predicted class with the highest probability
        _, predicted = torch.max(probabilities, 1)

        # Menampilkan probabilitas untuk setiap kelas
        class_names = ["Jerawat", "Komedo", "Hiperpigmentasi", "Normal", "Berminyak", "Kemerahan"]  # Sesuaikan dengan nama kelas yang digunakan
        probs = probabilities.squeeze().tolist()  # Menghapus dimensi batch dan mengonversi ke list
        probability_dict = {class_names[idx]: f"{prob * 100:.2f}%" for idx, prob in enumerate(probs)}

        # Menampilkan hasil prediksi
        print(f"Predicted class: {class_names[predicted.item()]}")
        print("Probabilities for each class:")
        for class_name, prob in probability_dict.items():
            print(f"{class_name}: {prob}")

    return probability_dict

def classify(file):
    scaler_path = default_storage.path('model/' + 'scaler2.pkl')
    scaler = joblib.load(scaler_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Classifying file: {file}")

    model_path = default_storage.path('model/' + 'testing.keras')    
    xgb_path = default_storage.path('model/' + 'xgb_model_mobilenetv3small_test.json')
    lgb_path = default_storage.path('model/' + "lgb_mobilenetv3small-3_test.pkl")
    cb_path = default_storage.path('model/' + "cb_mobilenetv3small-3_test.pkl")
    yolo_path = default_storage.path('model/' + "yolo.pt")
    cnn_path = default_storage.path('model/' + "best_model_cnn_imp.pth")
    cnn_ex_feat_path = default_storage.path('model/' + "best_model_cnn_imp_log_cm_feat.pth")
    
    custom_model = load_model(model_path)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    lgb_model = joblib.load(lgb_path)
    cb_model = joblib.load(cb_path)
    yolo_model = YOLO(yolo_path)

    cnn_model = CNN().to(device)
    cnn_model.load_state_dict(torch.load(cnn_path))

    cnn_ex_feat_model = CNN2ExtractionFeatures().to(device)
    cnn_ex_feat_model.load_state_dict(torch.load(cnn_ex_feat_path))

    layer_name = 'global_average_pooling2d'
    feature_extractor = Model(inputs=custom_model.input, outputs=custom_model.get_layer(layer_name).output)

    SIZE = 224
    img = preprocess_image(file, SIZE)
    
    # Mengklasifikasikan gambar dan mendapatkan label dan probabilitas
    predicted_label, predicted_probability, all_probabilities = classify_image(img, feature_extractor, scaler, xgb_model)
    print(f"Prediksi XGB: {predicted_label}, Probabilitas XGB: {predicted_probability:.2f}%")
    # Menampilkan probabilitas semua kelas
    print("Probabilitas untuk setiap kelas XGB:")
    for label, prob in all_probabilities.items():
        print(f"{label}: {prob:.2f}%")


    pred_lgb_label, pred_lgb_prob, all_lgb_prob = classify_image(img, feature_extractor, scaler, lgb_model)
    print(f"Prediksi LGB: {pred_lgb_label}, Probabilitas LGB: {pred_lgb_prob:.2f}%")
    # Menampilkan probabilitas semua kelas
    print("Probabilitas untuk setiap kelas LGB:")
    for label, prob in all_lgb_prob.items():
        print(f"{label}: {prob:.2f}%")


    pred_cb_label, pred_cb_prob, all_cb_prob = classify_image(img, feature_extractor, scaler, cb_model)
    print(f"Prediksi CB: {pred_cb_label}, Probabilitas CB: {pred_cb_prob:.2f}%")
    # Menampilkan probabilitas semua kelas
    print("Probabilitas untuk setiap kelas CB:")
    for label, prob in all_cb_prob.items():
        print(f"{label}: {prob:.2f}%")


    all_yolo_prob = classify_yolo(img, yolo_model)
    print(all_yolo_prob)


    all_cnn_prob = predict_cnn(img, cnn_model)
    print(all_cnn_prob)

    all_cnn_ex_feat_prob = predict_cnn(img, cnn_ex_feat_model)

    return all_probabilities, all_lgb_prob, all_cb_prob, all_yolo_prob, all_cnn_prob, all_cnn_ex_feat_prob