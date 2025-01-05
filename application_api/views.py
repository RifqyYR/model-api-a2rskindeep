from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import UploadedImageSerializer
from django.core.files.storage import default_storage
from .classification_model.mobilenetv3small_model import classify
import time
import os

# Create your views here.
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()

            image_file = request.FILES['image']
            unique_filename = str(int(time.time())) + os.path.splitext(image_file.name)[1]

            file_path = default_storage.save('images/' + unique_filename, image_file)
            full_file_path = default_storage.path(file_path)
            print(full_file_path)
            
            # Mengklasifikasikan gambar menggunakan fungsi classify
            all_probabilities_xgb, all_probabilities_lgb, all_probabilities_cb, all_probabilities_yolo, all_probabilities_cnn, all_probabilities_cnn_ex_feat = classify(full_file_path)

            # Memformat probabilitas setiap kelas dengan dua angka di belakang koma
            formatted_probabilities_xgb = {label: f"{prob:.2f}%" for label, prob in all_probabilities_xgb.items()}

            formatted_probabilities_lgb = {label: f"{prob:.2f}%" for label, prob in all_probabilities_lgb.items()}

            formatted_probabilities_cb = {label: f"{prob:.2f}%" for label, prob in all_probabilities_cb.items()}

            # Menyusun hasil dalam format JSON
            response_data = {
                "class_probabilities_xgb": formatted_probabilities_xgb,
                "class_probabilities_lgb": formatted_probabilities_lgb,
                "class_probabilities_cb": formatted_probabilities_cb,
                "class_probabilities_yolo": all_probabilities_yolo,
                "class_probabilities_cnn": all_probabilities_cnn,
                "class_probabilities_cnn_ex_feat": all_probabilities_cnn_ex_feat,
            }

            # Mengembalikan response dalam format JSON
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)