from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import UploadedImageSerializer
from django.core.files.storage import default_storage
# from .classification_model.mobilenetv2_model import predict, classify
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
            label, prob, all_probabilities = classify(full_file_path)

            # Memformat probabilitas setiap kelas dengan dua angka di belakang koma
            formatted_probabilities = {label: f"{prob:.2f}%" for label, prob in all_probabilities.items()}

            # Menyusun hasil dalam format JSON
            response_data = {
                "result": label,
                "probability": f"{prob:.2f}%",  # Memformat probabilitas hasil prediksi utama
                "class_probabilities": formatted_probabilities  # Probabilitas setiap kelas dengan format .2f
            }

            # Mengembalikan response dalam format JSON
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)