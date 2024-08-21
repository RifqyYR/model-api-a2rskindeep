from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import UploadedImageSerializer
from django.core.files.storage import default_storage
from .classification_model.mobilenetv2_model import predict, classify

# Create your views here.
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()

            image_file = request.FILES['image']
            file_path = default_storage.path('images/' + image_file.name)
            print(file_path)
            predicted_classes = classify(file_path)
            result = predict(predicted_classes)
            return Response({"result": result}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)