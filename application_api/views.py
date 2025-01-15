from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import UploadedImageSerializer
from django.core.files.storage import default_storage
from django.conf import settings
from .classification_model.mobilenetv3small_model import classify
import time
import os

# Create your views here.
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            serializer = UploadedImageSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            image_file = request.FILES.get('image')
            if not image_file:
                return Response({'error': 'No image file provided'}, 
                              status=status.HTTP_400_BAD_REQUEST)

            # Create upload directory if it doesn't exist
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'images')
            os.makedirs(upload_dir, exist_ok=True)

            # Generate unique filename
            unique_filename = f"{int(time.time())}_{image_file.name}"
            file_path = os.path.join('images', unique_filename)

            # Save the file using default_storage
            file_path = default_storage.save(file_path, image_file)
            full_file_path = default_storage.path(file_path)

            # Ensure file exists before processing
            if not os.path.exists(full_file_path):
                return Response({'error': 'Failed to save uploaded file'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Process the image
            try:
                results = classify(full_file_path)
                all_probabilities_xgb, all_probabilities_lgb, all_probabilities_cb, \
                all_probabilities_yolo, all_probabilities_cnn, all_probabilities_cnn_ex_feat = results
            except Exception as e:
                return Response({'error': f'Classification failed: {str(e)}'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Format probabilities
            formatted_probabilities_xgb = {label: f"{prob:.2f}%" 
                                         for label, prob in all_probabilities_xgb.items()}
            formatted_probabilities_lgb = {label: f"{prob:.2f}%" 
                                         for label, prob in all_probabilities_lgb.items()}
            formatted_probabilities_cb = {label: f"{prob:.2f}%" 
                                        for label, prob in all_probabilities_cb.items()}

            # Prepare response
            response_data = {
                "class_probabilities_xgb": formatted_probabilities_xgb,
                "class_probabilities_lgb": formatted_probabilities_lgb,
                "class_probabilities_cb": formatted_probabilities_cb,
                "class_probabilities_yolo": all_probabilities_yolo,
                "class_probabilities_cnn": all_probabilities_cnn,
                "class_probabilities_cnn_ex_feat": all_probabilities_cnn_ex_feat,
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': f'Unexpected error: {str(e)}'}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Clean up the temporary file if it exists
            try:
                if 'full_file_path' in locals() and os.path.exists(full_file_path):
                    default_storage.delete(file_path)
            except Exception:
                pass