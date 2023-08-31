from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session
from django.core.files.storage import default_storage
from django.views.generic import TemplateView
from keras.applications import vgg16
from django.shortcuts import render
from django.conf import settings
import tensorflow as tf
import numpy as np

def index(request):
    if request.method == 'POST':
        """
        Django Image API
        """
        file = request.FILES['imageFile']
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)

        #
        # Load and preprocess the image
        #
        image = load_img(file_url, target_size=(224, 224))
        numpy_array = img_to_array(image)
        image_batch = np.expand_dims(numpy_array, axis=0)

        processed_image = vgg16.preprocess_input(image_batch.copy())

        # #
        # # Get the predicted probabilities
        # #
        predictions = settings.IMAGE_MODEL.predict(processed_image)

        top_predictions = decode_predictions(predictions, top=5)

        # Define the path to the image with bounding boxes
        bounded_image_path = 'path_to_bounded_image.jpg'  # Update with the actual path
        
        return render(request, "index.html", {"predictions": top_predictions[0], "image_path": bounded_image_path})
    else:
        return render(request, 'index.html')
            