from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session
from django.core.files.storage import default_storage
from django.views.generic import TemplateView
from django.http import HttpResponse
from keras.applications import vgg16
from django.shortcuts import render
from django.conf import settings
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
        # https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img
        #
        image = load_img(file_url, target_size=(224,224))
        numpy_array = img_to_array(image)
        image_batch = np.expand_dims(numpy_array, axis=0)
        process_image = vgg16.preprocess_input(image_batch.copy())

        #
        # get the predicted probabilities
        #
        with settings.GRAPH1.as_default():
            set_session(settings.SESS)
            predicitions = settings.IMAGE_MODEL.predict(process_image)

        #
        # Output/Return Data
        #
        
        label = decode_predictions(predicitions, top=10)
        return render(request, "index.html", {"predictions": label})
    else:
        return render(request, 'index.html')
    
    return render(request, "index.html")
