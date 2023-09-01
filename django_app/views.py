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
    return render(request, 'index.html')

def prediction(request):
    
    file = request.FILES['imageFile']
    file_name = default_storage.save(file.name, file)
    file_url = default_storage.url(file_name)

    return render(request, 'prediction.html', {"file_path": file_url})