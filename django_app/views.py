from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session
from django.core.files.storage import default_storage
from django.views.generic import TemplateView
from keras.preprocessing import image
from keras.applications import vgg16
from django.shortcuts import render
from keras.models import load_model
from django.conf import settings
import tensorflow as tf
import numpy as np
import json

img_height, img_width = 244,244
with open('./model/imagenet_classes.json', 'r', encoding="utf8") as f:
    labelInfo = f.read()

labelInfo = json.loads(labelInfo)


model_graph = tf.Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model = load_model('./model/MobileNetModelImagenet.h5') 



def index(request):
    return render(request, 'index.html')

def prediction(request):
    
    file = request.FILES['imageFile']
    file_name = default_storage.save(file.name, file)
    file_url = default_storage.url(file_name)

    testimage = '.'+file_url

    img_height, img_width = 224, 224
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x / 255.0
    x = x.reshape(1, img_height, img_width, 3)

    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)

    predictedLabel = labelInfo[str(np.argmax(predi[0]))]

    return render(request, 'prediction.html', {"file_path": file_url, "prdictedLabel": predictedLabel})