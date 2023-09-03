from django.core.files.storage import default_storage
from keras.preprocessing import image
from django.shortcuts import render
from keras.models import load_model
import tensorflow as tf
import numpy as np
import json

with open('./model/imagenet_classes.json', 'r') as f:
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
    
    print(predi)
    predictedLabel = labelInfo[str(np.argmax(predi[0]))]
    print(predictedLabel)

    return render(request, 'prediction.html', {"file_path": file_url, "predictedLabel": predictedLabel[1]})