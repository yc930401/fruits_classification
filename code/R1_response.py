import os
import sys
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')

path = '/Workspace-Github/fruit_classification/code/'
fruits = {0: 'grapefruit', 1: 'lemon', 2: 'orange', 3: 'tangerine'}

def respond(im = Image.open('/Workspace-Github/fruit_classification/test.jpg')):
    
    print('Responding -------------- ')
    size = np.shape(im)[:2]
    box = (size[0]/2-128/2, size[1]/2-128/2, size[0]/2+128/2, size[1]/2+128/2)
    # resize or crop images
    if size[0] >= 128 and size[1] >= 128:
        im = im.crop(box)
    else:
        im = im.resize((128,128), Image.ANTIALIAS)
    img = np.array(im.convert('RGB'))    
    
    # predict useing CNN
    result = {} 
    print('Predicting **************')         
    img = img.reshape(-1, 16384, 3)
    img = img.astype('float32')
    img /= 255
    img = img.reshape(img.shape[0], 128, 128, 3)
    CNN=load_model(path + 'model_CNN.h5')
    y_prob = CNN.predict(img)
    y_pred = np.argmax(y_prob)
    result['CNN'] = fruits[y_pred]
    
    response = '\n'.join([key + ': ' + str(value) for key, value in result.items()])
    return response

#print(respond())