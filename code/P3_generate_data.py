from PIL import Image
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')
import P2_resize_images


path = "/Workspace-Github/fruit_classification/processed_images"
dirs = os.listdir(path)
fruits = {'grapefruit':0, 'lemon':1, 'orange':2, 'tangerine':3}

def training_data(path = path, dirs = dirs, fruits = fruits):
    data = []
    labels = []
    for subpath in dirs:
        label = fruits[subpath]
        path1sub = path + '/' + str(subpath) + '/'
        dirs = os.listdir(path1sub)
        for item in dirs:
            if os.path.isfile(path1sub+item):
                im = Image.open(path1sub+item)
                data.append(np.array(im))
                labels.append(label)
    return data, labels


def get_data():
    x_train, y_train = training_data()
    x_test, y_test = P2_resize_images.resize_test()
    y_test =  [fruits[y] for y in y_test]
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=2017)
    print(np.shape(x_train), np.shape(y_test))
    return np.array(x_train), np.array(x_valid), np.array(x_test), np.array(y_train), np.array(y_valid), np.array(y_test)