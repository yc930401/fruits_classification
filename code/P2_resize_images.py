from PIL import Image
import os
import numpy as np

path = "/Workspace-Github/fruit_classification/images"
path2 = "/Workspace-Github/fruit_classification/processed_images"
path3 = "/Workspace-Github/fruit_classification/test_images"
dirs1 = os.listdir(path)
dirs3 = os.listdir(path3)

def preprocess(path = path, dirs1 = dirs1, path2 = path2):
    for subpath in dirs1:
        if subpath.endswith('.txt'):
            continue
        i = 1
        path1sub = ''
        path2sub = ''
        path1sub = path + '/' + str(subpath) + '/'
        path2sub = path2 + '/' + str(subpath) + '/'
        if subpath not in os.listdir(path2):
            os.mkdir(path2sub)
        dirs = os.listdir(path1sub)
        for item in dirs:
            if os.path.isfile(path1sub+item):
                im = Image.open(path1sub+item)
                
                imResize = im.resize((128,128), Image.ANTIALIAS)
                imResize.convert('RGB').save(path2sub + str(i) + '_' + 'process_resized.jpg', 'JPEG', quality=90)
                
                imFlip = imResize.transpose(Image.FLIP_LEFT_RIGHT) 
                imFlip.convert('RGB').save(path2sub + str(i) + '_' + 'process_flip.jpg', 'JPEG', quality=90)

                size = [np.random.randint(150, 180), np.random.randint(170, 200), np.random.randint(190, 220), np.random.randint(210, 240)]
                random = [np.random.randint(0, 20), np.random.randint(20, 40), np.random.randint(-40, -20), np.random.randint(-20, 0)]
                box = [(j/2-128/2, j/2-128/2, j/2+128/2, j/2+128/2) for j in size]
                #print(box)
                for j in range(len(size)):
                    imRotate = im.rotate(random[j]).resize((size[j],size[j]), Image.ANTIALIAS).crop(box[j])
                    imRotate2 = imFlip.rotate(random[j]).resize((size[j],size[j]), Image.ANTIALIAS).crop(box[j])
                    imRotate.convert('RGB').save(path2sub + str(i) + '_' + str(j) + 'process_rotate.jpg', 'JPEG', quality=90)
                    imRotate2.convert('RGB').save(path2sub + str(i) + '_' + str(j+len(size)) + 'process_rotate.jpg', 'JPEG', quality=90)
                i += 1
#preprocess()

def resize_test():
    data = []
    labels = []
    for subpath in dirs3:
        path3sub = path3 + '/' + str(subpath) + '/'
        dirs = os.listdir(path3sub)
        for item in dirs:
            if os.path.isfile(path3sub+item):
                im = Image.open(path3sub+item)
                imResize = im.resize((128,128), Image.ANTIALIAS)
                img = imResize.convert('RGB')
                data.append(np.array(img))
                labels.append(subpath.split('+')[1])
                #print(data)
    return data, labels
#resize_test()