from django.shortcuts import render
from fruit_web.forms import ImageForm
import numpy as np
from PIL import Image
from django.core.files.storage import FileSystemStorage
from io import BytesIO
from django.conf import settings
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = settings.MODEL


def fruit(request):
    if request.method == 'POST' and request.FILES:
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            print(request.FILES)
            im = request.FILES['im']
            result = handle_uploaded_image(im)

            if os.path.exists('test.jpg'):
                os.remove('test.jpg')

            fs = FileSystemStorage()
            fs.save('test.jpg', im)
            uploaded_file_url = fs.url('test.jpg')
            print(uploaded_file_url)

            context = {
                'fruit_list': result,
                'im': uploaded_file_url
            }
            print(context)
            return render(request, 'fruit.html', context)
    else:
        form = ImageForm()
    return render(request, 'fruit.html', {'form': form})


def predict(im):
    fruits = {0: 'grapefruit', 1: 'lemon', 2: 'orange', 3: 'tangerine'}
    img = np.array(im.convert('RGB'))

    # predict useing CNN
    img = img.reshape(-1, 16384, 3)
    img = img.astype('float32')
    img /= 255
    img = img.reshape(img.shape[0], 128, 128, 3)
    #model = load_model(BASE_DIR + '/fruit_web/model_CNN.h5')
    y_prob = model.predict(img)
    result = [{'name': fruits[i], 'probability': y_prob[0][i]} for i in range(4)]
    return result


def handle_uploaded_image(im):
    print('handle_uploaded_image')
    imagefile = BytesIO(im.read())
    im = Image.open(imagefile)
    # resize or crop images
    size = np.shape(im)[:2]
    box = (size[0] / 2 - 128 / 2, size[1] / 2 - 128 / 2, size[0] / 2 + 128 / 2, size[1] / 2 + 128 / 2)

    if size[0] >= 128 and size[1] >= 128:
        im = im.crop(box)
    else:
        im = im.resize((128, 128), Image.ANTIALIAS)
    return predict(im)
