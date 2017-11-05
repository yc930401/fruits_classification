import numpy as np
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

import sys
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')
import P3_generate_data as generate

batch_size = 20
num_classes = 4
epochs = 10

def prepare_data():
    x_train, x_valid, x_test, y_train, y_valid, y_test = generate.get_data()
    x_train = x_train.reshape(-1, 16384,3)
    x_valid = x_valid.reshape(-1, 16384,3)
    x_test = x_test.reshape(-1, 16384, 3)
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_valid /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'valid samples')
    print(x_test.shape[0], 'test samples')
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    
    img_rows, img_cols = 128, 128
    x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 3) 
    x_valid = x_valid.reshape(x_valid.shape[0], img_cols, img_rows, 3)
    x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 3)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def train_generator(x_train, y_train, batch_size):
    while True:
        i = 0
        while i < len(x_train):
            x_batch = x_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]
            i += batch_size
        yield x_batch, y_batch
        
def valid_generator(x_valid, y_valid, batch_size):
    while True:
        i = 0
        while i < len(x_valid):
            x_batch = x_valid[i: i + batch_size]
            y_batch = y_valid[i: i + batch_size]
            i += batch_size
        yield x_batch, y_batch
        

def test_generator(x_test, batch_size):
    while True:
        x_batch = []
        for i in range(len(x_test), batch_size):
            x_batch.append(x_test[i: i + batch_size])
        yield x_batch

                    

    
x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_data()
train_steps = int(len(x_train) / batch_size)
valid_steps = int(len(x_valid) / batch_size)
test_steps = int(len(x_test) / batch_size)
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer = keras.optimizers.Adam(lr=1e-4, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
model.fit_generator(train_generator(x_train, y_train, batch_size), train_steps, epochs=epochs, verbose=1, 
                        validation_data=valid_generator(x_valid, y_valid, batch_size), 
                        validation_steps=valid_steps)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer = keras.optimizers.Adam(lr=1e-4, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator(x_train, y_train, batch_size), train_steps, epochs=epochs, verbose=1,
                        validation_data=valid_generator(x_valid, y_valid, batch_size), 
                        validation_steps=valid_steps)
    
y_pred = np.argmax(model.predict(x_test), axis=1)

print(y_test)
print(y_pred)

model.save('/Workspace-Github/fruit_classification/code/model_InceptionV3.h5')

print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))
