import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import sys
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')
import P3_generate_data as generate

model = load_model('/Workspace-Github/fruit_classification/code/model_InceptionV3.h5')
x_train, x_valid, x_test, y_train, y_valid, y_test = generate.get_data()
y_pred = np.argmax(model.predict(x_test), axis=1)

print(y_test)
print(y_pred)

print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))