import numpy as np
import sys
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')
import P3_generate_data as generate
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

## PCA to reduce dimension
n_component = 6
x_train, x_valid, x_test, y_train, y_valid, y_test = generate.get_data()
x_train = [x.reshape(1, -1)[0] for x in x_train]
x_test = [x.reshape(1, -1)[0] for x in x_test]
pca = PCA(svd_solver='randomized', n_components=n_component)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(pca.explained_variance_ratio_)

print('train', np.shape(x_train))
print('test', np.shape(x_test))
## KNN model

parameters = [
    {'n_neighbors': [i for i in range(3,50)], 'weights': ['distance', 'uniform']}]

model = model_selection.GridSearchCV(KNeighborsClassifier(), parameters, cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2017))
model.fit(x_train, y_train)
y_pred = np.array(model.predict(x_test))
print('best parameters: ', model.best_params_)

print(y_test)
print(y_pred)

print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))