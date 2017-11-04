import numpy as np
import pickle
import sys
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')
import P3_generate_data as generate
from sklearn import naive_bayes
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

## PCA to reduce dimension
n_component = 6
x_train, x_valid, x_test, y_train, y_valid, y_test = generate.get_data()
x_train = [x.reshape(1, -1)[0] for x in x_train]
x_valid = [x.reshape(1, -1)[0] for x in x_valid]
x_test = [x.reshape(1, -1)[0] for x in x_test]
pca = PCA(svd_solver='randomized', n_components=n_component)
x_valid = pca.fit_transform(x_valid)
#x_train = pca.fit_transform(np.append(x_train, x_valid, axis=0))
#y_train = np.append(y_train, y_valid, axis=0)
x_test = pca.transform(x_test)
print(pca.explained_variance_ratio_)


model = naive_bayes.GaussianNB()
#clf = model_selection.GridSearchCV(svm.SVC(), parameters, cv = model_selection.StratifiedKFold(n_splits = 3, shuffle = True, random_state = 2017))
model.fit(x_valid, y_valid)
y_pred = np.array(model.predict(x_test))

print(y_test)
print(y_pred)
# Pickle dictionary using protocol 0.
pickle.dump(model, open('model_NB.pkl', 'wb'))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))