import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')
import P3_generate_data as generate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

n_component = 6

x_train, x_valid, x_test, y_train, y_valid, y_test = generate.get_data()
x_train = [x.reshape(1, -1)[0] for x in x_train]
x_valid = [x.reshape(1, -1)[0] for x in x_valid]
x_test = [x.reshape(1, -1)[0] for x in x_test]
pca = PCA(svd_solver='randomized', n_components=n_component)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

NB = pickle.load(open('model_NB.pkl', 'rb'))
KNN = KNeighborsClassifier(n_neighbors = 3, weights = 'distance').fit(x_train, y_train)
Dtree = pickle.load(open('model_Dtree.pkl', 'rb'))
model = VotingClassifier(estimators=[('NB', NB), ('KNN', KNN), ('Dtree', Dtree)], 
                         voting='hard', weights = [0.32, 0.31, 0.37])
model.fit(x_train, y_train)
y_pred = np.array(model.predict(x_test))

print(y_test)
print(y_pred)

# Pickle dictionary using protocol 0.
pickle.dump(model, open('model_Vote.pkl', 'wb'))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))