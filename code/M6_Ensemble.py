import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn import tree, ensemble
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

models = [ensemble.RandomForestClassifier(max_features = 0.5, oob_score = True, random_state = 2017),
          ensemble.BaggingClassifier(tree.DecisionTreeClassifier(), max_samples = 0.5, max_features = 0.8, oob_score = True, random_state = 2017),
          ensemble.ExtraTreesClassifier(max_features = 0.8, bootstrap = True, oob_score = True, random_state = 2017),
          ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 15), n_estimators = 10, algorithm ='SAMME', random_state = 2017),
          ensemble.GradientBoostingClassifier(n_estimators = 10, random_state = 2017)]

names = ['Random_Forest', 'Bagging', 'ExtraTrees', 'AdaBoost', 'Gradient_Boosting']
for i in range(len(models)):
    model = models[i]
    name = names[i]
    model.fit(x_train, y_train)
    y_pred = np.array(model.predict(x_test))
    
    print(y_test)
    print(y_pred)
    
    # Pickle dictionary using protocol 0.
    pickle.dump(model, open('model_' + name + '.pkl', 'wb'))
    
    print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
    print('Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))
    print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))