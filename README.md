# fruit_classification
For JJ's ML project

## Introduction
Some fruits look very similar and very difficult to distinguish, e.g. lemon, orange, tangerine and grapefruit. In this project, I build several fruit classifiers using CNN, KNN, Decision Tree, Naive Bayes, etc.. </br>
![fruits](/fruits.png)

## Methodology
1. Download images from imagenet and google images. 
2. Preprocess images. (e.g.resize, rotate, flip) It is very neccessary to preprocess thoese images because, it can reduce overfitting problem and increase the size of the training dataset. <br>
After preprocessing, I have more than 19900 training samples and more than 400 each for evaluation and test.
![fruits](/preprocess.png)
3. Train models and test.

## Result
CNN: </br>
Confusion matrix:  </br>
[[34,  1,  1,  2], </br>
 [ 1, 29, 14,  0], </br>
 [ 0,  0, 44,  3], </br>
 [ 3,  0, 26, 16]] </br>
Accuracy score: 0.7068965517241379 </br>
Precision score: 0.77671528381467536 </br>
Recall score: 0.7068965517241379 </br>
F1 score: 0.69906897493104392 </br>

InceptionV3: </br>
This overfits test examples badly. I add dropoutlayer to the model but it doesn't work. The base layers do not seem to have any dropout layers. I'll try to figure out the reason later. </br>

KNN(neighbors = 3): </br>
Confusion matrix: </br>
[[18,  4, 10,  6], </br>
[ 2, 36,  6,  0], </br>
[ 4, 20, 14,  9], </br>
[ 8,  8, 15, 14]] </br>
Accuracy score: 0.47126436781609193) </br>
Precision score: 0.46560619425468691) </br>
Recall score: 0.47126436781609193) </br>
F1 score: 0.45494222000968615) </br>

Naive Bayes: </br>
Confusion matrix:  </br>
[[32,  5,  1,  0], </br>
[ 0, 43,  1,  0], </br>
[ 1, 37,  9,  0], </br>
[ 9, 31,  5,  0]] </br>
Accuracy score:  0.48275862068965519 </br>
Precision score:  0.4120702631032595 </br>
Recall score:  0.48275862068965519 </br>
F1 score:  0.38780788177339903 </br>

Decision Tree(Depth=19): </br>
Confusion matrix:  </br>
[[25,  2,  1, 10], </br>
[ 2, 38,  4,  0], </br>
[ 3, 18, 18,  8], </br>
[12,  6, 13, 14]] </br>
Accuracy score:  0.54597701149425293 </br>
Precision score:  0.52834222769567596 </br>
Recall score:  0.54597701149425293 </br>
F1 score:  0.52564449135214963 </br>

VoteClassifier(NB, Dtree, KNN): </br>
Confusion matrix:   </br>
[[29,  2,  1,  6], </br>
[ 1, 41,  2,  0], </br>
[ 1, 26, 16,  4], </br>
[10, 11, 14, 10]] </br>
Accuracy score:  0.55172413793103448 </br>
Precision score:  0.5443444113124517 </br>
Recall score:  0.55172413793103448 </br>
F1 score:  0.51518196676389316 </br>

Random Forest: </br>
Confusion matrix:   </br>
[[23,  2,  5,  8], </br>
[ 0, 43,  1,  0], </br>
[ 4, 23, 19,  1], </br>
[11, 13, 17,  4]] </br>
Accuracy score:  0.5114942528735632 </br>
Precision score:  0.4681958810311300 </br>
Recall score:  0.5114942528735632 </br>
F1 score:  0.45716271426471977 </br>

Bagging: </br>
Confusion matrix:   </br>
[[10, 15,  7,  6], </br>
[ 5, 30,  6,  3], </br>
[12, 18, 14,  3], </br>
[16, 19,  6,  4]] </br>
Accuracy score:  0.33333333333333331 </br>
Precision score:  0.322552667915685540 </br>
Recall score:  0.33333333333333331 </br>
F1 score:  0.30279733532198827 </br>

ExtraTreesg: </br>
Confusion matrix:   </br>
[[21,  3,  6,  8], </br>
[ 2, 39,  3,  0], </br>
[ 3, 20, 20,  4], </br>
[10, 13, 19,  3]] </br>
Accuracy score:  0.47701149425287354 </br>
Precision score:  0.42316091954022989 </br>
Recall score:  0.47701149425287354 </br>
F1 score:  0.42929527233466597 </br>

AdaBoost: </br>
Confusion matrix:   </br>
[[18,  1, 18,  1], </br>
[ 1, 41,  2,  0], </br>
[ 7, 26,  9,  5], </br>
[11, 13, 19,  2]] </br>
Accuracy score:  0.40229885057471265 </br>
Precision score:  0.34954362887792007 </br>
Recall score:  0.40229885057471265 </br>
F1 score:  0.34141085961487977 </br>

AdaBoost: </br>
Confusion matrix:   </br>
[[27,  8,  2,  1], </br>
[ 1, 42,  1,  0], </br>
[ 3, 34, 10,  0], </br>
[ 9, 27,  5,  4]] </br>
Accuracy score:  0.47701149425287354 </br>
Precision score:  0.60005609057333198 </br>
Recall score:  0.47701149425287354 </br>
F1 score:  0.41272639114685838 </br>

## Analysis
Deep learning model is the best model, even outperform ensemble models. The reason is that deep learning models learn image features themselves, while in basic machine learning models, we need to choose features for the models (e.g. pca).

## Reference
https://keras.io/applications/ </br>
http://blog.yhat.com/posts/image-classification-in-Python.html </br>
https://datascience.stackexchange.com/questions/8847/feature-extraction-of-images-in-python </br>
https://benanne.github.io/2015/03/17/plankton.html </br>
http://image-net.org/synset?wnid=n07749582#
