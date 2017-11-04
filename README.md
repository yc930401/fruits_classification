# fruit_classification
For JJ's ML project

## Introduction
Some fruits look very similar and very difficult to distinguish, e.g. lemon, orange, tangerine and grapefruit. In this project, I build several fruit classifiers using CNN, KNN, Inception V3 and SVM. </br>
![fruits](/fruits.png)

## Methodology
1. Download images from imagenet and google images. 
2. Preprocess images. (e.g.resize, rotate, flip) It is very neccessary to preprocess thoese images because, it can reduce overfitting problem and increase the size of the training dataset. <br>
After preprocessing, I have more than 19900 training samples and more than 400 each for evaluation and test.
![fruits](/preprocess.png)
3. Train models and test.

## Result
CNN: </br>
('Confusion matrix: ',  </br>
array([[34,  1,  1,  2], </br>
       [ 1, 29, 14,  0], </br>
       [ 0,  0, 44,  3], </br>
       [ 3,  0, 26, 16]])) </br>
('Accuracy score: ', 0.7068965517241379) </br>
('Precision score: ', 0.77671528381467536) </br>
('Recall score: ', 0.7068965517241379) </br>
('F1 score: ', 0.69906897493104392) </br>

InceptionV3: </br>
This overfits test examples badly. I add dropoutlayer to the model but it doesn't work. The base layers do not seem to have any dropout layers. I'll try to figure out the reason later. </br>

KNN(neighbors = 6): </br>
('Confusion matrix: ', </br>
array([[17,  4,  9,  8], </br>
       [ 1, 35,  8,  0], </br>
       [ 1, 18, 21,  7], </br>
       [11,  8, 15, 11]])) </br>
('Accuracy score: ', 0.48275862068965519) </br>
('Precision score: ', 0.47636059812377313) </br>
('Recall score: ', 0.48275862068965519) </br>
('F1 score: ', 0.46517553279181567) </br>

SVM: </br>
I tried very small datasset, gridsearch shows that kernel = 'poly', C = 1, degree = 3 is the best model. But I cannot run it on the whole dataset. Very slow. Will try later.

## Reference
https://keras.io/applications/ </br>
http://blog.yhat.com/posts/image-classification-in-Python.html </br>
https://datascience.stackexchange.com/questions/8847/feature-extraction-of-images-in-python </br>
https://benanne.github.io/2015/03/17/plankton.html </br>
http://image-net.org/synset?wnid=n07749582#
