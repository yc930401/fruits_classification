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

Accuracy, Precisipn, Recall, F1 scores and confusion matrix: </br>
![fruits](/scores.png)
</br>
Compare different classifiers: </br>
![fruits](/plot.png)
</br>
Telegram interface for user to upload photos: </br>
![fruits](/telegram.png)

InceptionV3: </br>
This overfits test examples badly. I add dropoutlayer to the model but it doesn't work. The base layers do not seem to have any dropout layers. I'll try to figure out the reason later. </br>

## Analysis
Deep learning model is the best model, even outperform ensemble models. The reason is that deep learning models learn image features themselves, while in basic machine learning models, we need to choose features for the models (e.g. pca).

## Reference
https://keras.io/applications/ </br>
http://blog.yhat.com/posts/image-classification-in-Python.html </br>
https://datascience.stackexchange.com/questions/8847/feature-extraction-of-images-in-python </br>
https://benanne.github.io/2015/03/17/plankton.html </br>
http://image-net.org/synset?wnid=n07749582#
