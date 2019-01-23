#Import GaussianNB classifier function and fill it with two array inputs(data, label)
#Now we can predict which label a new element will produce from classifier
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
pred = clf.predict([[-0.8, -1], [5, 3]])

#How to calculate accuracy?
#After making predictions on test data, we can compare the prediction result
#with real results using accuracy_score(real_data, prediction_data)
labels_test = [1, 1]
from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, pred))

#Testing data is really important in machine learning
#When I have 100 input data for training, use 90 data and 10 for test and check accuracy
