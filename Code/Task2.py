"""

Yazad Davur - 1050178

This python file corresponds to experiments performed to accomplish Task 2 of Security Analytics - Assignment 2, 2020.

The below code implements a Support Vector Classifier to classify botnet traffic against normal traffic. It further
generates adversarial samples and shows how these samples are able to bypass the original discriminator.
As output, a CSV is generated containing the feature-wise numerical values of the orginal sample, corresponding
adversarial sample and the difference between the two for analysis.

The below code has been referenced from the following paper with changes wherever necessary.

Nicolae, M. I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M., ... & Molloy, I. M. (2018). Adversarial Robustness Toolbox v1. 0.0. arXiv preprint arXiv:1807.01069.

Link to Repo: https://github.com/Trusted-AI/adversarial-robustness-toolbox
"""


import pandas as pd
import numpy as np
from sklearn import metrics
from art.classifiers import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from art.attacks.evasion import FastGradientMethod

#Read all data for Task 2
X_train = pd.read_csv("train_data_preprocessed_task2.csv", index_col=0)
X_test = pd.read_csv("test_data_preprocessed_task2.csv", index_col=0)
X_valid = pd.read_csv("valid_data_preprocessed_task2.csv", index_col=0)

#Extract Labels for each partition
Y_train = X_train[['Label_<lambda>']]
Y_test = X_test[['Label_<lambda>']]
Y_valid = X_valid[['Label_<lambda>']]

# Extract outlier proportion for analysis and drop labels from training sets
train = X_train
valid = X_valid
train_normal = train[train['Label_<lambda>']==0]
train_outliers = train[train['Label_<lambda>']==1]
train.drop(['Label_<lambda>'], inplace=True, axis=1)
valid.drop(['Label_<lambda>'], inplace=True, axis=1)
outlier_prop = len(train_outliers) / len(train_normal)

outlier_prop #Proportion of outliers w.r.t. to normal traffic

# One hot encoding of the labels
y_train_label = pd.get_dummies(Y_train['Label_<lambda>'])
y_valid_label = pd.get_dummies(Y_valid['Label_<lambda>'])

#Fit the model on the wrapped Support Vector Classifier
svm = SVC(gamma=0.001)
art_classifier = SklearnClassifier(model=svm)
art_classifier.fit(train, np.array(y_train_label))

"""Evaluating performance on the Train Data."""

predictions_train = art_classifier.predict(train)
pred_train = []
for each in predictions_train:
    pred_train.append(np.argmax(each))
precision = metrics.precision_score(Y_train['Label_<lambda>'], pred_train)
accuracy = metrics.accuracy_score(Y_train['Label_<lambda>'], pred_train)
f1 = metrics.f1_score(Y_train['Label_<lambda>'], pred_train)
print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

"""Evaluating performance on the Validation Data"""

predictions = art_classifier.predict(valid)
pred = []
for each in predictions:
    pred.append(np.argmax(each))

precision = metrics.precision_score(Y_valid['Label_<lambda>'], pred)
accuracy = metrics.accuracy_score(Y_valid['Label_<lambda>'], pred)
f1 = metrics.f1_score(Y_valid['Label_<lambda>'], pred)
print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

"""Evaluating performance on the Test Data"""

#One hot encoding test labels and dropping label column from the X_test data
y_test_label = pd.get_dummies(Y_test['Label_<lambda>'])
X_test.drop(['Label_<lambda>'], inplace=True, axis=1)

predictions_test = art_classifier.predict(X_test)
pred_test = []
for each in predictions_test:
    pred_test.append(np.argmax(each))

precision = metrics.precision_score(Y_test['Label_<lambda>'], pred_test)
accuracy = metrics.accuracy_score(Y_test['Label_<lambda>'], pred_test)
f1 = metrics.f1_score(Y_test['Label_<lambda>'], pred_test)
print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

"""Generating adversarial samples"""

#Using FGM to generate adverserial samples
attack = FastGradientMethod(estimator=art_classifier, eps=0.3, targeted=False)
x_test_adv = attack.generate(x=X_test)

predictions_adv = art_classifier.predict(x_test_adv)
pred_adv = []
for each in predictions_adv:
    pred_adv.append(np.argmax(each))

precision = metrics.precision_score(Y_test['Label_<lambda>'], pred_adv)
accuracy = metrics.accuracy_score(Y_test['Label_<lambda>'], pred_adv)
f1 = metrics.f1_score(Y_test['Label_<lambda>'], pred_adv)

# Metric scores plummet down as the model has been bypassed several times involving misclassification
print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

#Generates all those index values of test data where the flow is actually botnet traffic, 
# and the prediction by the SVC model is correct (TP) but bypasses the model due to added perturbations
print(len(Y_test['Label_<lambda>']))
print(len(pred_adv))
print(len(pred_test))
ground_truth = Y_test['Label_<lambda>'].to_list()
spcl_lst = []
for index in range(0,len(pred_adv)):
  if ground_truth[index] == 1 and pred_test[index] == 1 and pred_adv[index] == 0:
    print(index)
    spcl_lst.append(index)

chosen_index = 5370 # Index chosen randomly out of the above list

"""Capturing difference between the Actual and Adverserial sample"""

np.array(X_test.loc[chosen_index]) # Actual sample data

x_test_adv[chosen_index] # Adverserial sample data

print(np.array(X_test.loc[chosen_index]) - x_test_adv[chosen_index]) # Perturbation values

test_SrcIP = list(np.load("preprocessing2_test_srcIP_task2.npy", allow_pickle=True))

test_SrcIP[chosen_index] #Extracting Src_IP for the particular index

cols = list(X_test.columns)
df = pd.DataFrame(columns=cols)

df.loc["Original"] = list(np.array(X_test.loc[chosen_index]))

df.loc["Adversarial"] = list(x_test_adv[chosen_index])

df.loc["Perturbation"] = list(np.array(X_test.loc[chosen_index]) - x_test_adv[chosen_index])

df #DataFrame containing Actual, Adverserial and Perturbation data values for the chosen botnet IP address

df.to_csv(test_SrcIP[chosen_index]+".csv") # Write to CSV
