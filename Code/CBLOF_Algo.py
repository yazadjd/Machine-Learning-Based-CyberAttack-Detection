"""

Yazad Davur - 1050178

This python file corresponds to experiments performed to accomplish Task 1 of Security Analytics - Assignment 2, 2020.

The below code uses the CBLOF algorithm for performing botnet detection through unsupervised learning on the 3 separately generated feature sets.

The end outputs are 3 CSV files containing all predicted anomalous traffic for the 3 different feature sets.

The below code has references to the PYOD toolkit as cited below:

Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.

Documentation link: https://pyod.readthedocs.io/en/latest/
"""

pip install pyod

import pandas as pd
import numpy as np
from pyod.models.cblof import CBLOF
from sklearn import metrics


# Reading train data for first feature set
X_train1 = pd.read_hdf('preprocessing1.h5', key='data')
X_train1.reset_index(drop=True, inplace=True)
X_train1.drop('window_id', axis=1, inplace=True)

# Reading train data for second feature set
X_train2 = pd.read_hdf('preprocessing2.h5', key='data')
X_train2.reset_index(drop=True, inplace=True)

# Reading train data for third feature set
X_train3 = pd.read_csv('training_data_preprocessed3.csv')
X_train3 = X_train3.drop(['Unnamed: 0'], axis=1)

# Reading Test Data for first feature set
X_test1 = pd.read_hdf('preprocessing1_test.h5', key='data')
X_test1.reset_index(drop=True, inplace=True)
X_test1.drop('window_id', axis=1, inplace=True)

# Reading Test Data for second feature set
X_test2 = pd.read_hdf('preprocessing2_test.h5', key='data')
X_test2.reset_index(drop=True, inplace=True)

# Reading Test Data for third feature set
X_test3 = pd.read_csv('test_data_preprocessed.csv')
X_test3= X_test3.drop(['Unnamed: 0'], axis=1)
X_test3.drop(['Duration_max',	'Duration_median', 'BiDirection_Bytes_median', 'SrcToDst_Bytes_median'], inplace = True, axis = 1)

# Reading Validation data for first feature set
X_valid1 = pd.read_hdf('preprocessing1_valid.h5', key='data')
X_valid1.reset_index(drop=True, inplace=True)
X_valid1.drop('window_id', axis=1, inplace=True)

# Reading Validation data for second feature set
X_valid2 = pd.read_hdf('preprocessing2_valid.h5', key='data')
X_valid2.reset_index(drop=True, inplace=True)

# Reading Validation data for third feature set
X_valid3 = pd.read_csv('valid_data_preprocessed.csv')
X_valid3= X_valid3.drop(['Unnamed: 0'], axis=1)
X_valid3.drop(['Duration_max',	'Duration_median', 'BiDirection_Bytes_median', 'SrcToDst_Bytes_median', 'Label_<lambda>'], inplace = True, axis = 1)

#Extracting y-labels for the validation data and dropping in X data. Y labels will be the same for all feature sets ofcourse
Y_valid1 = X_valid1['Label_<lambda>']
X_valid1.drop(['Label_<lambda>'], inplace=True, axis=1)

# Reading original test data to extract the malicious flow data after prediction
orig_test_data = pd.read_csv("test_data.csv", header=None)
orig_test_data.columns = ['Date_Flow_Start', 'Duration','Protocol','Src_IP','Src_Port','Direction','Dst_IP','Dst_Port','State','Source_Service','Dest_Service','Total_Packets','BiDirection_Bytes','SrcToDst_Bytes']

""" Training on Feature Set 1

CBLOF on Default Parameters
"""

clf1 = CBLOF(random_state=42) # Default contamination 0.1
clf1.fit(X_train1)

#Setting threshold using the contamination parameter
dec_scores = clf1.decision_scores_
dec_scores_sorted=sorted(dec_scores, reverse=True)
a = round(len(X_train1) * clf1.contamination)
print(a)

anomalies=dec_scores_sorted[:a]
threshold = anomalies[-1]
print(threshold)

# Validation data is scored
y_valid_scores = clf1.decision_function(X_valid1)
y_valid_scores = pd.Series(y_valid_scores)

valid_SrcIP = np.load('preprocessing1_valid_srcIP.npy',allow_pickle=True)

# For each score, if it is above threshold value, it is considered outlier, else inlier
valid_outliers = []
y_pred_valid = []
for score in range(0,len(y_valid_scores)):
  if y_valid_scores[score] > threshold:
    reg = (valid_SrcIP[score], y_valid_scores[score])
    valid_outliers.append(reg)
    y_pred_valid.append(1.0)
  else:
    y_pred_valid.append(0.0)

precision = metrics.precision_score(Y_valid1, y_pred_valid)
accuracy = metrics.accuracy_score(Y_valid1, y_pred_valid)
f1 = metrics.f1_score(Y_valid1, y_pred_valid)

print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

"""Experiment with Tuned Contamination"""

clf1 = CBLOF(random_state = 42, contamination=0.125) 
clf1.fit(X_train1)

dec_scores = clf1.decision_scores_
dec_scores_sorted=sorted(dec_scores, reverse=True)
a = round(len(X_train1) * clf1.contamination)
print(a)

anomalies=dec_scores_sorted[:a]
threshold = anomalies[-1]
print(threshold)

y_valid_scores = clf1.decision_function(X_valid1)
y_valid_scores = pd.Series(y_valid_scores)

valid_SrcIP = np.load('preprocessing1_valid_srcIP.npy',allow_pickle=True)

valid_outliers = []
y_pred_valid = []
for score in range(0,len(y_valid_scores)):
  if y_valid_scores[score] > threshold:
    reg = (valid_SrcIP[score], y_valid_scores[score])
    valid_outliers.append(reg)
    y_pred_valid.append(1.0)
  else:
    y_pred_valid.append(0.0)

precision = metrics.precision_score(Y_valid1, y_pred_valid)
accuracy = metrics.accuracy_score(Y_valid1, y_pred_valid)
f1 = metrics.f1_score(Y_valid1, y_pred_valid)

print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

# Test data is scored
y_test_scores = clf1.decision_function(X_test1)  # outlier scores
y_test_scores = pd.Series(y_test_scores)

test_SrcIP = np.load('preprocessing1_test.npy',allow_pickle=True) # File used to trace the IP address of each index

# For each outlier score, if it is above threshold value, it is considered outlier, else inlier
test_outliers = []
y_pred_test = []
for score in range(0,len(y_test_scores)):
  if y_test_scores[score] > threshold:
    reg = (test_SrcIP[score], y_test_scores[score])
    test_outliers.append(reg)
    y_pred_test.append(1.0)
  else:
    y_pred_test.append(0.0)

src_ip_list = []
for each in test_outliers:
  src_ip_list.append(each[0])
src_ip_list = list(set(src_ip_list))

# Create new dataframe to extract all anomalous data
extracted_rows = pd.DataFrame(columns=['Date_Flow_Start', 'Duration','Protocol','Src_IP','Src_Port','Direction','Dst_IP','Dst_Port','State','Source_Service','Dest_Service','Total_Packets','BiDirection_Bytes','SrcToDst_Bytes'])

# Extract all anomalous data from test data and append into a dataframe
for each in src_ip_list:
  df = orig_test_data[orig_test_data.Src_IP == each]
  extracted_rows = extracted_rows.append(df)

# Drop unrequired data
extracted_rows.drop(labels=['Direction', 'State', 'Source_Service',	'Dest_Service', 'Total_Packets', 'BiDirection_Bytes',	'SrcToDst_Bytes'], inplace=True, axis=1)

extracted_rows.to_csv('CBLOF_Set1.csv', index = True) #Write to CSV

"""Training similarly on Feature Set 2

CBLOF on Default Parameters
"""

clf1 = CBLOF(random_state = 42) #Contamination = 0.1
clf1.fit(X_train2)

dec_scores = clf1.decision_scores_
dec_scores_sorted=sorted(dec_scores, reverse=True)
a = round(len(X_train2) * clf1.contamination)
print(a)

anomalies=dec_scores_sorted[:a]
threshold = anomalies[-1]
print(threshold)

y_valid_scores = clf1.decision_function(X_valid2)
y_valid_scores = pd.Series(y_valid_scores)

valid_SrcIP = np.load('preprocessing2_valid_srcIP.npy', allow_pickle=True)

valid_outliers = []
y_pred_valid = []
for score in range(0,len(y_valid_scores)):
  if y_valid_scores[score] > threshold:
    reg = (valid_SrcIP[score], y_valid_scores[score])
    valid_outliers.append(reg)
    y_pred_valid.append(1.0)
  else:
    y_pred_valid.append(0.0)

precision = metrics.precision_score(Y_valid1, y_pred_valid)
accuracy = metrics.accuracy_score(Y_valid1, y_pred_valid)
f1 = metrics.f1_score(Y_valid1, y_pred_valid)

print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

"""Experiment with Tuned Contamination"""

clf1 = CBLOF(random_state = 42, contamination=0.175) 
clf1.fit(X_train2)

dec_scores = clf1.decision_scores_
dec_scores_sorted=sorted(dec_scores, reverse=True)
a = round(len(X_train2) * clf1.contamination)
print(a)

anomalies=dec_scores_sorted[:a]
threshold = anomalies[-1]
print(threshold)

y_valid_scores = clf1.decision_function(X_valid2)
y_valid_scores = pd.Series(y_valid_scores)

valid_SrcIP = np.load('preprocessing2_valid_srcIP.npy',allow_pickle=True)

valid_outliers = []
y_pred_valid = []
for score in range(0,len(y_valid_scores)):
  if y_valid_scores[score] > threshold:
    reg = (valid_SrcIP[score], y_valid_scores[score])
    valid_outliers.append(reg)
    y_pred_valid.append(1.0)
  else:
    y_pred_valid.append(0.0)

precision = metrics.precision_score(Y_valid1, y_pred_valid)
accuracy = metrics.accuracy_score(Y_valid1, y_pred_valid)
f1 = metrics.f1_score(Y_valid1, y_pred_valid)

print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

y_test_scores = clf1.decision_function(X_test2)
y_test_scores = pd.Series(y_test_scores)

test_SrcIP = np.load('preprocessing2_test.npy', allow_pickle=True)

test_outliers = []
y_pred_test = []
for score in range(0,len(y_test_scores)):
  if y_test_scores[score] > threshold:
    reg = (test_SrcIP[score], y_test_scores[score])
    test_outliers.append(reg)
    y_pred_test.append(1.0)
  else:
    y_pred_test.append(0.0)

src_ip_list = []
for each in test_outliers:
  src_ip_list.append(each[0])
src_ip_list = list(set(src_ip_list))

extracted_rows = pd.DataFrame(columns=['Date_Flow_Start', 'Duration','Protocol','Src_IP','Src_Port','Direction','Dst_IP','Dst_Port','State','Source_Service','Dest_Service','Total_Packets','BiDirection_Bytes','SrcToDst_Bytes'])

for each in src_ip_list:
  df = orig_test_data[orig_test_data.Src_IP == each]
  extracted_rows = extracted_rows.append(df)

extracted_rows.drop(labels=['Direction', 'State', 'Source_Service',	'Dest_Service', 'Total_Packets', 'BiDirection_Bytes',	'SrcToDst_Bytes'], inplace=True, axis=1)

extracted_rows.to_csv('CBLOF_Set2.csv', index = True)

""" Training similarly on Feature Set 3

CBLOF on Default Parameters
"""

clf1 = CBLOF(random_state = 42) #Default Contamination = 0.1
clf1.fit(X_train3)

dec_scores = clf1.decision_scores_
dec_scores_sorted=sorted(dec_scores, reverse=True)
a = round(len(X_train3) * clf1.contamination)
print(a)

anomalies=dec_scores_sorted[:a]
threshold = anomalies[-1]
print(threshold)

y_valid_scores = clf1.decision_function(X_valid3)
y_valid_scores = pd.Series(y_valid_scores)

valid_SrcIP = np.load('preprocessing2_valid_srcIP.npy', allow_pickle=True)

valid_outliers = []
y_pred_valid = []
for score in range(0,len(y_valid_scores)):
  if y_valid_scores[score] > threshold:
    reg = (valid_SrcIP[score], y_valid_scores[score])
    valid_outliers.append(reg)
    y_pred_valid.append(1.0)
  else:
    y_pred_valid.append(0.0)

precision = metrics.precision_score(Y_valid1, y_pred_valid)
accuracy = metrics.accuracy_score(Y_valid1, y_pred_valid)
f1 = metrics.f1_score(Y_valid1, y_pred_valid)

print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

"""Experiment with Tuned Contamination"""

clf1 = CBLOF(random_state = 42, contamination=0.35) 
clf1.fit(X_train3)

dec_scores = clf1.decision_scores_
dec_scores_sorted=sorted(dec_scores, reverse=True)
a = round(len(X_train3) * clf1.contamination)
print(a)

anomalies=dec_scores_sorted[:a]
threshold = anomalies[-1]
print(threshold)

y_valid_scores = clf1.decision_function(X_valid3)
y_valid_scores = pd.Series(y_valid_scores)

valid_SrcIP = np.load('preprocessing2_valid_srcIP.npy', allow_pickle=True)

valid_outliers = []
y_pred_valid = []
for score in range(0,len(y_valid_scores)):
  if y_valid_scores[score] > threshold:
    reg = (valid_SrcIP[score], y_valid_scores[score])
    valid_outliers.append(reg)
    y_pred_valid.append(1.0)
  else:
    y_pred_valid.append(0.0)

precision = metrics.precision_score(Y_valid1, y_pred_valid)
accuracy = metrics.accuracy_score(Y_valid1, y_pred_valid)
f1 = metrics.f1_score(Y_valid1, y_pred_valid)

print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

y_test_scores = clf1.decision_function(X_test3)
y_test_scores = pd.Series(y_test_scores)

test_SrcIP = np.load('preprocessing2_test.npy', allow_pickle=True)

test_outliers = []
y_pred_test = []
for score in range(0,len(y_test_scores)):
  if y_test_scores[score] > threshold:
    reg = (test_SrcIP[score], y_test_scores[score])
    test_outliers.append(reg)
    y_pred_test.append(1.0)
  else:
    y_pred_test.append(0.0)

src_ip_list = []
for each in test_outliers:
  src_ip_list.append(each[0])
src_ip_list = list(set(src_ip_list))

extracted_rows = pd.DataFrame(columns=['Date_Flow_Start', 'Duration','Protocol','Src_IP','Src_Port','Direction','Dst_IP','Dst_Port','State','Source_Service','Dest_Service','Total_Packets','BiDirection_Bytes','SrcToDst_Bytes'])

for each in src_ip_list:
  df = orig_test_data[orig_test_data.Src_IP == each]
  extracted_rows = extracted_rows.append(df)

extracted_rows.drop(labels=['Direction', 'State', 'Source_Service',	'Dest_Service', 'Total_Packets', 'BiDirection_Bytes',	'SrcToDst_Bytes'], inplace=True, axis=1)

extracted_rows.to_csv('CBLOF_Set3.csv', index = True)

"""THE END"""
