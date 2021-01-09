"""

Yazad Davur - 1050178

This python file corresponds to experiments performed to accomplish Task 1 and 2 of Security Analytics - Assignment 2, 2020.

The below code reads data that contains labels and categorizes them into 1s and 0s depending on whether the label contains the word 'Botnet' or not.

If the label contains the term 'Botnet', it is classified as 1, else 0.

The final output are CSVs of the corresponding files.
"""

pip install pyod

import pandas as pd
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import re


X_train = pd.read_csv("training_data_with_labels.csv", header=None)
X_train.head()

X_test = pd.read_csv("test_data_with_labels.csv", header=None)
X_test.head()

X_valid = pd.read_csv("valid_data_with_labels.csv", header=None)
X_valid.head()

x = X_valid[14].to_list()

valid_label = []
pattern = re.compile(".*Botnet*")
for item in tqdm(x):
    if pattern.search(str(item)) is not None:
        valid_label.append(1)
    else:
        valid_label.append(0)

x = X_train[14].to_list()

train_label = []
pattern = re.compile(".*Botnet*")
for item in tqdm(x):
    if pattern.search(str(item)) is not None:
        train_label.append(1)
    else:
        train_label.append(0)

x = X_test[14].to_list()

test_label = []
pattern = re.compile(".*Botnet*")
for item in tqdm(x):
    if pattern.search(str(item)) is not None:
        test_label.append(1)
    else:
        test_label.append(0)

X_train[14] = train_label
X_train.head()

X_test[14] = test_label
X_test.head()

X_valid[14] = valid_label
X_valid.head()

X_train.to_csv("bin_labelled_train.csv", index=False)
X_test.to_csv("bin_labelled_test.csv", index=False)
X_valid.to_csv("bin_labelled_valid.csv",index=False)
