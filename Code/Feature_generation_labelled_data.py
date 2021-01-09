"""

Yazad Davur - 1050178

This python file corresponds to experiments performed to accomplish Task 1 and Task 2 of Security Analytics - Assignment 2, 2020.

The below code preprocesses and cleans the data to generate 3 different feature sets for labelled data that will further be used with machine learning models.

The output of this file are the 3 different feature sets in the form of CSV or HD5 files.

The below code is referenced from the below paper with changes wherever needed:

Delplace, A., Hermoso, S., & Anandita, K. (2020). Cyber Attack Detection thanks to Machine Learning Algorithms. arXiv preprint arXiv:2001.06309.

Link to Repo: https://github.com/antoinedelplace/Cyberattack-Detection
"""

import pandas as pd
import numpy as np
import datetime
import h5py
from scipy.stats import mode

data = pd.read_csv("bin_labelled_test.csv") # This data file can be changed to accommodate any labelled data file

labels = np.load("valid_label.npy")
label_df = pd.DataFrame(data=labels)

data.columns = ['Date_Flow_Start', 'Duration','Protocol','Src_IP','Src_Port','Direction','Dst_IP','Dst_Port','State','Source_Service','Dest_Service','Total_Packets','BiDirection_Bytes','SrcToDst_Bytes', 'Label']

data['Label'] = label_df[0]

"""Set 1 Feature Generation"""

#Setting up window width and stride
window_width = 120 
window_stride = 60

data['Date_Flow_Start'] = pd.to_datetime(data['Date_Flow_Start']).astype(np.int64)*1e-9
datetime_start = data['Date_Flow_Start'].min()
data['Window_lower'] = (data['Date_Flow_Start']-datetime_start-window_width)/window_stride+1
data['Window_lower'].clip(lower=0, inplace=True)
data['Window_upper_excl'] = (data['Date_Flow_Start']-datetime_start)/window_stride+1
data = data.astype({"Window_lower": int, "Window_upper_excl": int})
data.drop('Date_Flow_Start', axis=1, inplace=True)
labels=data['Label']

X = pd.DataFrame()
nb_windows = data['Window_upper_excl'].max()

for i in range(0, nb_windows):
    gb = data.loc[(data['Window_lower'] <= i) & (data['Window_upper_excl'] > i)].groupby('Src_IP')
    X = X.append(gb.size().to_frame(name='counts').join(gb.agg({'Src_Port':'nunique', 
                                                       'Dst_IP':'nunique', 
                                                       'Dst_Port':'nunique', 
                                                       'Duration':['sum', 'mean', 'std', 'max', 'median'],
                                                       'BiDirection_Bytes':['sum', 'mean', 'std', 'max', 'median'],
                                                       'SrcToDst_Bytes':['sum', 'mean', 'std', 'max', 'median'],
                                                       'Label':lambda x: mode(x)[0]})).reset_index().assign(window_id=i))

X.columns = ["_".join(x) if isinstance(x, tuple) else x for x in X.columns.ravel()]
X.fillna(-1, inplace=True)

print("Preprocessing")
def normalize_column(dt, column):
    mean = dt[column].mean()
    std = dt[column].std()
    print(mean, std)
    dt[column] = (dt[column]-mean) / std

columns_to_normalize = list(X.columns.values)
columns_to_normalize.remove('Src_IP')
columns_to_normalize.remove('Label_<lambda>')
columns_to_normalize.remove('window_id')

normalize_column(X, columns_to_normalize)

with pd.option_context('display.max_rows', 10, 'display.max_columns', 22):
    print(X.shape)
    print(X)
    print(X.dtypes)

X.drop('Src_IP', axis=1).to_hdf('preprocessing1_test_task2.h5', key="data", mode="w")
np.save("preprocessing1_test_srcIP_task2.npy", X['Src_IP'])
np.save("preprocessing1_test_labels_task2.npy", labels)
X = X.drop('Src_IP', axis=1)
X.to_csv("preprocessing_valid1.csv")

"""Set 2 Feature Generation"""

def RU(df):
    if df.shape[0] == 1:
        return 1.0
    else:
        proba = df.value_counts()/df.shape[0]
        h = proba*np.log10(proba)
        return -h.sum()/np.log10(df.shape[0])

X = pd.DataFrame()
nb_windows = data['Window_upper_excl'].max()

for i in range(0, nb_windows):
    gb = data.loc[(data['Window_lower'] <= i) & (data['Window_upper_excl'] > i)].groupby('Src_IP')
    X = X.append(gb.agg({'Src_Port':[RU], 
                         'Dst_IP':[RU], 
                         'Dst_Port':[RU]}).reset_index())
    print(X.shape)

X.columns = ["_".join(x) if isinstance(x, tuple) else x for x in X.columns.ravel()]

columns_to_normalize = list(X.columns.values)
columns_to_normalize.remove('Src_IP_')

normalize_column(X, columns_to_normalize)

with pd.option_context('display.max_rows', 10, 'display.max_columns', 22):
    print(X.shape)
    print(X)
    print(X.dtypes)

X.drop('Src_IP_', axis=1).to_hdf('preprocessing2_test_task2.h5', key="data", mode="w")
np.save("preprocessing2_test_srcIP_task2.npy", X['Src_IP_'])
np.save("preprocessing2_test_labels_task2.npy", labels)
X = X.drop('Src_IP_', axis=1)
X.to_csv("preprocessing_valid2.csv")

"""Set 3 Feature Generation

Set 3 combines and eliminates features on the basis of pearson correlation from the first 2 sets
"""

X = pd.read_hdf('preprocessing1_test_task2.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('preprocessing2_test_task2.h5', key='data')
X2.reset_index(drop=True, inplace=True)

X = X.join(X2)
X.drop('window_id', axis=1, inplace=True)

X.to_csv("test_data_preprocessed_task2.csv")

df = pd.read_csv("test_data_preprocessed.csv",index_col=False)
df = df.drop(['Unnamed: 0'], axis=1)

corr = df.corr()

# Columns with correlation factor magnitude greater than 0.8 are eliminated. 
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.8:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
print("No. of columns eliminated due to high correlation", (len(df.columns)-len(selected_columns)))

df_new = df[selected_columns]
df_new.to_csv("preprocessing_test3.csv")
