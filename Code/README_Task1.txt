This file contains information on running the code for Task 1 of Assignment 2
Security Analytics, 2020.

Following are the overall dependencies to run the Python Code for Task 1:

- Python 3
- pandas
- numpy
- datetime
- h5py
- scipy
- PyOD
- sklearn
- tqdm
- re


All the code files can directly be run provided the input files are in the
current working directory. Output files will also be generated in the cwd.


Following are the Python files corresponding to Task 1:


- Feature_generation_labelled_data.py: 

This file preprocesses and cleans the raw data to generate 3 different feature 
sets for labelled data that will further be used with machine learning models.

The output of this file are the 3 different feature sets in the form of CSV 
or HD5 files.

The code is referenced from the below paper with changes wherever needed:

Delplace, A., Hermoso, S., & Anandita, K. (2020). Cyber Attack Detection thanks
to Machine Learning Algorithms. arXiv preprint arXiv:2001.06309.
Link to Repo: https://github.com/antoinedelplace/Cyberattack-Detection



- Feature_generation_unlabelled_data.py:

This file preprocesses and cleans the raw data to generate 3 different feature 
sets for unlabelled data that will further be used with machine learning models.

The output of this file are the 3 different feature sets in the form of CSV 
or HD5 files.

The code is referenced from the below paper with changes wherever needed:

Delplace, A., Hermoso, S., & Anandita, K. (2020). Cyber Attack Detection thanks
to Machine Learning Algorithms. arXiv preprint arXiv:2001.06309.
Link to Repo: https://github.com/antoinedelplace/Cyberattack-Detection



- Categorize_Labels.py:

This file reads data that contains labels (validation set) and categorizes them
into '1' and '0' depending on whether the label contains the word 'Botnet' or
not. If the label contains the term 'Botnet', it is classified as 1, else 0.

The final output are CSVs of the corresponding files.



- iForest_Algo.py

This file uses the iForest algorithm for performing botnet detection through
unsupervised learning on the 3 separately generated feature sets.
The end outputs are 3 CSV files containing all predicted anomalous traffic 
for the 3 different feature sets.

The below code has references to the PYOD toolkit as cited below:
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable
Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
Documentation link: https://pyod.readthedocs.io/en/latest/



- CBLOF_Algo.py:


The below code uses the CBLOF algorithm for performing botnet detection through
unsupervised learning on the 3 separately generated feature sets.
The end outputs are 3 CSV files containing all predicted anomalous traffic for
the 3 different feature sets.

The below code has references to the PYOD toolkit as cited below:
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
Documentation link: https://pyod.readthedocs.io/en/latest/


