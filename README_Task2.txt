This file contains information on running the code for Task 2 of Assignment 2
Security Analytics, 2020.

Following are the overall dependencies to run the Python Code for Task 2:

- Python 3
- pandas
- numpy
- sklearn
- tqdm
- re
- art (Aversarial Robustness Toolbox)


All the code files can directly be run provided the input files are in the
current working directory. Output files will also be generated in the cwd.


Following are the Python files corresponding to Task 2:


- Categorize_Labels.py:

This file reads data that contains labels (train, valid and test) and
categorizes them into '1' and '0' depending on whether the label contains
the word 'Botnet' or not. If the label contains the term 'Botnet', it is 
classified as 1, else 0.
The final output are CSVs of the corresponding files.



- Task2.py:

The below code implements a Support Vector Classifier to classify botnet
traffic against normal traffic. It further generates adversarial samples
and shows how these samples are able to bypass the original discriminator.
As output, a CSV is generated containing the feature-wise numerical values
of the orginal sample, corresponding adversarial sample and the difference
between the two for analysis.

The below code has been referenced from the following paper with changes
wherever necessary.
Nicolae, M. I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M.,
... & Molloy, I. M. (2018). Adversarial Robustness Toolbox v1. 0.0. arXiv
preprint arXiv:1807.01069.

Link to Repo: https://github.com/Trusted-AI/adversarial-robustness-toolbox


