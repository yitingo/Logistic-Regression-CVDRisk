# CA05-Logistic-Regression
I built a binary classifier model to predict the CVD Risk (Yes/No, or 1/0) for patients using a Logistic Regression Model.
After splitting the dataset into 75% training and 25% testing, importing and checking the dataset, I defined grid searching and key parameters.
After training, the model was able to display the Feature Importance of all the features sorted in the order of decreasing influence on the CVD Risk and compute confusion matrix, accuracy/precision/recall scores, ROC curve and AUC score.


Packages used:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
