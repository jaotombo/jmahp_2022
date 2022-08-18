# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:39:58 2020

@author: Franck
"""


#%% Import data

# Import usual modules
import os
import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
import statsmodels.api as sm

# plotting
import seaborn as sns 
import matplotlib.pyplot as plt

plt.style.use('seaborn')

pd.set_option('display.max_columns', 200) # affichage de plus de colonnes dans la console


# Import all the necessary CV and Performance modules
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

#%% import rawdata in csv format

path = "F:/AP-HM/Rehospitalisation/Data"

data = pd.read_csv(path+"/sejour_quanti_quali_adulte_73k.csv", low_memory=False, index_col=0)

data.info()
data.head()

#%% Prepare the data for analysis

# Define the outcome variable
outcomeName =  "loscat"
y_data = data[outcomeName]
print(y_data.value_counts())
print(y_data.value_counts()/y_data.shape[0])


#%% Exclude Severity and Aggregated Charlson

X_data = data.drop([outcomeName,"LOS","age","Severity","charlscore"], axis=1)

X_data.info()

#%% Generate onehot encoding

# Get dummies
X_dummy = pd.get_dummies(X_data, prefix_sep='_', drop_first=True)

# X head
X_dummy.info()

#%% Select 80% of the full dataset for training

X_train_full, X_test, y_train_full, y_test = train_test_split(X_dummy, y_data, test_size=0.20, random_state=42, stratify=y_data)

# Check stratification frequency and proportion
print(y_train_full.value_counts())
print(y_train_full.value_counts()/y_train_full.shape[0])

#%%

#================================================
# Moving into modeling with different classifiers
#================================================

REPEAT = 10

#%%

# initialize CART
ct_auc = []
ct_accuracy = []
ct_impMat = []
ct_proba = pd.DataFrame({'A' : []})
ct_class = pd.DataFrame({'A' : []})
ct_best = []
ct_params = []


for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True, stratify=y_train_full)


#--------------------
# CART
#--------------------

    # Import the Decision Tree Model
    from sklearn.tree import DecisionTreeClassifier
    
    # Choose a sample space grid to optimize the model

    param_ct = {"min_samples_split": range(1,10),
                   "min_samples_leaf": range(1,60)}
        
    # Instantiate the Decision Tree : ct
    ct = DecisionTreeClassifier()
    
    grid_ct = GridSearchCV(ct, param_ct, cv=10, n_jobs=14, 
                          scoring=['accuracy','roc_auc'], refit = 'roc_auc')
    
    # Fit the model to the training data
    grid_ct.fit(X_train, y_train)
    
    # Select the best model
    best_ct = grid_ct.best_estimator_
    ct_best.append(best_ct)
    
     # Compute predicted probabilities: y_pred_proba
    ct_pred_proba = pd.DataFrame(best_ct.predict_proba(X_val)[:,1])
    ct_pred_class = pd.DataFrame(best_ct.predict(X_val))
    ct_proba = pd.concat([ct_proba,ct_pred_proba], axis=1)
    ct_class = pd.concat([ct_class,ct_pred_class], axis=1)

    # Save the parameters
    ct_params.append(grid_ct.best_params_)
      
    # Compute scores
    ct_auc.append(roc_auc_score(y_val, ct_pred_proba))
    ct_accuracy.append(accuracy_score(y_val, ct_pred_class))

    # Print the optimal parameters and best score
    print("Tuned Gradient Boosting Classifier parameters: {}".format(grid_ct.best_params_))
    print("Tuned Gradient Boosting Classifier Accuracy: {}".format(accuracy_score(y_val, ct_pred_class)))
    print("Tuned Gradient Boosting Classifier AUC: {}".format(roc_auc_score(y_val, ct_pred_proba)))
    
    ct_imp = permutation_importance(best_ct, X_val, y_val, scoring='roc_auc')
    ct_impMat.append(ct_imp.importances_mean)

#%%

#%%

#--------------------
# Decision Tree
#--------------------
# Compute mean probability prediction :

ct_proba_mean = ct_proba.mean(axis=1)
ct_class_mean = ct_class.mean(axis=1)
    
# Compute and print mean AUC score
ct_meanAccuracy = np.mean(ct_accuracy)
print("'\n' average Accuracy for the Decision Tree Classifier: {:.3f}".format(ct_meanAccuracy))

ct_meanAuc = np.mean(ct_auc) 
print("'\n' average AUC for the Decision Tree Classifier: {:.3f}".format(ct_meanAuc))

ct_scoreData = pd.DataFrame( {'Accuracy':[ct_meanAccuracy], 'ROC AUC':[ct_meanAuc]})
ct_impMean = np.mean(ct_impMat, axis=0)
ct_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':ct_impMean/ct_impMean.max()*100})

ct_result = ct_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Decision Tree Classifier '\n'{}".format(ct_result))

#%% Evaluating Test Sample
best_ct = ct_best[ct_auc.index(max(ct_auc))]
ct_test_pred = best_ct.predict(X_test)
ct_test_pred_proba = best_ct.predict_proba(X_test)
ct_test_accuracy = accuracy_score(y_test, ct_test_pred)
ct_test_auc = roc_auc_score(y_test, ct_test_pred_proba[:,1])

ct_best_params = ct_params[ct_auc.index(max(ct_auc))]

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(ct_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(ct_test_auc))
print("'\n' Best Parameters :", ct_best_params)

#%% Saving model
model_path = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"

from joblib import dump, load
dump(best_ct, model_path+"/cart.joblib")

#%% Test model importing

test_model = load(model_path+"/cart.joblib")

test_predict = test_model.predict(X_test)
test_predict_proba = test_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, test_predict)
test_auc = roc_auc_score(y_test, test_predict_proba[:,1])

