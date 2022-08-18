# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:28:44 2020

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

# initialize elasticnet
en_auc = []
en_accuracy = []
en_impMat = []
en_proba = pd.DataFrame({'A' : []})
en_class = pd.DataFrame({'A' : []})
en_best = []
en_params = []

for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True, stratify=y_train_full)


#--------------------
# Elasticnet
#--------------------

    # Import the Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    
    # Choose a sample space grid to optimize the model

    param_en = {'C': np.logspace(-3, 4, 10), 'l1_ratio':np.linspace(0,1,11) }
        
    # Instantiate the Logistic Regression Classifier : en
    en = LogisticRegression(max_iter=200, penalty ='elasticnet', solver='saga')
    
    grid_en = GridSearchCV(en, param_en, cv=10, n_jobs=14, 
                          scoring=['accuracy','roc_auc'], refit = 'roc_auc')
    
    # grid_en = GridSearchCV(en, param_en, cv=10, n_jobs=14, 
    #                       scoring=['accuracy','roc_auc'], refit = 'accuracy')
    
    # Fit the model to the training data
    grid_en.fit(X_train, y_train)
    
    # Select the best model
    best_en = grid_en.best_estimator_
    en_best.append(best_en)
      
    # Compute predicted probabilities: y_pred_proba
    en_pred_proba = pd.DataFrame(best_en.predict_proba(X_val)[:,1])
    en_pred_class = pd.DataFrame(best_en.predict(X_val))
    en_proba = pd.concat([en_proba,en_pred_proba], axis=1)
    en_class = pd.concat([en_class,en_pred_class], axis=1)

    # Print the optimal parameters and best score
    print("Tuned Regularized Logistic Regression Parameters: {}".format(grid_en.best_params_))
    print("Tuned Regularized Logistic Regression Accuracy: {}".format(accuracy_score(y_val, en_pred_class)))
    print("Tuned Regularized Logistic Regression AUC: {}".format(roc_auc_score(y_val, en_pred_proba)))

    # Save the parameters
    en_params.append(grid_en.best_params_)
    
    # Compute scores
    en_accuracy.append(accuracy_score(y_val, en_pred_class))
    en_auc.append(roc_auc_score(y_val, en_pred_proba))
    
    en_imp = permutation_importance(best_en, X_val, y_val, scoring='roc_auc')
    en_impMat.append(en_imp.importances_mean)


#%%

#--------------------
# Elasticnet
#--------------------
# Compute mean probability prediction :

en_proba_mean = en_proba.mean(axis=1)
en_class_mean = en_class.mean(axis=1)
    
# Compute and print mean AUC score
en_meanAccuracy = np.mean(en_accuracy)
print("'\n' average Accuracy for the penalized Logistic Regression: {:.3f}".format(en_meanAccuracy))

en_meanAuc = np.mean(en_auc) 
print("'\n' average AUC for the penalized Logistic Regression: {:.3f}".format(en_meanAuc))

en_scoreData = pd.DataFrame( {'Accuracy':[en_meanAccuracy], 'ROC AUC':[en_meanAuc]})
en_impMean = np.mean(en_impMat, axis=0)
en_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':en_impMean/en_impMean.max()*100})

en_result = en_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Logistic Regression '\n'{}".format(en_result))

#%% Evaluating Test Sample
best_en = en_best[en_auc.index(max(en_auc))]
en_test_pred = best_en.predict(X_test)
en_test_pred_proba = best_en.predict_proba(X_test)
en_test_accuracy = accuracy_score(y_test, en_test_pred)
en_test_auc = roc_auc_score(y_test, en_test_pred_proba[:,1])

en_best_params = en_params[en_auc.index(max(en_auc))]

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(en_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(en_test_auc))
print("'\n' Best Parameters :", en_best_params)

#%% Saving model
model_path = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"

from joblib import dump, load
dump(best_en, model_path+"/elasticnet_severity.joblib")

#%% Test model importing

test_model = load(model_path+"/elasticnet_severity.joblib")

test_predict = test_model.predict(X_test)
test_predict_proba = test_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, test_predict)
test_auc = roc_auc_score(y_test, test_predict_proba[:,1])

