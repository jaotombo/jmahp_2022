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

# initialize RANDOM FOREST
rf_auc = []
rf_accuracy = []
rf_impMat = []
rf_proba = pd.DataFrame({'A' : []})
rf_class = pd.DataFrame({'A' : []})
rf_best = []
rf_params = []


for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True, stratify=y_train_full)


#--------------------
# RANDOM FOREST
#--------------------

    # Import the Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    
    # Choose a sample space grid to optimize the model

    param_rf = {"n_estimators": [100, 350, 500],
                "max_depth" : [None, 8, 32],
              "max_features": ["auto"],
              "min_samples_leaf": [2, 10, 30]}
        
    # Instantiate the Random Forest : ct
    rf = RandomForestClassifier()
    
    random_rf = RandomizedSearchCV(rf, param_rf, cv=10, n_jobs=14, 
                          scoring=['accuracy','roc_auc'], refit = 'roc_auc')
    
    # Fit the model to the training data
    random_rf.fit(X_train, y_train)
    
    # Select the best model
    best_rf = random_rf.best_estimator_
    rf_best.append(best_rf)
    
    # Print the optimal parameters and best score
    print("Tuned Random Forest Classifier parameters: {}".format(random_rf.best_params_))
    print("Tuned Random Forest Classifier AUC: {}".format(random_rf.best_score_))
    
    # Compute predicted probabilities: y_pred_proba
    rf_pred_proba = pd.DataFrame(best_rf.predict_proba(X_val)[:,1])
    rf_pred_class = pd.DataFrame(best_rf.predict(X_val))
    rf_proba = pd.concat([rf_proba,rf_pred_proba], axis=1)
    rf_class = pd.concat([rf_class,rf_pred_class], axis=1)

    # Save the parameters
    rf_params.append(random_rf.best_params_)
      
    # Compute scores
    rf_auc.append(roc_auc_score(y_val, rf_pred_proba))
    rf_accuracy.append(accuracy_score(y_val, rf_pred_class))

    # Print the optimal parameters and best score
    print("Tuned Random Forest Classifier parameters: {}".format(random_rf.best_params_))
    print("Tuned Random Forest Classifier Accuracy: {}".format(accuracy_score(y_val, rf_pred_class)))
    print("Tuned Random Forest Classifier AUC: {}".format(roc_auc_score(y_val, rf_pred_proba)))
    
    # par souci de comparabilité, on utilise ce mode de calcul
    rf_imp = permutation_importance(best_rf, X_val, y_val, scoring='roc_auc')
    rf_impMat.append(rf_imp.importances_mean)


#%%

#--------------------
# Random Forest
#--------------------
# Compute mean probability prediction :

rf_proba_mean = rf_proba.mean(axis=1)
rf_class_mean = rf_class.mean(axis=1)
    
# Compute and print mean AUC score
rf_meanAccuracy = np.mean(rf_accuracy)
print("'\n' average Accuracy for the Random Forest Classifier: {:.3f}".format(rf_meanAccuracy))

rf_meanAuc = np.mean(rf_auc) 
print("'\n' average AUC for the Random Forest Classifier: {:.3f}".format(rf_meanAuc))

rf_scoreData = pd.DataFrame( {'Accuracy':[rf_meanAccuracy], 'ROC AUC':[rf_meanAuc]})
rf_impMean = np.mean(rf_impMat, axis=0)
rf_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':rf_impMean/rf_impMean.max()*100})

rf_result = rf_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Random Forest Classifier '\n'{}".format(rf_result))

#%% Evaluating Test Sample
best_rf = rf_best[rf_auc.index(max(rf_auc))]
rf_test_pred = best_rf.predict(X_test)
rf_test_pred_proba = best_rf.predict_proba(X_test)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
rf_test_auc = roc_auc_score(y_test, rf_test_pred_proba[:,1])

rf_best_params = rf_params[rf_auc.index(max(rf_auc))]

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(rf_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(rf_test_auc))
print("'\n' Best Parameters :", rf_best_params)

#%% Saving model
model_path = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"

from joblib import dump, load
dump(best_rf, model_path+"/randomforest.joblib")

#%% Test model importing

test_model = load(model_path+"/randomforest.joblib")

test_predict = test_model.predict(X_test)
test_predict_proba = test_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, test_predict)
test_auc = roc_auc_score(y_test, test_predict_proba[:,1])

