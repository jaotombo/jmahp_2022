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

# initialize Gradient Boosting
gb_auc = []
gb_accuracy = []
gb_impMat = []
gb_proba = pd.DataFrame({'A' : []})
gb_class = pd.DataFrame({'A' : []})
gb_best = []
gb_params = []


for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True, stratify=y_train_full)


#--------------------
# Gradient Boosting
#--------------------

    # Import the Gradient Boosting Classifier
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Choose a sample space grid to optimize the model

    param_gb = {'max_depth':[2, 3, 4, 6],
                'subsample': [0.8, 0.9, 1],
                'n_estimators':[100, 200, 300],
                'max_features':['auto','sqrt']
                }
        
    # Instantiate theRandom Forest Classifier : gb
    gb = GradientBoostingClassifier()
    
    random_gb = RandomizedSearchCV(estimator=gb,
                        param_distributions=param_gb,
                        cv=10,
                        verbose=1,
                        n_jobs=14,
                        scoring=['accuracy','roc_auc'], refit = 'roc_auc')
    
    # Fit the model to the training data
    random_gb.fit(X_train, y_train)
    
    # Selegb the best model
    best_gb = random_gb.best_estimator_
    gb_best.append(best_gb)
    
        # Compute predigbed probabilities: y_pred_proba
    gb_pred_proba = pd.DataFrame(best_gb.predict_proba(X_val)[:,1])
    gb_pred_class = pd.DataFrame(best_gb.predict(X_val))
    gb_proba = pd.concat([gb_proba,gb_pred_proba], axis=1)
    gb_class = pd.concat([gb_class,gb_pred_class], axis=1)

   # Save the parameters
    gb_params.append(random_gb.best_params_) 
   
    # Compute scores
    gb_auc.append(roc_auc_score(y_val, gb_pred_proba))
    gb_accuracy.append(accuracy_score(y_val,gb_pred_class))
    
    # Print the optimal parameters and best score
    print("Tuned Gradient Boosting Classifier parameters: {}".format(random_gb.best_params_))
    print("Tuned Gradient Boosting Classifier Accuracy: {}".format(accuracy_score(y_val,gb_pred_class)))
    print("Tuned Gradient Boosting Classifier AUC: {}".format(roc_auc_score(y_val, gb_pred_proba)))
    
     #
    # # on peut récupérer directement l'importance ici pour random forest
    # gb_imp = best_gb.feature_importances_
    # gb_impMat.append(gb_imp)
    
    # par souci de comparabilité, on utilise ce mode de calcul
    gb_imp = permutation_importance(best_gb, X_val, y_val, scoring='roc_auc')
    gb_impMat.append(gb_imp.importances_mean)


#%%

#--------------------
# Gradient Boosting
#--------------------
# Compute mean probability prediction :

gb_proba_mean = gb_proba.mean(axis=1)
gb_class_mean = gb_class.mean(axis=1)
    
# Compute and print mean AUC score
gb_meanAccuracy = np.mean(gb_accuracy)
print("'\n' average Accuracy for the Gradient Boosting : {}".format(gb_meanAccuracy))

gb_meanAuc = np.mean(gb_auc) 
print("'\n' average AUC for the Gradient Boosting : {}".format(gb_meanAuc))

gb_scoreData = pd.DataFrame( {'Accuracy':[gb_meanAccuracy], 'ROC AUC':[gb_meanAuc]})
gb_impMean = np.mean(gb_impMat, axis=0)
gb_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':gb_impMean/gb_impMean.max()*100})

gb_result = gb_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Gradient Boosting  '\n'{}".format(gb_result))

#%% Evaluating Test Sample
best_gb = gb_best[gb_auc.index(max(gb_auc))]
gb_test_pred = best_gb.predict(X_test)
gb_test_pred_proba = best_gb.predict_proba(X_test)
gb_test_accuracy = accuracy_score(y_test, gb_test_pred)
gb_test_auc = roc_auc_score(y_test, gb_test_pred_proba[:,1])

gb_best_params = gb_params[gb_auc.index(max(gb_auc))]

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(gb_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(gb_test_auc))
print("'\n' Best Parameters :", gb_best_params)

#%% Saving model
model_path = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"

from joblib import dump, load
dump(best_gb, model_path+"/gradientboosting.joblib")

#%% Test model importing

test_model = load(model_path+"/gradientboosting.joblib")

test_predict = test_model.predict(X_test)
test_predict_proba = test_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, test_predict)
test_auc = roc_auc_score(y_test, test_predict_proba[:,1])

