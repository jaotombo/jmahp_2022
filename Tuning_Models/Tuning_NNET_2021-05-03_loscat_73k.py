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

# initialize nnet
nnet_auc = []
nnet_accuracy = []
nnet_impMat = []
nnet_proba = pd.DataFrame({'A' : []})
nnet_class = pd.DataFrame({'A' : []})
score = pd.DataFrame({'A' : []})
nnet_best = []
nnet_params = []

for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True, stratify=y_train_full)

#------------------------
# Shallow Neural Network
#------------------------

    # Import the required module for neural networks
    import tensorflow as tf
    from tensorflow import keras
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from functools import partial
    
    # Define  the Neural Network Model :
       
    def create_model(nl=1,nn=300):
        model = Sequential()
        model.add(Dense(nn, input_shape=(X_train.shape[1],), activation='relu'))
        # Add as many hidden layers as specified in nl
        for i in range(nl):
            # Layers have nn neurons
            model.add(Dense(nn, activation='relu'))
        # End defining and compiling your model...
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(),'accuracy'])
        return model
    
    # Create a model as a sklearn estimator
    model = KerasClassifier(build_fn=create_model)
    
    # Define parameters, named just like in create_model()
    param_nnet = dict(nl=[3, 4, 5], nn=[100, 200, 300])

    # Create a random search cv object and fit it to the data
    random_nnet = RandomizedSearchCV(estimator = model, 
                                param_distributions=param_nnet,
                                cv=3,
                                scoring=['accuracy','roc_auc'], refit = 'roc_auc')
        
   # Fit the model to the training data
    checkpoint_cb = ModelCheckpoint("Rehosp_NNET_model.h5", save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
   
    random_nnet.fit(X_train, y_train, epochs=20, 
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint_cb, early_stopping_cb])
   
    # Select the best model
    best_nnet = random_nnet.best_estimator_
    nnet_best.append(best_nnet)
      
    # Compute predinneted probabilities: y_pred_proba
    nnet_pred_proba = pd.DataFrame(best_nnet.predict_proba(X_val)[:,1])
    nnet_pred_class = pd.DataFrame(best_nnet.predict(X_val))
    nnet_proba = pd.concat([nnet_proba,nnet_pred_proba], axis=1)
    nnet_class = pd.concat([nnet_class,nnet_pred_class], axis=1)

    # Collect the best parameters
    nnet_params.append(random_nnet.best_params_)

    # Compute scores
    nnet_accuracy.append(accuracy_score(y_val, nnet_pred_class))
    nnet_auc.append(roc_auc_score(y_val, nnet_pred_proba))


   # Print the optimal parameters and best score
    print("Tuned Neural Network Classifier parameters: {}".format(random_nnet.best_params_))
    print("Tuned Neural Network Classifier Accuracy: {}".format(accuracy_score(y_val, nnet_pred_class))) 
    print("Tuned Neural Network Classifier AUC: {}".format(roc_auc_score(y_val, nnet_pred_proba)))
    
    nnet_imp = permutation_importance(best_nnet, X_val, y_val, scoring='roc_auc')
    nnet_impMat.append(nnet_imp.importances_mean)
    

#%% 
#==============================================================================
# OUTPUTS
#==============================================================================

#-----------------------
# Shallow Neural Network
#-----------------------
# Compute mean probability prediction :

nnet_proba_mean = nnet_proba.mean(axis=1)
nnet_class_mean = nnet_class.mean(axis=1)
    
# Compute and print mean AUC score
nnet_meanAccuracy = np.mean(nnet_accuracy)
print("'\n' average Accuracy for the Neural Network: {:.3f}".format(nnet_meanAccuracy))

nnet_meanAuc = np.mean(nnet_auc) 
print("'\n' average AUC for the NeuralNetwork: {:.3f}".format(nnet_meanAuc))

nnet_scoreData = pd.DataFrame( {'Accuracy':[nnet_meanAccuracy], 'ROC AUC':[nnet_meanAuc]})
nnet_impMean = np.mean(nnet_impMat, axis=0)
nnet_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':nnet_impMean/nnet_impMean.max()*100})

nnet_result = nnet_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Neural Network '\n'{}".format(nnet_result))

#%% Evaluating Test Sample
best_nnet = nnet_best[nnet_auc.index(max(nnet_auc))]
nnet_test_pred = best_nnet.predict(X_test)
nnet_test_pred_proba = best_nnet.predict_proba(X_test)
nnet_test_accuracy = accuracy_score(y_test, nnet_test_pred)
nnet_test_auc = roc_auc_score(y_test, nnet_test_pred_proba[:,1])

nnet_best_params = nnet_params[nnet_auc.index(max(nnet_auc))]

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(nnet_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(nnet_test_auc))
print("'\n' Best Parameters :", nnet_best_params)

#%% Saving parameters
model_path = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"

np.save(model_path+"/nnet_params.npy", nnet_best_params)


#%% Test model importing

# test_dict = np.load(model_path+"/nnet_params.npy",allow_pickle='TRUE').item()

# def create_model(nl=test_dict['nl'],nn=test_dict['nn']):
#     model = Sequential()
#     model.add(Dense(nn, input_shape=(X_train.shape[1],), activation='relu'))
#     # Add as many hidden layers as specified in nl
#     for i in range(nl):
#         # Layers have nn neurons
#         model.add(Dense(nn, activation='relu'))
#     # End defining and compiling your model...
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(),'accuracy'])
#     return model


# # Create a model as a sklearn estimator
# test_model = KerasClassifier(build_fn=create_model, batch_size=8)


# test_model.fit(X_train, y_train, epochs=10, 
#                 validation_data=(X_val, y_val),
#                 callbacks=[checkpoint_cb, early_stopping_cb])

# test_predict = test_model.predict(X_test)
# test_predict_proba = test_model.predict_proba(X_test)
# test_accuracy = accuracy_score(y_test, test_predict)
# test_auc = roc_auc_score(y_test, test_predict_proba[:,1])
