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
from sklearn.metrics import roc_curve


# Import the required module for neural networks
import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

#%%

DATA_PATH = "F:/AP-HM/Rehospitalisation/Data"
IMAGES_PATH = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Images"
MODEL_PATH = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"
RESULT_PATH = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Results/loscat_2021-05-04" 
REPEAT = 100

#%%
def plot_roc_curve(fpr, tpr, color="b", label=None):
    plt.plot(fpr, tpr, linewidth=2, color=color, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

#%%
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
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

#%% Initialize lists and dataframes

#================================================
# Moving into modeling with different classifiers
#================================================

# initialize logistic regression
lr_auc = []
lr_accuracy = []
lr_impMat = []
lr_fpr = pd.DataFrame({'A' : []})
lr_tpr =pd.DataFrame({'A' : []})
lr_thres = pd.DataFrame({'A' : []})

# initialize elasticnet
en_auc = []
en_accuracy = []
en_impMat = []
en_fpr = pd.DataFrame({'A' : []})
en_tpr =pd.DataFrame({'A' : []})
en_thres = pd.DataFrame({'A' : []})

# initialize CART
ct_auc = []
ct_accuracy = []
ct_impMat = []
ct_fpr = pd.DataFrame({'A' : []})
ct_tpr =pd.DataFrame({'A' : []})
ct_thres = pd.DataFrame({'A' : []})

# initialize Random Forest
rf_auc = []
rf_accuracy = []
rf_impMat = []
rf_fpr = pd.DataFrame({'A' : []})
rf_tpr =pd.DataFrame({'A' : []})
rf_thres = pd.DataFrame({'A' : []})

# initialize Gradient Boosting
gb_auc = []
gb_accuracy = []
gb_impMat = []
gb_fpr = pd.DataFrame({'A' : []})
gb_tpr =pd.DataFrame({'A' : []})
gb_thres = pd.DataFrame({'A' : []})

# initialize nnet
nnet_auc = []
nnet_accuracy = []
nnet_impMat = []
nnet_score = pd.DataFrame({'A' : []})
nnet_fpr = pd.DataFrame({'A' : []})
nnet_tpr =pd.DataFrame({'A' : []})
nnet_thres = pd.DataFrame({'A' : []})


#%%

import time
start = time.perf_counter()

for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True, stratify=y_train_full)

#--------------------
# Logistic Regression
#--------------------
    start_lr = time.perf_counter()
    # Import the Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    
    # Instantiate the Logistic Regression Classifier : lr
    lr = LogisticRegression(penalty='none', max_iter=200)
      
    # Fit the model to the training data
    lr.fit(X_train, y_train)   

    # Compute predicted probabilities: y_pred_proba
    lr_pred_proba = pd.DataFrame(lr.predict_proba(X_val)[:,1])
    lr_pred_class = pd.DataFrame(lr.predict(X_val))
    fpr, tpr, thres = roc_curve(y_val, lr_pred_proba)
    
    # Compute true positive and false positive rate
    lr_fpr = pd.concat([pd.DataFrame(fpr)], axis=1)
    lr_tpr = pd.concat([pd.DataFrame(tpr)], axis=1)
    lr_thres = pd.concat([pd.DataFrame(thres)], axis=1)

    
    # Compute scores
    lr_auc.append(roc_auc_score(y_val, lr_pred_proba))
    lr_accuracy.append(accuracy_score(y_val, lr_pred_class))
    
    lr_imp = permutation_importance(lr, X_val, y_val, scoring='roc_auc', n_jobs=14)
    lr_impMat.append(lr_imp.importances_mean)

    end_lr = time.clock()
    print("\n elapsed time for Logistic Regression: {:.3f}\n".format(end_lr-start_lr))   

#--------------------
# Elasticnet
#--------------------
    start_elnet = time.perf_counter()   
    # Instantiate the Penalized Logistic Regression Classifier : en
    best_en = LogisticRegression(C= 0.21544346900318845, penalty= 'l2')

   # Fit the model to the training data
    best_en.fit(X_train, y_train)   
    
    # Compute predicted probabilities: y_pred_proba
    en_pred_proba = pd.DataFrame(best_en.predict_proba(X_val)[:,1])
    en_pred_class = pd.DataFrame(best_en.predict(X_val))
    fpr, tpr, thres = roc_curve(y_val, en_pred_proba)
    
    # Compute true positive and false positive rate
    en_fpr = pd.concat([pd.DataFrame(fpr)], axis=1)
    en_tpr = pd.concat([pd.DataFrame(tpr)], axis=1)
    en_thres = pd.concat([pd.DataFrame(thres)], axis=1)

    
    # Compute scores
    en_accuracy.append(accuracy_score(y_val, en_pred_class))
    en_auc.append(roc_auc_score(y_val, en_pred_proba))
    
    en_imp = permutation_importance(best_en, X_val, y_val, scoring='roc_auc', n_jobs=14)
    en_impMat.append(en_imp.importances_mean)

    end_elnet = time.clock()
    print("\n elapsed time for Elaticnet Regression: {:.3f}\n".format(end_elnet-start_elnet))   

#--------------------
# CART
#--------------------
    start_cart = time.perf_counter()
    # Import the Decision Tree Model
    from sklearn.tree import DecisionTreeClassifier
    
    # Instantiate the Decision Tree Model : ct
    best_ct = DecisionTreeClassifier(min_samples_leaf= 43, min_samples_split= 2)
 

   # Fit the model to the training data
    best_ct.fit(X_train, y_train)  
     
    # Compute predicted probabilities: y_pred_proba
    ct_pred_proba = pd.DataFrame(best_ct.predict_proba(X_val)[:,1])
    ct_pred_class = pd.DataFrame(best_ct.predict(X_val))
    fpr, tpr, thres = roc_curve(y_val, ct_pred_proba)
    
    # Compute true positive and false positive rate
    ct_fpr = pd.concat([pd.DataFrame(fpr)], axis=1)
    ct_tpr = pd.concat([pd.DataFrame(tpr)], axis=1)
    ct_thres = pd.concat([pd.DataFrame(thres)], axis=1)

    
    # Compute scores
    ct_auc.append(roc_auc_score(y_val, ct_pred_proba))
    ct_accuracy.append(accuracy_score(y_val, ct_pred_class))
    
    ct_imp = permutation_importance(best_ct, X_val, y_val, scoring='roc_auc', n_jobs=14)
    ct_impMat.append(ct_imp.importances_mean)

    end_cart = time.clock()
    print("\n elapsed time for CART Regression: {:.3f}\n".format(end_cart-start_cart))   

#--------------------
# Random Forest
#--------------------
    start_rf = time.perf_counter()    
    # Import the Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    
    # Instantiate the Random Forest Classifier
    best_rf = RandomForestClassifier(n_estimators= 350, min_samples_leaf= 10, max_features= 'auto', max_depth= None)
 

   # Fit the model to the training data
    best_rf.fit(X_train, y_train)  
 
    # Compute predicted probabilities: y_pred_proba
    rf_pred_proba = pd.DataFrame(best_rf.predict_proba(X_val)[:,1])
    rf_pred_class = pd.DataFrame(best_rf.predict(X_val))
    fpr, tpr, thres = roc_curve(y_val, rf_pred_proba)
    
    # Compute true positive and false positive rate
    rf_fpr = pd.concat([pd.DataFrame(fpr)], axis=1)
    rf_tpr = pd.concat([pd.DataFrame(tpr)], axis=1)
    rf_thres = pd.concat([pd.DataFrame(thres)], axis=1)

    
    # Compute scores
    rf_auc.append(roc_auc_score(y_val, rf_pred_proba))
    rf_accuracy.append(accuracy_score(y_val, rf_pred_class))
    
    #
    # # on peut récupérer directement l'importance ici pour random forest
    # rf_imp = best_rf.feature_importances_
    # rf_impMat.append(rf_imp)
    
    rf_imp = permutation_importance(best_rf, X_val, y_val, scoring='roc_auc', n_jobs=14)
    rf_impMat.append(rf_imp.importances_mean)

    end_rf = time.clock()
    print("\n elapsed time for Random Forest Regression: {:.3f}\n".format(end_rf-start_rf))   

#--------------------
# Gradient Boosting
#--------------------
    start_gb = time.perf_counter()
    
    # Import the Gradient Boosting Classifier
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Instantiate the Gradient Boosting Classifier
    best_gb = GradientBoostingClassifier(subsample= 0.9, n_estimators= 300, max_features= 'sqrt', max_depth= 4)


   # Fit the model to the training data
    best_gb.fit(X_train, y_train)  
   
     # Compute predicted probabilities: y_pred_proba
    gb_pred_proba = pd.DataFrame(best_gb.predict_proba(X_val)[:,1])
    gb_pred_class = pd.DataFrame(best_gb.predict(X_val))
    fpr, tpr, thres = roc_curve(y_val, gb_pred_proba)
    
    # Compute true positive and false positive rate
    gb_fpr = pd.concat([pd.DataFrame(fpr)], axis=1)
    gb_tpr = pd.concat([pd.DataFrame(tpr)], axis=1)
    gb_thres = pd.concat([pd.DataFrame(thres)], axis=1)

    
    # Compute scores
    gb_auc.append(roc_auc_score(y_val, gb_pred_proba))
    gb_accuracy.append(accuracy_score(y_val, gb_pred_class))

    #
     # # on peut récupérer directement l'importance ici pour random forest
    # gb_imp = best_gb.feature_importances_
    # gb_impMat.append(gb_imp)
   
    gb_imp = permutation_importance(best_gb, X_val, y_val, scoring='roc_auc', n_jobs=14)
    gb_impMat.append(gb_imp.importances_mean)

    end_gb = time.clock()
    print("\n elapsed time for Gradient Boosting Regression: {:.3f}\n".format(end_gb-start_gb))   

#------------------------
# Shallow Neural Network
#------------------------
    start_nnet = time.perf_counter() 
   # Define the Neural Network Classifier
    nnet_dict = {"nl":3, "nn":100}
    
    def create_model(nl=nnet_dict['nl'],nn=nnet_dict['nn']):
        model = Sequential()
        model.add(Dense(nn, input_shape=(X_train_full.shape[1],), activation='relu'))
        # Add as many hidden layers as specified in nl
        for i in range(nl):
            # Layers have nn neurons
            model.add(Dense(nn, activation='relu'))
        # End defining and compiling your model...
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(),'accuracy'])
        return model
    
    keras_nnet = KerasClassifier(build_fn=create_model)

    # Fit the model to the training data
    checkpoint_cb = ModelCheckpoint("Rehosp_NNET_model.h5", save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=20, restore_best_weights=True)
     
    history = keras_nnet.fit(X_train, y_train, epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint_cb, early_stopping_cb])
    

     
    # Compute predicted probabilities: y_pred_proba

    nnet_pred_proba = pd.DataFrame(keras_nnet.predict_proba(X_val)[:,1])   
    nnet_pred_class = pd.DataFrame(keras_nnet.predict(X_val))
    fpr, tpr, thres = roc_curve(y_val, nnet_pred_proba)
    
    # Compute true positive and false positive rate
    nnet_fpr = pd.concat([pd.DataFrame(fpr)], axis=1)
    nnet_tpr = pd.concat([pd.DataFrame(tpr)], axis=1)
    nnet_thres = pd.concat([pd.DataFrame(thres)], axis=1)
    
    # Compute scores
    nnet_auc.append(roc_auc_score(y_val, nnet_pred_proba))
    nnet_accuracy.append(accuracy_score(y_val, nnet_pred_class))

    # Compute importance    
  
    nnet_imp = permutation_importance(keras_nnet, X_val, y_val, scoring='roc_auc')
    nnet_impMat.append(nnet_imp.importances_mean)

    end_nnet = time.clock()
    print("\n elapsed time for Neural Network Regression: {:.3f}\n".format(end_nnet-start_nnet))   

end = time.perf_counter()
print("\n elapsed time: {:.3f}\n".format(end-start))

#%% 
#==============================================================================
# OUTPUTS
#==============================================================================

#--------------------
# Logistic Regression
#--------------------
# Compute mean true positive and false positive rate :

lr_fpr_mean = lr_fpr.mean(axis=1)
lr_tpr_mean = lr_tpr.mean(axis=1)
lr_thres_mean = lr_thres.mean(axis=1)

lr_roc_mean = pd.DataFrame({"fpr":lr_fpr_mean, "tpr":lr_tpr_mean, "threshold":lr_thres_mean})

lr_accuracy_data = pd.DataFrame({'Accuracy':lr_accuracy})
lr_auc_data = pd.DataFrame({'AUC':lr_auc})
   
# Compute and print mean AUC score
lr_meanAccuracy = np.mean(lr_accuracy)
print("'\n' average Accuracy for the Logistic Regression: {:.3f}".format(lr_meanAccuracy))
lr_meanAuc = np.mean(lr_auc) 
print("'\n' average AUC for the Logistic Regression: {:.3f}".format(lr_meanAuc))

lr_impMean = np.mean(lr_impMat, axis=0)
lr_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':lr_impMean/lr_impMean.max()*100})

lr_result = lr_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Logistic Regression '\n'{}".format(lr_result))

#%% Evaluating Test Sample

lr_test_pred = lr.predict(X_test)
lr_test_pred_proba = lr.predict_proba(X_test)
lr_test_accuracy = accuracy_score(y_test, lr_test_pred)
lr_test_auc = roc_auc_score(y_test, lr_test_pred_proba[:,1])

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(lr_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(lr_test_auc))
print("======================================================================'\n")

lr_scoreData = pd.DataFrame( {'Mean Accuracy':[lr_meanAccuracy], 'Mean AUC':[lr_meanAuc],
                              'Test Accuracy':[lr_test_accuracy], 'Test AUC':[lr_test_auc]})

# Export results
lr_roc_mean.to_excel(RESULT_PATH+"/lr_roc_mean.xlsx")
lr_scoreData.to_excel(RESULT_PATH+"/lr_perf_mean.xlsx")
lr_result.to_excel(RESULT_PATH+"/lr_result.xlsx")
lr_accuracy_data.to_excel(RESULT_PATH+"/lr_accuracy.xlsx")
lr_auc_data.to_excel(RESULT_PATH+"/lr_auc.xlsx")

#%%

#--------------------
# Elasticnet
#--------------------
# Compute mean true positive and false positive rate :

en_fpr_mean = en_fpr.mean(axis=1)
en_tpr_mean = en_tpr.mean(axis=1)
en_thres_mean = en_thres.mean(axis=1)

en_roc_mean = pd.DataFrame({"fpr":en_fpr_mean, "tpr":en_tpr_mean, "threshold":en_thres_mean})

en_accuracy_data = pd.DataFrame({'Accuracy':en_accuracy})
en_auc_data = pd.DataFrame({'AUC':en_auc})
    
# Compute and print mean AUC score
en_meanAccuracy = np.mean(en_accuracy)
print("'\n' average Accuracy for the Elasticnet: {:.3f}".format(en_meanAccuracy))
en_meanAuc = np.mean(en_auc) 
print("'\n' average AUC for the Elasticnet: {:.3f}".format(en_meanAuc))

en_impMean = np.mean(en_impMat, axis=0)
en_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':en_impMean/en_impMean.max()*100})

en_result = en_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Elasticnet'\n'{}".format(en_result))

#%% Evaluating Test Sample

en_test_pred = best_en.predict(X_test)
en_test_pred_proba = best_en.predict_proba(X_test)
en_test_accuracy = accuracy_score(y_test, en_test_pred)
en_test_auc = roc_auc_score(y_test, en_test_pred_proba[:,1])

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(en_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(en_test_auc))
print("======================================================================'\n")

en_scoreData = pd.DataFrame( {'Mean Accuracy':[en_meanAccuracy], 'Mean AUC':[en_meanAuc],
                              'Test Accuracy':[en_test_accuracy], 'Test AUC':[en_test_auc]})

# Export results
en_roc_mean.to_excel(RESULT_PATH+"/en_roc_mean.xlsx")
en_scoreData.to_excel(RESULT_PATH+"/en_perf_mean.xlsx")
en_result.to_excel(RESULT_PATH+"/en_result.xlsx")
en_accuracy_data.to_excel(RESULT_PATH+"/en_accuracy.xlsx")
en_auc_data.to_excel(RESULT_PATH+"/en_auc.xlsx")

#%%

#--------------------
# Decision Tree
#--------------------
# Compute mean true positive and false positive rate :

ct_fpr_mean = ct_fpr.mean(axis=1)
ct_tpr_mean = ct_tpr.mean(axis=1)
ct_thres_mean = ct_thres.mean(axis=1)

ct_roc_mean = pd.DataFrame({"fpr":ct_fpr_mean, "tpr":ct_tpr_mean, "threshold":ct_thres_mean})

ct_accuracy_data = pd.DataFrame({'Accuracy':ct_accuracy})
ct_auc_data = pd.DataFrame({'AUC':ct_auc})
    
# Compute and print mean AUC score
ct_meanAccuracy = np.mean(ct_accuracy)
print("'\n' average Accuracy for the Decision Tree: {:.3f}".format(ct_meanAccuracy))
ct_meanAuc = np.mean(ct_auc) 
print("'\n' average AUC for the Decision Tree: {:.3f}".format(ct_meanAuc))

ct_impMean = np.mean(ct_impMat, axis=0)
ct_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':ct_impMean/ct_impMean.max()*100})

ct_result = ct_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Decision Tree '\n'{}".format(ct_result))

#%% Evaluating Test Sample

ct_test_pred = best_ct.predict(X_test)
ct_test_pred_proba = best_ct.predict_proba(X_test)
ct_test_accuracy = accuracy_score(y_test, ct_test_pred)
ct_test_auc = roc_auc_score(y_test, ct_test_pred_proba[:,1])

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(ct_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(ct_test_auc))
print("======================================================================'\n")

ct_scoreData = pd.DataFrame( {'Mean Accuracy':[ct_meanAccuracy], 'Mean AUC':[ct_meanAuc],
                              'Test Accuracy':[ct_test_accuracy], 'Test AUC':[ct_test_auc]})

# Export results
ct_roc_mean.to_excel(RESULT_PATH+"/ct_roc_mean.xlsx")
ct_scoreData.to_excel(RESULT_PATH+"/ct_perf_mean.xlsx")
ct_result.to_excel(RESULT_PATH+"/ct_result.xlsx")
ct_accuracy_data.to_excel(RESULT_PATH+"/ct_accuracy.xlsx")
ct_auc_data.to_excel(RESULT_PATH+"/ct_auc.xlsx")

#%%

#--------------------
# Random Forest
#--------------------
# Compute mean true positive and false positive rate :

rf_fpr_mean = rf_fpr.mean(axis=1)
rf_tpr_mean = rf_tpr.mean(axis=1)
rf_thres_mean = rf_thres.mean(axis=1)

rf_roc_mean = pd.DataFrame({"fpr":rf_fpr_mean, "tpr":rf_tpr_mean, "threshold":rf_thres_mean})

rf_accuracy_data = pd.DataFrame({'Accuracy':rf_accuracy})
rf_auc_data = pd.DataFrame({'AUC':rf_auc})
    
# Compute and print mean AUC score
rf_meanAccuracy = np.mean(rf_accuracy)
print("'\n' average Accuracy for the Random Forest: {:.3f}".format(rf_meanAccuracy))
rf_meanAuc = np.mean(rf_auc) 
print("'\n' average AUC for the Random Forest: {:.3f}".format(rf_meanAuc))

rf_impMean = np.mean(rf_impMat, axis=0)
rf_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':rf_impMean/rf_impMean.max()*100})

rf_result = rf_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Random Forest '\n'{}".format(rf_result))

#%% Evaluating Test Sample

rf_test_pred = best_rf.predict(X_test)
rf_test_pred_proba = best_rf.predict_proba(X_test)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
rf_test_auc = roc_auc_score(y_test, rf_test_pred_proba[:,1])

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(rf_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(rf_test_auc))
print("======================================================================'\n")

rf_scoreData = pd.DataFrame( {'Mean Accuracy':[rf_meanAccuracy], 'Mean AUC':[rf_meanAuc],
                              'Test Accuracy':[rf_test_accuracy], 'Test AUC':[rf_test_auc]})

# Export results
rf_roc_mean.to_excel(RESULT_PATH+"/rf_roc_mean.xlsx")
rf_scoreData.to_excel(RESULT_PATH+"/rf_perf_mean.xlsx")
rf_result.to_excel(RESULT_PATH+"/rf_result.xlsx")
rf_accuracy_data.to_excel(RESULT_PATH+"/rf_accuracy.xlsx")
rf_auc_data.to_excel(RESULT_PATH+"/rf_auc.xlsx")

#%%

#--------------------
# Gradient Boosting
#--------------------
# Compute mean true positive and false positive rate :

gb_fpr_mean = gb_fpr.mean(axis=1)
gb_tpr_mean = gb_tpr.mean(axis=1)
gb_thres_mean = gb_thres.mean(axis=1)

gb_roc_mean = pd.DataFrame({"fpr":gb_fpr_mean, "tpr":gb_tpr_mean, "threshold":gb_thres_mean})

gb_accuracy_data = pd.DataFrame({'Accuracy':gb_accuracy})
gb_auc_data = pd.DataFrame({'AUC':gb_auc})
    
# Compute and print mean AUC score
gb_meanAccuracy = np.mean(gb_accuracy)
print("'\n' average Accuracy for the Gradient Boosting: {:.3f}".format(gb_meanAccuracy))
gb_meanAuc = np.mean(gb_auc) 
print("'\n' average AUC for the Gradient Boosting: {:.3f}".format(gb_meanAuc))

gb_impMean = np.mean(gb_impMat, axis=0)
gb_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':gb_impMean/gb_impMean.max()*100})

gb_result = gb_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Gradient Boosting '\n'{}".format(gb_result))

#%% Evaluating Test Sample

gb_test_pred = best_gb.predict(X_test)
gb_test_pred_proba = best_gb.predict_proba(X_test)
gb_test_accuracy = accuracy_score(y_test, gb_test_pred)
gb_test_auc = roc_auc_score(y_test, gb_test_pred_proba[:,1])

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(gb_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(gb_test_auc))
print("======================================================================'\n")

gb_scoreData = pd.DataFrame( {'Mean Accuracy':[gb_meanAccuracy], 'Mean AUC':[gb_meanAuc],
                              'Test Accuracy':[gb_test_accuracy], 'Test AUC':[gb_test_auc]})

# Export results
gb_roc_mean.to_excel(RESULT_PATH+"/gb_roc_mean.xlsx")
gb_scoreData.to_excel(RESULT_PATH+"/gb_perf_mean.xlsx")
gb_result.to_excel(RESULT_PATH+"/gb_result.xlsx")
gb_accuracy_data.to_excel(RESULT_PATH+"/gb_accuracy.xlsx")
gb_auc_data.to_excel(RESULT_PATH+"/gb_auc.xlsx")

#%%
#-----------------------
# Shallow Neural Net
#-----------------------

# Compute mean true positive and false positive rate :

nnet_fpr_mean = nnet_fpr.mean(axis=1)
nnet_tpr_mean = nnet_tpr.mean(axis=1)
nnet_thres_mean = nnet_thres.mean(axis=1)

nnet_roc_mean = pd.DataFrame({"fpr":nnet_fpr_mean, "tpr":nnet_tpr_mean, "threshold":nnet_thres_mean})

nnet_accuracy_data = pd.DataFrame({'Accuracy':nnet_accuracy})
nnet_auc_data = pd.DataFrame({'AUC':nnet_auc})
    
# Compute and print mean AUC score
nnet_meanAccuracy = np.mean(nnet_accuracy)
print("'\n' average Accuracy for the  Shallow Neural Net: {:.3f}".format(nnet_meanAccuracy))
nnet_meanAuc = np.mean(nnet_auc) 
print("'\n' average AUC for the  Shallow Neural Net: {:.3f}".format(nnet_meanAuc))

nnet_impMean = np.mean(nnet_impMat, axis=0)
nnet_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':nnet_impMean/nnet_impMean.max()*100})

nnet_result = nnet_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the  Shallow Neural Net '\n'{}".format(nnet_result))

#%% Evaluating Test Sample

nnet_test_pred = keras_nnet.predict(X_test)
nnet_test_pred_proba = keras_nnet.predict_proba(X_test)[:,1]
nnet_test_accuracy = accuracy_score(y_test, nnet_test_pred)
nnet_test_auc = roc_auc_score(y_test, nnet_test_pred_proba)

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(nnet_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(nnet_test_auc))
print("======================================================================'\n")

nnet_scoreData = pd.DataFrame( {'Mean Accuracy':[nnet_meanAccuracy], 'Mean AUC':[nnet_meanAuc],
                              'Test Accuracy':[nnet_test_accuracy], 'Test AUC':[nnet_test_auc]})

#
# Export results
nnet_roc_mean.to_excel(RESULT_PATH+"/nnet_roc_mean.xlsx")
nnet_scoreData.to_excel(RESULT_PATH+"/nnet_perf_mean.xlsx")
nnet_result.to_excel(RESULT_PATH+"/nnet_result.xlsx")
nnet_accuracy_data.to_excel(RESULT_PATH+"/nnet_accuracy.xlsx")
nnet_auc_data.to_excel(RESULT_PATH+"/nnet_auc.xlsx")

#%% ==========================================================================
# Comparing performance of classifiers
#=============================================================================
#%% Concatenated data table

all_accuracy = pd.concat([lr_accuracy_data, en_accuracy_data, ct_accuracy_data,
                         rf_accuracy_data, gb_accuracy_data, nnet_accuracy_data], axis=1)
all_accuracy.columns =  ["LR","EN","CT","RF","GB","NNET"]

all_auc = pd.concat([lr_auc_data, en_auc_data, ct_auc_data,
                         rf_auc_data, gb_auc_data, nnet_auc_data], axis=1)
all_auc.columns =  ["LR","EN","CT","RF","GB","NNET"]

#%% Export concatenated data tables

all_accuracy.to_excel(RESULT_PATH+"/all_accuracy.xlsx")

all_auc.to_excel(RESULT_PATH+"/all_auc.xlsx")

#%% Compare means (independent t_tests)

def ind_ttest(data) :

    col_names = ["LR","EN","CT","RF","GB","NNET"]
    ttest_stats = []
    ttest_pvalue = []
    effsize=[]
    row_names=[]
    
    for i in range(len(col_names)) :
        for j in range(i+1, len(col_names)):
            ttest_stats.append(stats.ttest_ind(data.iloc[:,i],data.iloc[:,j])[0])
            ttest_pvalue.append(stats.ttest_ind(data.iloc[:,i],data.iloc[:,j])[1])
            effsize.append(pg.compute_effsize(data.iloc[:,i],data.iloc[:,j], eftype='cohen'))
            row_names.append([col_names[i],col_names[j]])
    
          
    ttest_data = pd.DataFrame({'Comparison':row_names,'Statistics':ttest_stats, 'p value':ttest_pvalue, 'effect size':effsize})
    return ttest_data

#%%

ind_accuracy = ind_ttest(all_accuracy)
ind_auc = ind_ttest(all_auc)

#%% Compare means (paired t_tests)

def paired_ttest(data) :

    col_names = ["LR","EN","CT","RF","GB","NNET"]
    ttest_stats = []
    ttest_pvalue = []
    effsize=[]
    row_names=[]
    
    for i in range(len(col_names)) :
        for j in range(i+1, len(col_names)):
            ttest_stats.append(stats.ttest_rel(data.iloc[:,i],data.iloc[:,j])[0])
            ttest_pvalue.append(stats.ttest_rel(data.iloc[:,i],data.iloc[:,j])[1])
            effsize.append(pg.compute_effsize(data.iloc[:,i],data.iloc[:,j], paired=True, eftype='cohen'))            
            row_names.append([col_names[i],col_names[j]])
    
          
    ttest_data = pd.DataFrame({'Comparison':row_names,'Statistics':ttest_stats, 'p value':ttest_pvalue, 'effect size':effsize})
    return ttest_data

#%%

paired_accuracy = paired_ttest(all_accuracy)
paired_accuracy.to_excel(RESULT_PATH+'/paired_accuracy.xlsx')

paired_auc = paired_ttest(all_auc)
paired_auc.to_excel(RESULT_PATH+'/paired_auc.xlsx')

#%%

# #=================
# # Plot ROC Curves
# #=================

# #%%
# plot_roc_curve(lr_fpr_mean, lr_tpr_mean, "b", "LR")
# plot_roc_curve(nnet_fpr_mean, nnet_tpr_mean, "r", "NNET")
# save_fig("roc_curve_plot")
# plt.show()
