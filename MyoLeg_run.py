# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 12:28:24 2025

@author: msi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:00:06 2024

@author: msi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:44:50 2024

@author: msi
"""
import math
import libemg 
import MTRLibEMG
import numpy as np
import pandas as pd
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics#
import matplotlib.pyplot as plt
from libemg.utils import get_windows
from scipy.signal import decimate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
mottype = 'WAK'
sf=200
ws=40
si=8
accdic={}
sddic={}
featsets=["HTD","TDPSD","TDAR","LS09"]
models=['LDA','SVM']
for q in models:
 if q=='KNN':
     params = {'n_neighbors': 5} # Optional
 elif q=='SVM':
     params = {'kernel':'rbf',
               'gamma':'scale'} # Optional
 elif q=='RF':
     params = {
    'n_estimators': 99,
    'max_depth': 20,
    'max_leaf_nodes': 10
    }
 elif q=='LDA':
     params={}
 elif q=='QDA':
     params={}
      
 classifier = q
 for k in featsets:
  accuracies = []
  for i in range(1,21):
   for j in range(1,6):
    if i < 10:
        data_path = f'D:/Dataset_v1.1/Sub0{i}/Data/Sub0{i}_{mottype}_0{j}.csv'
        labels_path = f'D:/Dataset_v1.1/Sub0{i}/Label/Sub0{i}_{mottype}_0{j}.csv'
    else:
        data_path = f'D:/Dataset_v1.1/Sub{i}/Data/Sub{i}_{mottype}_0{j}.csv'
        labels_path = f'D:/Dataset_v1.1/Sub{i}/Label/Sub{i}_{mottype}_0{j}.csv'
 
    if j==1:
     data = pd.read_csv(data_path)
     labels = pd.read_csv(labels_path)
    else:
        data = pd.concat([data,pd.read_csv(data_path)],ignore_index=True)
        labels = pd.concat([labels,pd.read_csv(labels_path)],ignore_index=True)
   emgcall = data.columns.str.contains('sEMG')
   semg = data.iloc[:, emgcall].to_numpy()
   fi = libemg.filtering.Filter(sf)
   fi.install_filters(filter_dictionary={ "name":"highpass", "cutoff": 10, "order": 2})
   fsemg = fi.filter(semg)
    
   gaitphase = labels.iloc[:, 1].to_numpy()
   trialno = labels.iloc[:, 2].to_numpy()
   nann = np.isnan(gaitphase)
   fsemg = fsemg[~nann]
   gaitphase = gaitphase[~nann]
   trialno = trialno[~nann]
    # gaitphase[gaitphase==3]=1
    # gaitphase[gaitphase==2]=1
    # gaitphase[gaitphase==4]=2
    # gaitphase[gaitphase==5]=2
   windows = get_windows(fsemg, ws, si)
   fe = FeatureExtractor()
   emgfeat=fe.extract_feature_group(k, windows)
    
   emgfeat,gaitphase_downsampled,trialno_downsampled=MTRLibEMG.DSOut(si,trialno,gaitphase,emgfeat)
   gaitphase_downsampled=gaitphase_downsampled.astype(int)
    # gaitphase_downsampled=gaitphase_downsampled-1
   [feattrain, classtrain, feattest, classtest]=MTRLibEMG.DataSplit(emgfeat, trialno_downsampled, gaitphase_downsampled, 5,b=None)
    
   feattrain,feattest=MTRLibEMG.Znorm(feattrain, feattest)
   data_set = {}
   data_set['training_features'] = feattrain
   data_set['training_labels'] = classtrain
   model = EMGClassifier(classifier)
   model.fit(data_set,None,params)
   preds, probs = model.run(feattest)
   evm=OfflineMetrics()
   accuracies.append(evm.get_CA(classtest, preds+1))
   
   [feattrain, classtrain, feattest, classtest]=MTRLibEMG.DataSplit(emgfeat, trialno_downsampled, gaitphase_downsampled, 4,b=None)
   
   feattrain,feattest=MTRLibEMG.Znorm(feattrain, feattest)
   
   data_set = {}
   data_set['training_features'] = feattrain
   data_set['training_labels'] = classtrain
   model.fit(data_set,None,params)
   preds, probs = model.run(feattest)
   accuracies.append(evm.get_CA(classtest, preds+1))
   
   [feattrain, classtrain, feattest, classtest]=MTRLibEMG.DataSplit(emgfeat, trialno_downsampled, gaitphase_downsampled, 3,b=None)
    
   feattrain,feattest=MTRLibEMG.Znorm(feattrain, feattest)
   
   data_set = {}
   data_set['training_features'] = feattrain
   data_set['training_labels'] = classtrain
   model.fit(data_set,None,params)
   preds, probs = model.run(feattest)
   accuracies.append(evm.get_CA(classtest, preds+1))
   
   [feattrain, classtrain, feattest, classtest]=MTRLibEMG.DataSplit(emgfeat, trialno_downsampled, gaitphase_downsampled, 2,b=None)
    
   feattrain,feattest=MTRLibEMG.Znorm(feattrain, feattest)
   
   data_set = {}
   data_set['training_features'] = feattrain
   data_set['training_labels'] = classtrain
   model.fit(data_set,None,params)
   preds, probs = model.run(feattest)
   accuracies.append(evm.get_CA(classtest, preds+1))
    
   [feattrain, classtrain, feattest, classtest]=MTRLibEMG.DataSplit(emgfeat, trialno_downsampled, gaitphase_downsampled, 1,b=None)
    
   feattrain,feattest=MTRLibEMG.Znorm(feattrain, feattest)
   
   data_set = {}
   data_set['training_features'] = feattrain
   data_set['training_labels'] = classtrain
   model.fit(data_set,None,params)
   preds, probs = model.run(feattest)
   accuracies.append(evm.get_CA(classtest, preds+1))
    
  accdic[q,k]=np.mean(accuracies)  
  sddic[q,k]=np.std(accuracies,ddof=1) 
    
    
    
    