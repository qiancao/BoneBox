#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:18:05 2022

https://thatascience.com/learn-machine-learning/pipeline-in-scikit-learn/

@author: qian.cao
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append("../bonebox/metrics/")

from FeaturesRadiomics import *

import matplotlib.pyplot as plt

if __name__ == "__main__":

    outDir = "/gpfs_projects/qian.cao/BoneBox-out/test_20220422_bin_parallel/"
    
    featureNames = getRadiomicFeatureNames() # TODO: save and read from file
    features = np.load(outDir+"featuresArray.npy")
    
    fem_dir = "../data/"
    roi_vm_mean = np.load(fem_dir+"roi_vm_mean.npy")
    
    # Training and testing scores
    # scoreTrain = np.zeros((features.shape[2],features.shape[3]))
    # scoreTest = np.zeros((features.shape[2],features.shape[3]))
    r2Test = np.zeros((features.shape[2],features.shape[3]))
    
    for cind in range(features.shape[2]): # imaging condition
        for sind in range(features.shape[3]): # seed, instance
            
            # feature
            feat = features[:,:,cind,sind]
            
            # data and target
            X = feat
            y = roi_vm_mean
            
            # Splitting data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 3, shuffle=False)
            
            # Random forest Tree Regression Pipeline
            rf_pipe = Pipeline([('scl', StandardScaler()),
                                ('clf',RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=0))])
            
            rf_pipe.fit(X_train, y_train)
            y_pred = rf_pipe.predict(X_test)
            
            # scoreTest[cind,sind] = rf_pipe.score(y_pred, y_test)
            r2Test[cind,sind] = np.corrcoef(y_pred, y_test)[0,1]**2
            
            importances = forest.feature_importances_
            
            
            # Correlation plot
            plt.figure()
            plt.plot(y_test, y_pred,'b.')
            plt.plot(*(np.linspace(0,np.max(y)),)*2,'k--')
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.xlim([0,np.max(y)])
            plt.ylim([0,np.max(y)])
            plt.savefig(outDir+f"correlation_{cind}.png")
            
            
        fp = features[:,:,cind,0]
        fp -= np.mean(fp,axis=0) # center on mean
        fp /= np.std(fp,axis=0) # scale to standard deviation
            
        plt.figure()
        plt.imshow(fp,cmap="plasma",vmin=-1,vmax=1)
        plt.savefig(outDir+f"features_{cind}.png")
        plt.xlabel("Features")
        plt.ylabel("ROIs")
        plt.axis("tight")
        plt.close("all")

    plt.figure()
    plt.boxplot(r2Test.T)
    plt.xlabel("Imaging Simulation")
    plt.ylabel("Model Performance (r2)")
    plt.savefig(outDir+"r2.png")
    