#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:18:05 2022

https://thatascience.com/learn-machine-learning/pipeline-in-scikit-learn/

@author: qian.cao

# stratify based on BMD
#

"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import os

import sys
sys.path.append("../bonebox/metrics/")
from FeaturesRadiomics import *

import matplotlib.pyplot as plt

def pathBetsy(path):
    return path.replace("/gpfs_projects/qian.cao/BoneBox-out","/projects01/didsr-aiml/qian.cao/BoneBox-out")

if __name__ == "__main__":

    outDir = pathBetsy("/gpfs_projects/qian.cao/BoneBox-out/test_20220422_bin_cross_parallel_biomarker_SPIE/")
    os.makedirs(outDir,exist_ok = True)
    
    featuresDir = pathBetsy("/gpfs_projects/qian.cao/BoneBox-out/test_20220422_bin_cross_parallel/")
    
    # Copied from bin_cross_parallel
    nScales = np.linspace(1.2, 0.2, 60) # change noise only # sweeps across noise and resolution settings
    rScales = np.linspace(1, 0.3, 40)
    
    # Size of the test split
    num_bones_test = 7 # number of bones reserved for testing
    test_split_size = num_bones_test/16
    num_test = int(num_bones_test*13)
    
    featureNames = getRadiomicFeatureNames() # TODO: save and read from file
    features = np.load(featuresDir+"featuresArray.npy")
    
    fem_dir = "../data/"
    roi_vm_mean = np.load(fem_dir+"roi_vm_mean.npy")
    
    # Training and testing scores
    y_preds = np.zeros((num_test,features.shape[2],features.shape[3],features.shape[4]))
    r2Test = np.zeros((features.shape[2],features.shape[3],features.shape[4]))
    importances = np.zeros((features.shape[1],features.shape[2],features.shape[3],features.shape[4]))
    # remember to save y_test as well, this is constant throughout the script
    
    # for cind in range(features.shape[2]): # imaging condition
    
    #%% Run through all scenarios
    
    for indNoise, nscale in enumerate(nScales):
        for indResolution, rscale in enumerate(rScales):
            
            print(f"noise: {indNoise}, resolution: {indResolution}")
    
            for sind in range(features.shape[4]): # seed, instance
                
                # feature
                feat = features[:,:,indNoise,indResolution,sind]
                
                # data and target
                X = feat
                y = roi_vm_mean
                
                # Splitting data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, random_state = 3, shuffle=False)
                
                # Random forest Tree Regression Pipeline
                rf_pipe = Pipeline([('scl', StandardScaler()),
                                    ('reg',RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=0, n_jobs=-1))])
                
                rf_pipe.fit(X_train, y_train)
                y_pred = rf_pipe.predict(X_test)
                
                # scoreTest[cind,sind] = rf_pipe.score(y_pred, y_test)
                
                # Save output
                y_preds[:,indNoise,indResolution,sind] = y_pred
                r2Test[indNoise,indResolution,sind] = np.corrcoef(y_pred, y_test)[0,1]**2
                importances[:,indNoise,indResolution,sind] = rf_pipe['reg'].feature_importances_
                
                # Correlation plot
                plt.ioff()
                plt.figure()
                plt.plot(y_test, y_pred,'b.')
                plt.plot(*(np.linspace(0,np.max(y)),)*2,'k--')
                plt.xlabel("True")
                plt.ylabel("Predicted")
                plt.xlim([0,np.max(y)])
                plt.ylim([0,np.max(y)])
                plt.title(f"r2: {r2Test[indNoise,indResolution,sind]} Noise: {indNoise}, Resolution: {indResolution}, Instance: {sind}")
                plt.savefig(outDir+f"correlation_{indNoise}_{indResolution}_{sind}.png")
                plt.close("all")
                
            np.save(outDir+"y_preds",y_preds)
            np.save(outDir+"y_test",y_test)
            np.save(outDir+"r2Test",r2Test)
            np.save(outDir+"importances",importances)
            
    #%% Figures and Analysis (run the first cell first, skip the loops)
    
    y_preds = np.load(outDir+"y_preds.npy")
    y_test = np.load(outDir+"y_test.npy")
    r2Test = np.load(outDir+"r2Test.npy")
    importances = np.load(outDir+"importances.npy")
    
    plt.ion()
    
    fig = plt.figure(figsize=(7,8))
    cax = fig.axes
    im = plt.imshow(np.mean(r2Test,axis=2))
    plt.xlabel("Resolution")
    plt.ylabel("Noise Level")
    plt.title("r2 mean")
    plt.colorbar()
    plt.savefig(outDir+"fig-r2-mean.png")
    
    plt.figure(figsize=(7,8))
    plt.imshow(np.std(r2Test,axis=2),cmap="inferno")
    plt.xlabel("Resolution")
    plt.ylabel("Noise Level")
    plt.title("r2 std")
    plt.colorbar()
    plt.savefig(outDir+"fig-r2-std.png")
    
    #%% feature importances
    
    for ind in range(importances.shape[0]):
        print(ind)
        
        img = importances[ind,:,:,:]
        fn = featureNames[ind]
        
        plt.ioff()
        
        fig = plt.figure(figsize=(7,8))
        cax = fig.axes
        im = plt.imshow(np.mean(img,axis=2),cmap="YlGn")
        plt.xlabel("Resolution")
        plt.ylabel("Noise Level")
        plt.title(f"Importance Mean: {fn}")
        plt.colorbar()
        plt.savefig(outDir+f"fig-imp-{fn}-mean.png")
        
        plt.figure(figsize=(7,8))
        plt.imshow(np.std(img,axis=2),cmap="BuPu")
        plt.xlabel("Resolution")
        plt.ylabel("Noise Level")
        plt.title(f"Importance Std: {fn}")
        plt.colorbar()
        plt.savefig(outDir+f"fig-imp-{fn}-std.png")
        
        plt.close("all")
        
    #%% Sum all feature importances EXCEPT first order
    
    img = importances[18:,:,:,:]

    fig = plt.figure(figsize=(7,8))
    cax = fig.axes
    im = plt.imshow(np.mean(np.sum(img[19:],axis=0),axis=2),cmap="YlGn")
    plt.xlabel("Resolution")
    plt.ylabel("Noise Level")
    plt.title(f"Importance Mean: nonfirstorder")
    plt.colorbar()
    plt.savefig(outDir+f"fig-imp-nonfirstorder-mean.png")
        