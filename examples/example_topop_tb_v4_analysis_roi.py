#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 21:29:32 2021

@author: qcao

Analysis code for example_topop_tb_v3.py

Parses and cleans load-driven phantoms. Computes Radiomic signatures. Compares with BvTv.

Compare with ROIs

"""

# FEA and BoneBox Imports
import os
import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
from bonebox.FEA.fea import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import vtk
from pyvistaqt import BackgroundPlotter

from skimage.morphology import ball, closing, binary_dilation, binary_closing

import pyvista as pv
pv.set_plot_theme("document")

# For PyRadiomics
import logging
import six
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

volumeShape = (100,100,100)

def ind2dir(ss,uu):
    # converts ss and uu to output directories
    saveNameAppend = "_phantom_ss_"+str(ss)+"_uu_"+str(uu)
    return "/data/BoneBox-out/topopt/lazy_v3_sweep/randstate_"+str(ss)+saveNameAppend+"/"

def getBVFandE(ss,uu):
    # Parse output directory given series and Ul index
    BVF = np.nan
    elasticModulus = np.nan
    
    out_dir = ind2dir(ss,uu)
    
    if os.path.exists(out_dir):
        if os.path.exists(out_dir+"bvf7.npy"):
            BVF = np.load(out_dir+"bvf7.npy")
        if os.path.exists(out_dir+"elasticModulus7.npy"):
            elasticModulus = np.load(out_dir+"elasticModulus7.npy")
    
    return BVF, elasticModulus

def getVolume(ss,uu):
    # Parse output directory and get volume
    volume = np.zeros(volumeShape)
    volume[:] = np.nan
    
    out_dir = ind2dir(ss,uu)
    
    if os.path.exists(out_dir):
        if os.path.exists(out_dir+"volume_8.npy"):
            volume = np.load(out_dir+"volume_8.npy")
    
    return volume

def computeWaveletFeatures(image, mask, featureFunc=glcm.RadiomicsGLCM):
    
    """
    featureFunc:
        firstorder.RadiomicsFirstOrder
        glcm.RadiomicsGLCM
    
    """
    
    featureNames = []
    featureVals = []
    
    for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
        
        waveletFirstOrderFeaturs = featureFunc(decompositionImage, mask, **inputKwargs)
        waveletFirstOrderFeaturs.enableAllFeatures()
        results = waveletFirstOrderFeaturs.execute()
        print('Calculated firstorder features with wavelet ', decompositionName)
        
        for (key, val) in six.iteritems(results):
            
            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
            print('  ', waveletFeatureName, ':', val)
            
            featureNames.append(waveletFeatureName)
            featureVals.append(val)
            
    return featureNames, np.array(featureVals)

def calculate_fid(act1, act2):
    
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    
	return fid

if __name__ == "__main__":
    
    save_dir = "/data/BoneBox-out/topopt/lazy_v3_sweep/"
    
    # Generate N phantom series, 3 resorption intensities per series
    Nseries = 400
    Nresorption = 3
    
    # Create array of BVFs and ElasticModuli
    bvfs = np.zeros((Nseries, Nresorption))
    Es = np.zeros((Nseries, Nresorption))
    
    # Array of random Uls (between 0.1 and 0.25), should be same as in example script.
    randStateUls = 3012
    Ulmin = 0.1
    Ulmax = 0.25
    Uls = sampleUniformZeroOne(((Nseries,Nresorption)), randState=randStateUls)*(Ulmax-Ulmin) + Ulmin
    
    # Retrieve BVF and ElasticModulus
    for ss in range(Nseries):
        for uu in range(Nresorption):
            bvfs[ss,uu], Es[ss,uu] = getBVFandE(ss,uu)
            
    inds = np.invert(np.isnan(bvfs))
    inds_nz = np.nonzero(inds)
    
    # Correlation Coefficients
    def linearFit(xx, yy):
    # r2 with radiomics
    # returns fit x, fit y, rs
        mfit, bfit = np.polyfit(xx, yy, 1)
        rs = np.corrcoef(xx, yy)[0,1]**2
        mi, ma = np.min(xx), np.max(yy)
        xxx = np.array([mi, ma])
        yyy = mfit*xxx + bfit
        return xxx, yyy, rs
    
    def reject_outliers(data, m=2):
        ind  = abs(data - np.mean(data)) < m * np.std(data)
        return ind, data[ind]
    
    # Correlation Coefficients
    def linearFitRejectOutliers(xx, yy):
    # r2 with radiomics
    # returns fit x, fit y, rs
        
        ind, yy = reject_outliers(yy, m=2)
        xx = xx[ind]
    
        mfit, bfit = np.polyfit(xx, yy, 1)
        rs = np.corrcoef(xx, yy)[0,1]**2
        mi, ma = np.min(xx), np.max(yy)
        xxx = np.array([mi, ma])
        yyy = mfit*xxx + bfit
        return xxx, yyy, rs
    
    # Correlation Coefficients
    def polyFitRejectOutliers(xx, yy, order = 2):
    # r2 with radiomics
    # returns fit x, fit y, rs
        
        ind, yy = reject_outliers(yy, m=2)
        xx = xx[ind]
    
        p = np.polyfit(xx, yy, 1)
        yyf = np.polyval(p,xx)
        rs = np.corrcoef(yy, yyf)[0,1]**2
        mi, ma = np.min(xx), np.max(yy)
        xxx = np.array([mi, ma])

        return np.sort(xx), np.sort(yyf), rs
    
    # Plot BVF and Elastic Modulus vs Uls
    fig, ax1 = plt.subplots()
    
    xx, yy, rs1 = linearFitRejectOutliers(Uls[inds].flatten(), bvfs[inds].flatten())
    ax1.plot(Uls[inds].flatten(), bvfs[inds].flatten(),'ko')
    ax1.plot(xx, yy, 'k-')
    ax1.set_ylim(0.16,0.28)
    ax1.set_xlabel("Resorption Threshold $U_l$")
    ax1.set_ylabel("BVF")
    ax1.grid("major")
    ax1.set_xlim(0.1,0.25)
    
    xx, yy, rs2 = linearFitRejectOutliers(Uls[inds].flatten(), Es[inds].flatten())
    ax2 = ax1.twinx()
    ax2.plot(Uls[inds].flatten(), Es[inds].flatten(),'rv')
    ax2.plot(xx, yy, 'r--')
    ax2.set_ylabel("Elastic Modulus $E$",color='r')
    ax2.set_ylim(0,10e7)
    
    ax2.tick_params(axis ='y', labelcolor = 'r')
    
    plt.savefig(save_dir+"BVF_Es_vs_Ul.png")
    
    print("BVF vs Ul: r2="+str(rs1))
    print("Es vs Ul: r2="+str(rs2))
    

    # np.corrcoef(bvfs[inds], Es[inds])
    # np.corrcoef(bvfs[inds], Uls[inds])
    # # np.corrcoef(Es[inds], Uls[inds])
    
    # Plot Es vs BVF
    fig, ax1 = plt.subplots()
    xx, yy, rs3 = polyFitRejectOutliers(bvfs[inds].flatten(), Es[inds].flatten())
    ax1.plot(bvfs[inds].flatten(), Es[inds].flatten(),'ko')
    ax1.plot(xx, yy, 'k-')
    ax1.set_ylim(0,3e7)
    ax1.set_xlim(0.16,0.28)
    ax1.set_xlabel("BVF")
    ax1.set_ylabel("Elastic Modulus $E$")
    ax1.grid("major")

    plt.savefig(save_dir+"Es_vs_BVF.png")
    
    print("Es vs BVF: r2="+str(rs3))

    #% Look at radiomics features
    
    # Initialize array of features
    features = np.zeros((Nseries, Nresorption, 93))
    features[:] = np.nan
    
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    settings['imageType'] = ['original','wavelet']
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    extractor.enableImageTypeByName("Wavelet")
    
    # extractor.disableAllImageTypes()
    # extractor.enableImageTypeByName(imageType="Original")
    # extractor.enableImageTypeByName(imageType="Wavelet")
    # extractor.enableFeatureClassByName("glcm")
    
    # Test extraction pipeline on one volume
    ss = 0; uu = 0
    volume = getVolume(ss,uu).astype(int)*255
    volumeSITK = sitk.GetImageFromArray(volume)
    maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
    
    wvltFeatureNames, wvltFeatures = computeWaveletFeatures(volumeSITK, maskSITK)
    
    featureVectorOriginal = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    volumeSITKWavelets = radiomics.imageoperations.getWaveletImage(volumeSITK, maskSITK)
    
    featureVectorWavelet = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="wavelet")
    
    featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    
    #%
    
    computeFeatures = False
    
    if computeFeatures:
    
        wvltFeatures = np.zeros((Nseries, Nresorption, 192))
        wvltFeatures[:] = np.nan
        
        # Extract volume and compute features
        for ss in range(Nseries):
            for uu in range(Nresorption):
                
                if inds[ss,uu]:
                
                    volume = getVolume(ss,uu).astype(int)*255
                    
                    volumeSITK = sitk.GetImageFromArray(volume)
                    maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
                    featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
                    featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
                    features[ss,uu,:] = featureVectorArray
                    
                    wvltFeatureNames, wvltFeatures[ss,uu,:] = computeWaveletFeatures(volumeSITK, maskSITK)
        

        
        # Reshape feature matrices
        featuresReshaped = features.reshape((-1,93), order='F')
        wvltFeaturesReshaped = wvltFeatures.reshape((-1,192), order='F')
        indsReshaped = inds.reshape((-1,), order='F')
        featuresReshaped = featuresReshaped[indsReshaped,:]
        wvltFeaturesReshaped = wvltFeaturesReshaped[indsReshaped,:]
        
        # Save feature vectors
        np.save(save_dir+"features",features)
        np.save(save_dir+"featuresReshaped",featuresReshaped)
        np.save(save_dir+"wvltFeaturesReshaped",wvltFeaturesReshaped)

    #%% Radiomic Features of ROIs
    
    plt.close('all')
    
    import nrrd
    import glob
    
    def readROI(filename):
        roiBone, header = nrrd.read(filename)
        roiBone[roiBone==255] = 1 # units for this is volume
        return roiBone
    
    roi_dir = "/data/BoneBox/data/rois/"
    
    Nrois  = len(glob.glob(roi_dir+"isodata_*_roi_*.nrrd"))
    
    featuresROI = np.zeros((Nrois,93))
    
    for ind in range(Nrois):
        print(ind)
        fn = glob.glob(roi_dir+"isodata_*_roi_"+str(ind)+".nrrd")[0]
        roiBone = readROI(fn)
        
        volume = roiBone.astype(int)*255
        
        # Take ROI center
        volume = volume[50:150,50:150,50:150]
        
        volumeSITK = sitk.GetImageFromArray(volume)
        maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
        featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
        featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
        featuresROI[ind,:] = featureVectorArray
        
        # wvltFeatureNames, wvltFeatures[ss,uu,:] = computeWaveletFeatures(volumeSITK, maskSITK)
    
    np.save(save_dir+"featuresROI",featuresROI)
    
    #%%
    
    featureNames = list(featureVector.keys())
    
    import seaborn as sns
    import pandas as pd
    
    sns.set_theme(style="whitegrid")
    
    featuresReshaped = np.load(save_dir+"featuresReshaped.npy")
    featuresROI = np.load(save_dir+"featuresROI.npy")
    
    featuresAll = np.vstack((featuresReshaped,featuresROI))
    
    sourceList = []
    for ii in range(200):
        sourceList.append("Phantom")
    for ii in range(208):
        sourceList.append("L1 Spine")
    
    df = pd.DataFrame(data = featuresAll,
                      columns = featureNames) 
    df["source"] = sourceList
    df["all"] = ""
    
    fig_dir = save_dir+"comparison_with_rois/"
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    
    # Draw a nested violinplot and split the violins for easier comparison
    for ind in range(93):
        fig, ax = plt.subplots(figsize=(5,10))
        sns.violinplot(data=df, x="all", y=featureNames[ind], hue="source",
                   split=True, inner="quart", linewidth=1)
        sns.despine(left=True)
        plt.savefig(fig_dir+"fig_"+str(ind)+"_"+featureNames[ind])
        plt.close("all")
        
    #%% Komogorov-smirnov test
    
    from scipy.stats import ks_2samp
    
    kss = np.zeros(93)
    ps = np.zeros(93)
    
    for ind in range(93):
        kss[ind], ps[ind] = scipy.stats.ks_2samp(featuresReshaped[:,ind], featuresROI[:,ind])
        
    
    
    #%% Prep data for regressor
    
    # # Extract Feature Names
    # featureNames = list(featureVector.keys())
    
    # indsReshaped = inds.reshape((-1,), order='F')
    # features = np.load(save_dir+"features.npy")
    # featuresReshaped = np.load(save_dir+"featuresReshaped.npy")
    # wvltFeaturesReshaped = np.load(save_dir+"wvltFeaturesReshaped.npy")
    
    # EsReshaped = Es.reshape((-1,), order='F')[indsReshaped]
    # bvfsReshaped = bvfs.reshape((-1,), order='F')[indsReshaped]
    
    # # combine BVF with wavelet GLCM features
    # # features_norm = np.concatenate((bvfsReshaped[:,None],wvltFeaturesReshaped),axis=1) # featuresReshaped # Feature Vector
    # features_norm = np.concatenate((bvfsReshaped[:,None],featuresReshaped),axis=1) # featuresReshaped # Feature Vector
    # features_norm -= np.mean(features_norm,axis=0) # center on mean
    # features_norm /= np.std(features_norm,axis=0) # scale to standard deviation
    # features_norm[np.isnan(features_norm)] = 0
    
    # # features_norm_names = ["BVF"]+wvltFeatureNames
    # features_norm_names = ["BVF"]+featureNames

    # roi_vm_mean = EsReshaped # Label
    
    # # Reject pathologic outliers in the dataset
    # ii, roi_vm_mean = reject_outliers(roi_vm_mean, m=1)
    # features_norm = features_norm[ii,:]
    # bvfsReshaped = bvfsReshaped[ii]
    
    # Ntrain = 110 # Training Testing Split
    
    # #% Feature selection
    # from sklearn.feature_selection import SelectKBest, VarianceThreshold
    # from sklearn.feature_selection import chi2, f_classif, f_regression
    
    # # # features_norm = SelectKBest(f_regression, k=20).fit_transform(features_norm, roi_vm_mean)
    # # features_norm = VarianceThreshold(0.95).fit_transform(features_norm)
    # # print(features_norm.shape)
    
    # #%
    # ytrain = roi_vm_mean[:Ntrain]
    # ytest = roi_vm_mean[Ntrain:]
    # Xtrain1 = features_norm[:Ntrain,:]
    # Xtrain2 = bvfsReshaped[:Ntrain].reshape(-1,1)
    # Xtest1 = features_norm[Ntrain:,:]
    # Xtest2 = bvfsReshaped[Ntrain:].reshape(-1,1)
    
    # # from xgboost import XGBRegressor
    # # from sklearn.model_selection import cross_val_score
    
    # # scores = cross_val_score(XGBRegressor(objective='reg:squarederror'), Xtrain1, ytrain, scoring='neg_mean_squared_error')
    
    # #%% Radiomics + Random Forestocu

    # plt.close('all')
    
    # import random
    
    # randState = 123
    
    # random.seed(randState)
    
    # # non-linear without feature selection
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.model_selection import GridSearchCV
    
    # param_grid = [
    #         {'max_depth': [2,4,8,16,32,64], # 16
    #         'max_leaf_nodes': [2,4,8,16,32,64], # 8
    #         'n_estimators': [10,50,100,150,200]} # 50
    #             ]
    
    # # param_grid = [
    # #         {'max_depth': [2,4,8,16], # 16
    # #         'max_leaf_nodes': [2,4,8,16], # 8
    # #         'n_estimators': [10,50,100]} # 50
    # #             ]
    
    # rfr = GridSearchCV(
    #         RandomForestRegressor(random_state = randState), 
    #         param_grid, cv = 5,
    #         scoring = 'explained_variance',
    #         n_jobs=-1
    #         )
    
    # rfr2 = GridSearchCV(
    #         RandomForestRegressor(random_state = randState), 
    #         param_grid, cv = 5,
    #         scoring = 'explained_variance',
    #         n_jobs=-1
    #         )
    
    # # Fit with full set of features.
    # grid_result = rfr.fit(Xtrain1, ytrain)
    # yTrain_fit_rfr = rfr.predict(Xtest1)
    
    # print("Best estimator for BvF+radiomics...")
    # rfr.best_estimator_
    
    # # Fit with BVTV only.
    # grid_result2 = rfr2.fit(Xtrain2, ytrain)
    # yTrain_fit_rfr2 = rfr2.predict(Xtest2)
    
    # print("Best estimator for BVF...")
    # rfr2.best_estimator_
    
    # # r2 with radiomics
    # mfit, bfit = np.polyfit(ytest, yTrain_fit_rfr, 1)
    # rs = np.corrcoef(roi_vm_mean[Ntrain:], yTrain_fit_rfr)[0,1]**2
    # print("BVF+Radiomics rs:"+str(rs))
    
    # # r2 with BVFS
    # mfit2, bfit2 = np.polyfit(roi_vm_mean[Ntrain:], yTrain_fit_rfr2, 1)
    # rs2 = np.corrcoef(roi_vm_mean[Ntrain:], yTrain_fit_rfr2)[0,1]**2
    # print("BVF rs:"+str(rs2))
    
    # plt.figure()
    # plt.plot(roi_vm_mean[Ntrain:],yTrain_fit_rfr2,'bv')
    # plt.plot(roi_vm_mean[Ntrain:], mfit2*roi_vm_mean[Ntrain:] + bfit2, "b--")
    # plt.plot(roi_vm_mean[Ntrain:],yTrain_fit_rfr,'ko')
    # plt.plot(roi_vm_mean[Ntrain:], mfit*roi_vm_mean[Ntrain:] + bfit, "k-")
    # plt.xlabel("$\mu$FE Elastic Modulus")
    # plt.ylabel("Predicted Elastic Modulus")
    # plt.savefig(save_dir+"Elastic Modulus Predicted vs True.png")
    # plt.close("all")
    
    # # Plot feature importance
    # importances = rfr.best_estimator_.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # std = np.std([tree.feature_importances_ for tree in rfr.best_estimator_], axis = 0)
    # plt.figure()
    # plt.title('Feature importances')
    # plt.barh(range(20), importances[indices[0:20]], yerr = std[indices[0:20]], align = 'center',log=True)
    # plt.yticks(range(20), list(features_norm_names[i] for i in indices[0:20] ), rotation=0)
    # plt.gca().invert_yaxis()
    # plt.show()
    
    # plt.subplots_adjust(left=0.7,bottom=0.1, right=0.8, top=0.9)
    # plt.savefig(save_dir+"Feature Importances.png")
    # plt.close("all")
    
    # #%% Changes in radiomic signature with remodeling
    
    # # Retrieve indices of samples with continuous Uls
    # indsUl = (np.sum(inds,axis=1) == 3)
    # indsUlnz = np.nonzero(indsUl)[0]
    
    # featuresCrop = features[indsUlnz,:,:]
    # UlsCrop = Uls[indsUlnz,:]
    
    # plt.figure()
    
    
    #%%
    
    # grid_result = rfr.fit(features_norm[:Ntrain,:], roi_vm_mean[:Ntrain])
    # yTest_fit_rfr = rfr.predict(features_norm[Ntrain:])
    
    # # sns.set(font_scale=1)
    
    # mfit, bfit = np.polyfit(roi_vm_mean[Ntrain:], yTest_fit_rfr, 1)
    # pr2 = np.corrcoef(roi_vm_mean[Ntrain:], yTest_fit_rfr)[0,1]**2
    # print(pr2)
    
    # plt.figure()
    # plt.plot(roi_vm_mean[Ntrain:],yTest_fit_rfr,'ko')
    # plt.plot(roi_vm_mean[Ntrain:], mfit*roi_vm_mean[Ntrain:] + bfit, "b--")
    
    # importances = rfr.best_estimator_.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # std = np.std([tree.feature_importances_ for tree in rfr.best_estimator_], axis = 0)
    # plt.figure()
    # plt.title('Feature importances')
    # plt.barh(range(20), importances[indices[0:20]], yerr = std[indices[0:20]], align = 'center',log=True)
    # plt.yticks(range(20), list(featureNames[i] for i in indices[0:20] ), rotation=0)
    # plt.gca().invert_yaxis()
    # plt.show()
    
    # plt.subplots_adjust(left=0.7,bottom=0.1, right=0.8, top=0.9)