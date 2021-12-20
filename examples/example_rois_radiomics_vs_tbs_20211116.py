"""

Example script for bone analysis based on:
    all_proj_analysis_radiomics_4_bonej_v3_skeleton_FDASF.py
    -- Based on example_rois_radiomics_vs_tbs but with 3D radiomics features
    
Qian Cao

Example commands for generating radiomic features:
    
# pyradiomics pyradiomics_settings_all_projs_mask.csv --mode voxel --param pyradiomics-settings.yaml --out-dir voxel-out --jobs 80 --verbosity 5
# pyradiomics pyradiomics_settings_all_projs_mask.csv -o out.csv -f csv --jobs 80 --verbosity 5
# pyradiomics pyradiomics_settings_all_projs_mask.csv -o output.txt -f txt --jobs 80 --verbosity 5

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

import random

# # non-linear without feature selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from matplotlib.colors import ListedColormap
# https://stackoverflow.com/questions/37902459/seaborn-color-palette-as-matplotlib-colormap

from scipy.spatial import distance
from scipy.cluster import hierarchy
import seaborn as sns

from scipy.cluster import hierarchy

import scipy.stats as stats

def filter_substring(string_list,substr):
	list_filt = []
	for str in string_list:
		if substr in str:
			list_filt.append(str)
	return list_filt

def repeat_ele(x):
    return np.kron(x,[1,1])

def func(x, a, b):
    return a * x + b

def removeZeroRows(data):
    return data[~np.all(data == 0, axis=1)]

def clusterANOVA(g):

    ftree = hierarchy.to_tree(g.dendrogram_col.linkage)
    
    ind_c1 = ftree.left.left.pre_order()
    ind_c2 = ftree.left.right.pre_order()
    ind_c3 = ftree.right.left.pre_order()
    ind_c4 = ftree.right.right.pre_order()
    
    c1 = roi_vm_mean[ind_c1]
    c2 = roi_vm_mean[ind_c2]
    c3 = roi_vm_mean[ind_c3]
    c4 = roi_vm_mean[ind_c4]

    stat, pval = stats.f_oneway(c1, c2, c3, c4)
    
    return stat, pval

if __name__ == "__main__":
    
    """ 
    
    Feature Names: feature_names (92,)
    
    Features (radiomic features are normalized)
    
    3D radiomic features: featuresROI (416,92)
    2D radiomic featuresL features_norm (416,92)
    TBS: proj_tbs (416,)
    BVTV: roi_bvtv (416,)
    
    FEA
    
    Mean Von Mises Stress: roi_vm_mean (416,)
    
    """
    
    # Folders containing ROIs
    roi_dir = "../data/rois/"
    
    # Output Folders
    out_dir = "/data/BoneBox-out/example_rois_radiomics_vs_tbs_20211116/"
    
    # TBS corresponding to ROIs
    proj_tbs_fn = "meanTBSList.npy" # This is TBS
    radiomics_fn = "../data/output.txt" # This is 2D radiomic features.
    fem_dir = "../data/"
    
    # Get 3D radiomic features
    save_dir = "/data/BoneBox-out/topopt/lazy_v3_sweep/"
    featuresROI = np.load(save_dir+"featuresROI.npy") # This is 3D radiomic features.
    
    num_features = 93
    num_cases = 208*2
    
    with open(radiomics_fn,'r') as outfile:
    	lines = outfile.readlines()
        
    roi_delta_z = repeat_ele(np.load(fem_dir+"roi_delta_z.npy"))
    roi_num_nodes = repeat_ele(np.load(fem_dir+"roi_num_nodes.npy"))
    roi_vm_mean = repeat_ele(np.load(fem_dir+"roi_vm_mean.npy")) # This is mean stress.
    roi_bvtv = repeat_ele(np.load(fem_dir+"roi_bvtv.npy")) # This is BVTV
    featuresROI = np.repeat(featuresROI, 2, axis=0)
    
    # Noramlize features ROI
    featuresROIMean = np.mean(featuresROI, axis=(0))
    featuresROIStd = np.std(featuresROI, axis=(0))
    featuresROI = (featuresROI-featuresROIMean[None,:])/featuresROIStd[None,:]
    featuresROI[np.isnan(featuresROI)] = 0
    featuresROI[np.isinf(featuresROI)] = 0
    
    # pop the 10th feature
    featuresROI = np.delete(featuresROI,10,axis=1)
    
    roi_stiffness = - roi_num_nodes / roi_delta_z  / 1e9 # This is stiffness
    
    proj_tbs = np.load(proj_tbs_fn)

    #% Parse Feature Names
    
    features = np.zeros((num_cases,num_features))
    feature_names = [];
    feature_names_notype = [];
    feature_types = [];
    
    for ii in range(num_cases):
        case = filter_substring(lines,'Case-'+str(ii+1)+'_original_')
        for jj in range(num_features):
            features[ii,jj] = float(case[jj].split(':')[1].rstrip())
            if ii==0:
                case_split = case[jj].split('_')
                case_split2 = case_split[3].split(':')
                feature_names.append(case_split[2]+' '+case_split2[0])
                
    for jj in range(num_features):
        fn = feature_names[jj]
        sp = fn.split(" ")
        if sp[0] == 'firstorder':
            sp[0] = 'FirstOrder'
        else:
            sp[0] = sp[0].upper()
        feature_names[jj] = " ".join(sp)
        feature_types.append(sp[0])
        feature_names_notype.append(sp[1])
    
    # regression
    print('regression ...')
    
    # remove MIN
    features = np.delete(features,10,axis=1)
    feature_names.pop(10)
    feature_types.pop(10)
    
    # Normalization
    features_norm = features.copy()
    features_norm -= np.mean(features,axis=0) # center on mean
    features_norm /= np.std(features,axis=0) # scale to standard deviation

    #%%
    FIGSIZE = (13,10)
    
    cmap = sns.diverging_palette(240, 10, n=21)
    cmap = ListedColormap(cmap.as_hex())
    
    plt.figure(figsize=FIGSIZE)
    plt.imshow(features_norm.T,cmap=cmap,aspect='auto',interpolation="nearest")
    plt.clim(-2,2)
    plt.xticks([])
    plt.yticks(np.arange(92),labels=feature_names,fontsize=8)
    plt.gca().yaxis.tick_right()
    plt.tight_layout()
    # plt.ylabel()
    
    plt.savefig(out_dir+"Radiomics2D_features.png")
    
    cmap = sns.diverging_palette(240, 10, n=21)
    cmap = ListedColormap(cmap.as_hex())
    
    plt.figure(figsize=FIGSIZE)
    plt.imshow(featuresROI.T,cmap=cmap,aspect='auto',interpolation="nearest")
    plt.clim(-2,2)
    plt.xticks([])
    plt.yticks(np.arange(92),labels=feature_names,fontsize=8)
    plt.gca().yaxis.tick_right()
    plt.tight_layout()
    # plt.ylabel()
    
    plt.savefig(out_dir+"Radiomics3D_features.png")
    
    plt.close("all")
    
    #%% Manually compute clusters and pass intofigure
    
    # 2D Radiomics
    cmap = sns.diverging_palette(240, 10, n=21)
    g = sns.clustermap(features_norm.T, metric = 'euclidean', cmap = cmap, vmin=-3, vmax=3, cbar=False, method="average")
    ax = g.ax_heatmap
    ax.set_axis_off()
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    plt.savefig(out_dir+"Radiomics2D_dendrogram_1.png")
    
    plt.figure(figsize=(7.31,2.45))
    plt.plot(roi_vm_mean[g.dendrogram_col.reordered_ind],'ko',markersize=4)
    plt.xlim(0,416)
    plt.ylim(0,0.22)
    
    plt.savefig(out_dir+"Radiomics2D_dendrogram_2.png")
    
    plt.close("all")
    
    # 3D Radiomics
    cmap = sns.diverging_palette(240, 10, n=21)
    g = sns.clustermap(removeZeroRows(featuresROI).T, metric = 'euclidean', cmap = cmap, vmin=-3, vmax=3, cbar=False, method="average")
    ax = g.ax_heatmap
    ax.set_axis_off()
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    plt.savefig(out_dir+"Radiomics3D_dendrogram_1.png")
    
    plt.figure(figsize=(7.31,2.45))
    plt.plot(roi_vm_mean[g.dendrogram_col.reordered_ind],'ko',markersize=4)
    plt.xlim(0,416)
    plt.ylim(0,0.22)
    
    plt.savefig(out_dir+"Radiomics3D_dendrogram_2.png")
    
    plt.close("all")
    
    #%%
    
    def RFE(Xtrain,ytrain,Xtest,k=10):
        """
        Recursive Feature Elimination
        Fit on training data, transform on training and test data
        """
        
        from sklearn.feature_selection import RFE
        from sklearn.tree import DecisionTreeRegressor
        
        rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=k)
        rfe.fit(Xtrain, ytrain)
        Xtrain = rfe.transform(Xtrain)
        Xtest = rfe.transform(Xtest)
        
        return Xtrain, Xtest
    
    #%% Random Forest Grid Search using 5-fold cross validation
    
    plt.close('all')
        
    random.seed(1234)
    
    rfr_params = {'max_depth': 4,
                  'max_leaf_nodes': 8,
                  'n_estimators': 16}
    
    print(rfr_params)
    
    # plt.figure()
    # plt.plot(roi_vm_mean,yTrain_fit_rfr,'ko')
    
    # Plot feature importance
    
    # importances = rfr.best_estimator_.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # std = np.std([tree.feature_importances_ for tree in rfr.best_estimator_], axis = 0)
    # plt.figure()
    # plt.title('Feature importances')
    # plt.barh(range(20), importances[indices[0:20]], yerr = std[indices[0:20]], align = 'center',log=True)
    # plt.yticks(range(20), list( feature_names[i] for i in indices[0:20] ), rotation=0)
    # plt.gca().invert_yaxis()
    # plt.show()
    
    #% Random Forest Regression - Cross Validate on Final Model
    
    ProjectionsPerBone = 13*2
    
    # Projection Radiomics
    roi_vm_mean_tests = np.empty((16,ProjectionsPerBone))
    roi_vm_mean_preds = np.empty((16,ProjectionsPerBone))
    
    # BvTv only
    roi_vm_mean_tests0 = np.empty((16,ProjectionsPerBone))
    roi_vm_mean_preds0 = np.empty((16,ProjectionsPerBone))
    
    # TBS only
    roi_vm_mean_tests1 = np.empty((16,ProjectionsPerBone))
    roi_vm_mean_preds1 = np.empty((16,ProjectionsPerBone))
    
    # 3D Radiomics
    roi_vm_mean_tests2 = np.empty((16,ProjectionsPerBone))
    roi_vm_mean_preds2 = np.empty((16,ProjectionsPerBone))
    
    fits = np.empty((16,2))
    fitps = np.empty((16,2))
    
    ccs = np.empty((16,1))
    ccs0 = np.empty((16,1))
    ccs1 = np.empty((16,1))
    ccs2 = np.empty((16,1))
    
    nrmses = np.empty((16,1))
    nrmses0 = np.empty((16,1))
    nrmses1 = np.empty((16,1))
    nrmses2 = np.empty((16,1))
    
    # nrmses_fit = np.empty((16,18))
    
    nrmses_train = np.empty((16,1))
    nrmses_const = np.empty((16,1))
    
    rfs = np.empty((16,rfr_params['n_estimators']), dtype = RandomForestRegressor)
    
    pval = [0,np.max(roi_vm_mean)]
    
    imps = np.empty((16,92))
    
    train_scores = np.empty((16,1))
    
    for bb in range(16): # 16 bones total, 16 folds
        
        # Partition Training and Testing in Cross-validation
        features_norm_test, roi_vm_mean_test = features_norm[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone,:], roi_vm_mean[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone]
        
        # training set
        features_norm_train, roi_vm_mean_train = features_norm.copy(), roi_vm_mean.copy()
        
        features_norm_train = np.delete(features_norm_train, slice(ProjectionsPerBone*bb,ProjectionsPerBone*bb+ProjectionsPerBone),0)

        roi_vm_mean_train = np.delete(roi_vm_mean_train,slice(ProjectionsPerBone*bb,ProjectionsPerBone*bb+ProjectionsPerBone),0)
        
        # TODO RFE
        features_norm_train, features_norm_test = RFE(features_norm_train, roi_vm_mean_train, features_norm_test)
        
        rf = RandomForestRegressor(**rfr_params, n_jobs = -1,random_state =1)
        rf.fit(features_norm_train, roi_vm_mean_train)
        roi_vm_mean_pred = rf.predict(features_norm_test)
        roi_vm_mean_train_pred = rf.predict(features_norm_train)
    
        # rf.score(features_norm_train, roi_vm_mean_train)
        # rf.score(features_norm_test, roi_vm_mean_pred)
    
        nrmses[bb,:] = np.sqrt(np.mean((roi_vm_mean_pred-roi_vm_mean_test)**2))/np.max(roi_vm_mean_test)
        
        nrmses_train[bb,:] = np.sqrt(np.mean((roi_vm_mean_train_pred-roi_vm_mean_train)**2))/np.max(roi_vm_mean_train)
        
        
        # Partitions for 3D Radiomics, redo roi_vm_mean_test and train
        roi_vm_mean_test = roi_vm_mean[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone]
        features_norm_test2 = featuresROI[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone,:]
        
        features_norm_train2 = featuresROI.copy()
        roi_vm_mean_train = roi_vm_mean.copy()
        
        features_norm_train2 = np.delete(features_norm_train2, slice(ProjectionsPerBone*bb,ProjectionsPerBone*bb+ProjectionsPerBone),0)
        roi_vm_mean_train = np.delete(roi_vm_mean_train,slice(ProjectionsPerBone*bb,ProjectionsPerBone*bb+ProjectionsPerBone),0)
        
        # TODO RFE
        features_norm_train2, features_norm_test2 = RFE(features_norm_train2, roi_vm_mean_train, features_norm_test2)
        
        #% BvTv only
        
        xdata = roi_bvtv[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone]
        ydata = roi_vm_mean_test
        
        popt, pcov = curve_fit(func, xdata, ydata)
        
        residuals = ydata - func(xdata, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata-np.mean(ydata))**2)
        ccs0[bb] = (1 - (ss_res / ss_tot))
        nrmses0[bb] = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))/np.mean(ydata)
        # rmse = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))
        
        roi_vm_mean_tests0[bb,:] = roi_vm_mean_test
        roi_vm_mean_preds0[bb,:] = func(xdata, *popt)
        
        #% END BvTv ONLY
        
        xbvtv = xdata
    
        #% TBS only
        
        xdata = proj_tbs[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone]
        ydata = roi_vm_mean_test
        
        popt, pcov = curve_fit(func, xdata, ydata)
        
        residuals = ydata - func(xdata, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata-np.mean(ydata))**2)
        ccs1[bb] = (1 - (ss_res / ss_tot))
        nrmses1[bb] = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))/np.mean(ydata)
        # rmse = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))
        
        roi_vm_mean_tests1[bb,:] = roi_vm_mean_test
        roi_vm_mean_preds1[bb,:] = func(xdata,*popt)
        
        #% END BvTv ONLY
        
        xtbs = xdata
        
        #% 3D radiomics only
        
        rf2 = RandomForestRegressor(**rfr_params, n_jobs = -1, random_state = 1)
        rf2.fit(features_norm_train2, roi_vm_mean_train)
        roi_vm_mean_pred2 = rf.predict(features_norm_test2)
        roi_vm_mean_train_pred2 = rf.predict(features_norm_train2)
        
        roi_vm_mean_tests2[bb,:] = roi_vm_mean_test
        roi_vm_mean_preds2[bb,:] = roi_vm_mean_pred2
        
        #%
        
        nrmses_const[bb,:] = np.sqrt(np.mean((np.mean(roi_vm_mean_pred)-roi_vm_mean_test)**2))/np.max(roi_vm_mean_test)
        
        rfs[bb] = rf
        
        fit = np.polyfit(roi_vm_mean_test,roi_vm_mean_pred,1)
        fitp = np.polyval(fit,pval)
        
        roi_vm_mean_tests[bb,:] = roi_vm_mean_test
        roi_vm_mean_preds[bb,:] = roi_vm_mean_pred
        
        fits[bb,:] = fit
        fitps[bb,:] = fitp
        ccs[bb,:] = np.corrcoef(roi_vm_mean_test,roi_vm_mean_pred)[0,1]
        ccs2[bb,:] = np.corrcoef(roi_vm_mean_test,roi_vm_mean_pred2)[0,1]
        
        # # feature importance
        # imps[bb,:] = rf.feature_importances_
        # importances = rf.feature_importances_
        # indices = np.argsort(importances)[::-1]
        # std = np.std([tree.feature_importances_ for tree in rf], axis = 0)
        # plt.figure()
        # plt.bar(range(10), importances[indices[0:10]], yerr = std[indices[0:10]], align = 'center')
        # plt.xticks(range(10), list( feature_names[i] for i in indices[0:10] ), rotation=90)
        # plt.title('Feature importances')
            
    fit = np.polyfit(roi_vm_mean_tests.flatten(),roi_vm_mean_preds.flatten(),1)
    fitp = np.polyval(fit,pval)
    
    fit0 = np.polyfit(roi_vm_mean_tests0.flatten(),roi_vm_mean_preds0.flatten(),1)
    fitp0 = np.polyval(fit0,pval)
    
    cc = np.corrcoef(roi_vm_mean_test,roi_vm_mean_pred)[0,1]
    cc2 = np.corrcoef(roi_vm_mean_test,roi_vm_mean_pred2)[0,1]
    
    #%% Plot of correlation coefficients
    
    # Radiomics vs TBS 
    plt.figure()
    plt.plot(ccs1, ccs,'ko')
    plt.xlabel("r2 TBS")
    plt.ylabel("r2 Projection Radiomics")
    plt.axis('square')
    
    plt.plot([0,1],[0,1],'k-')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.savefig(out_dir+"Projection Radiomics vs TBS.png")
    plt.close("all")
    
    # Projection Radiomics vs BMD
    plt.figure()
    plt.plot(ccs0, ccs,'ko')
    plt.xlabel("r2 BMD")
    plt.ylabel("r2 Projection Radiomics")
    plt.axis('square')
    
    plt.plot([0,1],[0,1],'k-')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.savefig(out_dir+"Projection Radiomics vs BMD.png")
    plt.close("all")
    
    # 3D Radiomics vs TBS
    plt.figure()
    plt.plot(ccs1, ccs2,'ko')
    plt.xlabel("r2 TBS")
    plt.ylabel("r2 Volumetric Radiomics")
    plt.axis('square')
    
    plt.plot([0,1],[0,1],'k-')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.savefig(out_dir+"Volumetric Radiomics vs TBS.png")
    plt.close("all")
    
    # 3D Radiomics vs BMD
    plt.figure()
    plt.plot(ccs0, ccs2,'ko')
    plt.xlabel("r2 BMD")
    plt.ylabel("r2 Volumetric Radiomics")
    plt.axis('square')
    
    plt.plot([0,1],[0,1],'k-')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.savefig(out_dir+"Volumetric Radiomics vs BMD.png")
    plt.close("all")
        
    # 3D Radiomics vs 2D Radiomics
    plt.figure()
    plt.plot(ccs, ccs2,'ko')
    plt.xlabel("r2 Projection Radiomics")
    plt.ylabel("r2 Volumetric Radiomics")
    plt.axis('square')
    
    plt.plot([0,1],[0,1],'k-')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.savefig(out_dir+"Volumetric Radiomics vs Projection Radiomics.png")
    plt.close("all")
    
    
    #% Boxplot
    plt.boxplot([ccs0.flatten(),ccs1.flatten(),ccs.flatten(),ccs2.flatten()])
    plt.ylabel("r2 for cross-validation folds")
    plt.xlabel("Feature Type")
    
    plt.savefig(out_dir+"r2 BMD TBS ProjectionRadiomics VolumetricRadiomics.png")
    plt.close("all")
