# Trying out random forest decision trees

import numpy as np
import matplotlib.pyplot as plt
import nrrd
import radiomics
import os
import re

from matplotlib import style
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns

from scipy.optimize import curve_fit

import string

# pyradiomics pyradiomics_settings_all_projs_mask.csv --mode voxel --param pyradiomics-settings.yaml --out-dir voxel-out --jobs 80 --verbosity 5

# pyradiomics pyradiomics_settings_all_projs_mask.csv -o out.csv -f csv --jobs 80 --verbosity 5

# pyradiomics pyradiomics_settings_all_projs_mask.csv -o output.txt -f txt --jobs 80 --verbosity 5

roi_dir = '/gpfs_projects/qian.cao/data/L1_DICOM/isodata_transformed/rois_v2_mp/'
skeleton_dir = roi_dir + 'skeleton-2_v2_mp/'

out_dir = '/data/L1_DICOM/isodata_transformed/rois_v2_mp/proj-2_v2_mp_skeleton/'
out_fn = 'output.txt'
# bonej_fn = '/gpfs_projects/qian.cao/data/L1_DICOM/isodata_transformed/rois_v2_mp/roi_bonej_complete_sort.xlsx'

num_features = 93
num_cases = 208*2

with open(out_dir+out_fn,'r') as outfile:
	lines = outfile.readlines()

def filter_substring(string_list,substr):
	list_filt = []
	for str in string_list:
		if substr in str:
			list_filt.append(str)
	return list_filt

# linesf = filter_substring(lines,['_original_'])

# roi_vm_mean = np.load(out_dir+'vm_vals.npy')

def repeat_ele(x):
    return np.kron(x,[1,1])

roi_delta_z = repeat_ele(np.load(skeleton_dir+"roi_delta_z.npy"))
roi_num_nodes = repeat_ele(np.load(skeleton_dir+"roi_num_nodes.npy"))
roi_vm_mean = repeat_ele(np.load(skeleton_dir+"roi_vm_mean.npy"))
roi_vm_90p = repeat_ele(np.load(skeleton_dir+"roi_vm_90p.npy"))
roi_bvtv = repeat_ele(np.load(skeleton_dir+"roi_bvtv.npy"))

roi_stiffness = - roi_num_nodes / roi_delta_z  / 1e9

#%% BvTv vs Mean VM Stress

# def func(x, a, b):
#     return a * np.exp(-b * x)

# # def func(x, a, b, c):
# #     return a * np.exp(-b * x) + c

# # def func(x, a, b, c, d):
# #     p = [a,b,c]
# #     return np.polyval(p,x)

# xdata = roi_bvtv
# ydata = roi_vm_mean

# plt.figure()
# plt.plot(xdata, ydata, 'k.', label='data')

# xfit = np.linspace(0,np.max(xdata),2000)
# popt, pcov = curve_fit(func, xdata, ydata)
# # plt.plot(xfit, func(xfit, *popt), 'k--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
# # plt.plot(xfit, func(xfit, *popt), 'k--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# plt.plot(xfit, func(xfit, *popt), 'k--', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

# residuals = ydata- func(xdata, *popt)
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((ydata-np.mean(ydata))**2)
# r2 = (1 - (ss_res / ss_tot))**2
# nrmse = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))/np.max(ydata)
# rmse = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))

# print(" ".join([str(r2),str(nrmse),str(rmse)]))

#%% Stiffness fit

# def func(x, a, b):
#     return a * x + b

# xdata = roi_bvtv
# ydata = roi_stiffness

# plt.figure()
# plt.plot(xdata, ydata, 'k.', label='data')

# xfit = np.linspace(0,np.max(xdata),2000)
# popt, pcov = curve_fit(func, xdata, ydata)
# plt.plot(xfit, func(xfit, *popt), 'k--', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

# residuals = ydata- func(xdata, *popt)
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((ydata-np.mean(ydata))**2)
# r2 = (1 - (ss_res / ss_tot))**2
# nrmse = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))/np.max(ydata)
# rmse = np.sqrt(np.mean((func(xdata, *popt)-ydata)**2))
# print(" ".join([str(r2),str(nrmse),str(rmse)]))

#%%Toggle VM Mean vs Stiffness # define target  variable <<<<<<<<<<<<<<<<<<<<

roi_vm_mean =  roi_vm_mean  # define target  variable
# roi_vm_mean = roi_stiffness
# roi_vm_mean = roi_vm_90p

def func(x, a, b):
    return a * x + b

# def func(x, a, b):
#     return a * np.exp(-b * x)

#%%

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

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_validate

# remove MIN
features = np.delete(features,10,axis=1)
feature_names.pop(10)
feature_types.pop(10)

# Normalization
features_norm = features.copy()
features_norm -= np.mean(features,axis=0) # center on mean
features_norm /= np.std(features,axis=0) # scale to standard deviation

import matplotlib as mpl
from matplotlib.colors import ListedColormap
# https://stackoverflow.com/questions/37902459/seaborn-color-palette-as-matplotlib-colormap

FIGSIZE = (13,10)

cmap = sns.diverging_palette(240, 10, n=21)
cmap = ListedColormap(cmap.as_hex())

plt.figure(figsize=FIGSIZE)
plt.imshow(features_norm.T,cmap=cmap,aspect='auto')
plt.clim(-2,2)
plt.xticks([])
plt.yticks(np.arange(92),labels=feature_names,fontsize=8)
plt.gca().yaxis.tick_right()
plt.tight_layout()
# plt.ylabel()

#%% sklearn Clustering

# from sklearn.cluster import AgglomerativeClustering

# clustering = AgglomerativeClustering(n_clusters=3, 
#                                      affinity = 'correlation',
#                                      linkage = 'average').fit(features_norm)

#%% Manually compute clusters and pass intofigure
from scipy.spatial import distance
from scipy.cluster import hierarchy

# cmap = sns.diverging_palette(240, 10, n=21)

# correlations_array = np.corrcoef(features_norm)

# row_linkage = hierarchy.linkage(
#     distance.pdist(correlations_array), method='average')

# col_linkage = hierarchy.linkage(
#     distance.pdist(correlations_array.T), method='average')

# sns.clustermap(correlations, row_linkage=row_linkage, col_linkage=col_linkage, row_colors=network_colors, vmin=-3, vmax=3, method="average",
#                col_colors=network_colors, figsize=(13, 13), cmap=cmap)

#%% Visualize with seaborn clustermap

import seaborn as sns
cmap = sns.diverging_palette(240, 10, n=21)
g = sns.clustermap(features_norm.T, metric = 'correlation', cmap = cmap, vmin=-3, vmax=3, cbar=False)
ax = g.ax_heatmap
ax.set_axis_off()
ax.set_xlabel("")
ax.set_ylabel("")
# ax.update()

# plt.savefig('Figure_Dendrogram.png',
#             dpi=150, figsize=(8,8))

plt.figure(figsize=(7.31,2.45))
plt.plot(roi_vm_mean[g.dendrogram_col.reordered_ind],'ko',markersize=4)
plt.xlim(0,416)
plt.ylim(0,0.22)

#%% Plot example images from each fold

# roi_dir = '/gpfs_projects/qian.cao/data/L1_DICOM/isodata_transformed/rois_v2_mp/'
proj_dir = roi_dir + 'proj-2_v2_mp/'
filename = proj_dir+'pyradiomics_settings_all_projs_mask.csv'

out_proj_dir = roi_dir+'spie-roi-out-2-skeleton/'
if not os.path.isdir(out_proj_dir):
    os.mkdir(out_proj_dir)

df = pd.read_csv(filename)

pind = g.dendrogram_col.reordered_ind
plist = list(df['Image'])

avox = 0.05 # mm
rho = 1.85/1000 # g/cm3 * cm3/1000mm3

# for ii in range(len(plist)):
#     print(str(ii))
#     img, _ = nrrd.read(plist[pind[ii]])
#     plt.figure()
#     plt.imshow(img.T/255*rho*(avox**3)/(avox**2),cmap='gray')
#     plt.clim(0,0.006)
#     plt.title(str(ii))
#     plt.savefig(out_proj_dir+str(ii)+'_'+str(int(roi_vm_mean[pind[ii]]*1000))+'.png')
#     plt.close('all')

#%% ANOVA on clusters
from scipy.cluster import hierarchy
ftree = hierarchy.to_tree(g.dendrogram_col.linkage)

# dend = hierarchy.dendrogram(g.dendrogram_col.linkage, color_threshold = 0.8)
# dend.get('color_list')
# dend.get('icoord')

ind_c1 = ftree.left.left.pre_order()
ind_c2 = ftree.left.right.pre_order()
ind_c3 = ftree.right.left.pre_order()
ind_c4 = ftree.right.right.pre_order()

c1 = roi_vm_mean[ind_c1]
c2 = roi_vm_mean[ind_c2]
c3 = roi_vm_mean[ind_c3]
c4 = roi_vm_mean[ind_c4]

import scipy.stats as stats

stat, pval = stats.f_oneway(c1, c2, c3, c4)

#%%

# Save Feature Norm

# style.use('dark_background')
# plt.figure()
f, ax = plt.subplots(figsize=(11*5, 9*5))
plt.imshow(features_norm.T,aspect='auto')
plt.yticks(np.arange(len(feature_names)),feature_names,fontsize=4)
plt.colorbar()
plt.clim(-2,2)
plt.show()
plt.savefig(out_dir+'Figure_NormalizedFeatures.png')
plt.close()

#%% Random Forest Grid Search using 5-fold cross validation

plt.close('all')

import random

random.seed(1234)

# non-linear without feature selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'max_depth': [2,4,8,16,32,64,128,256], # 16
        'max_leaf_nodes': [2,4,8,16,32,64,128,256], # 8
        'n_estimators': [10,50,100,150,200]} # 50
            ]

rfr = GridSearchCV(
        RandomForestRegressor(), 
        param_grid, cv = 5,
        scoring = 'explained_variance',
        n_jobs=-1
        )

grid_result = rfr.fit(features_norm, roi_vm_mean)
yTrain_fit_rfr = rfr.predict(features_norm)

rfr_params = {'max_depth': rfr.best_estimator_.max_depth,
              'max_leaf_nodes': rfr.best_estimator_.max_leaf_nodes,
              'n_estimators': rfr.best_estimator_.n_estimators}

rfr_params = {'max_depth': 64,
              'max_leaf_nodes': 128,
              'n_estimators': 150}

print(rfr_params)

plt.figure()
plt.plot(roi_vm_mean,yTrain_fit_rfr,'ko')

# Plot feature importance

importances = rfr.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in rfr.best_estimator_], axis = 0)
plt.figure()
plt.title('Feature importances')
plt.barh(range(20), importances[indices[0:20]], yerr = std[indices[0:20]], align = 'center',log=True)
plt.yticks(range(20), list( feature_names[i] for i in indices[0:20] ), rotation=0)
plt.gca().invert_yaxis()
plt.show()
# print('Leading features:', feature_names[indices[0:5]])

#%% Random Forest Regression - Cross Validate on Final Model

ProjectionsPerBone = 13*2

roi_vm_mean_tests = np.empty((16,ProjectionsPerBone))
roi_vm_mean_preds = np.empty((16,ProjectionsPerBone))

roi_vm_mean_tests0 = np.empty((16,ProjectionsPerBone))
roi_vm_mean_preds0 = np.empty((16,ProjectionsPerBone))

fits = np.empty((16,2))
fitps = np.empty((16,2))

ccs = np.empty((16,1))
ccs0 = np.empty((16,1))

nrmses = np.empty((16,1))
nrmses0 = np.empty((16,1))

# nrmses_fit = np.empty((16,18))

nrmses_train = np.empty((16,1))
nrmses_const = np.empty((16,1))

rfs = np.empty((16,rfr_params['n_estimators']),dtype = RandomForestRegressor)

pval = [0,np.max(roi_vm_mean)]

imps = np.empty((16,92))

train_scores = np.empty((16,1))

for bb in range(16): # 16 bones total

    features_norm_test, roi_vm_mean_test = features_norm[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone,:], roi_vm_mean[ProjectionsPerBone*bb:ProjectionsPerBone*bb+ProjectionsPerBone]
    features_norm_train, roi_vm_mean_train = features_norm.copy(), roi_vm_mean.copy()
    features_norm_train = np.delete(features_norm_train,slice(ProjectionsPerBone*bb,ProjectionsPerBone*bb+ProjectionsPerBone),0)
    roi_vm_mean_train = np.delete(roi_vm_mean_train,slice(ProjectionsPerBone*bb,ProjectionsPerBone*bb+ProjectionsPerBone),0)
    
    rf = RandomForestRegressor(**rfr_params, n_jobs = -1,random_state =1)
    rf.fit(features_norm_train, roi_vm_mean_train)
    roi_vm_mean_pred = rf.predict(features_norm_test)
    roi_vm_mean_train_pred = rf.predict(features_norm_train)
    
    # rf.score(features_norm_train, roi_vm_mean_train)
    # rf.score(features_norm_test, roi_vm_mean_pred)

    nrmses[bb,:] = np.sqrt(np.mean((roi_vm_mean_pred-roi_vm_mean_test)**2))/np.max(roi_vm_mean_test)
    
    nrmses_train[bb,:] = np.sqrt(np.mean((roi_vm_mean_train_pred-roi_vm_mean_train)**2))/np.max(roi_vm_mean_train)
    
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
    
    #% Plot training set for proportional bias
    
    # fit = np.polyfit(roi_vm_mean_train,roi_vm_mean_train_pred,1)
    # fitp = np.polyval(fit,pval)
    
    # plt.figure()
    # plt.plot(roi_vm_mean_train,roi_vm_mean_train_pred,'ko')
    # plt.plot(pval,fitp,'k--')
    # plt.plot(pval,pval,'k-')
    # plt.xlim(1500,21000)
    # plt.ylim(1500,21000)
    # plt.xlabel('True Von Mises Stress')
    # plt.ylabel('Predicted Von Mises Stress')
    # plt.title('Training Error Fold' + str(bb))
    
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
    
    # # feature importance
    imps[bb,:] = rf.feature_importances_
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

#%% Plot Feature Importances

type_list = list(set(feature_types))
type_list.insert(0, type_list.pop(type_list.index("FirstOrder")))

color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'][:6]
color_dict = dict(zip(type_list, color_list))

mean_imps = np.mean(imps,axis=0)
std_imps = np.std(imps,axis=0)
indices = np.argsort(mean_imps)[::-1]

numFeatures = 20
color_name_list = list(feature_types[i] for i in indices[0:numFeatures])

plt.figure(figsize=[13,15])
ax = plt.barh(range(numFeatures), mean_imps[indices[0:numFeatures]], xerr = std_imps[indices[0:numFeatures]], align = 'center')
for ind, bar in enumerate(ax):
    bar.set_color(color_dict[color_name_list[ind]])
plt.yticks(range(numFeatures), list(feature_names_notype[i] for i in indices[0:numFeatures]), rotation=0, fontsize=15)
# plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=14)
# plt.title('Overall Feature Importances')
plt.gca().invert_yaxis()
plt.gca().set_xscale('log')
plt.gcf().subplots_adjust(left=0.5)
# ax.set_xscale('log')
plt.show()

# Look at contributions from each category
gini_imps = mean_imps[indices[0:numFeatures]]
for ii, ty in enumerate(type_list):
    mask = np.array([cc==type_list[ii] for cc in color_name_list])
    masked_array = np.ma.array(gini_imps, mask=~mask)
    print(type_list[ii]+" "+str(np.sum(masked_array)))

#%%

# pval = [0,np.max(roi_vm_mean_tests)]

# Scatter Plot
plt.figure()
for bb in range(16):
    plt.plot(roi_vm_mean_tests[bb,:],roi_vm_mean_preds[bb,:],'o')
    # plt.plot(pval,fitp,'k--')
plt.plot(pval,fitp,'k--')
plt.plot(pval,pval,'k-')
plt.xlim(0,np.max(pval))
plt.ylim(0,np.max(pval))

#%%

plt.figure()
for bb in range(16):
    plt.plot(roi_vm_mean_tests0[bb,:],roi_vm_mean_preds0[bb,:],'o')
    # plt.plot(pval,fitp,'k--')
plt.plot(pval,fitp0,'k--')
plt.plot(pval,pval,'k-')
plt.xlim(0,np.max(pval))
plt.ylim(0,np.max(pval))

print('correlation coefs and nrmses for Radiomics')
print('Mean of all folds ' + str(np.mean(ccs**2)))
print('STD of all folds ' + str(np.std(ccs**2)))
# print('All folds ' + str(cc**2))
print('NRMSE ' + str(np.mean(nrmses)))

print('correlation coefs and nrmses for Exponential Fit')
print('Mean of all folds ' + str(np.mean(ccs0**2)))
print('STD of all folds ' + str(np.std(ccs0**2)))
# print('All folds ' + str(cc**2))
print('NRMSE ' + str(np.mean(nrmses0)))

# plt.xlabel('True Stress Amplification')
# plt.ylabel('Predicted Stress Amplification')

# fitr = np.polyfit(roi_vm_mean_preds.flatten(),roi_vm_mean_tests.flatten(),1) # reverse fit

# def fitr1d(fit,x):
#     return (x-fit[1])/fit[0]

# # Scatter Plot with FIT
# plt.figure()
# for bb in range(16):
#     plt.plot(roi_vm_mean_tests[bb,:],fitr1d(fit,roi_vm_mean_preds[bb,:]),'o')
#     # plt.plot(pval,fitp,'k--')
# # plt.plot(pval,fitp,'k--')
# plt.plot(pval,pval,'k-')
# plt.xlim(1500,21000)s
# plt.ylim(1500,21000)
# plt.xlabel('True Von Mises Stress')
# plt.ylabel('Predicted Von Mises Stress')

#%% Normalized RMSE
    
A = roi_vm_mean_tests.flatten()
B = roi_vm_mean_preds.flatten()

nrmse = np.sqrt(np.mean(((B-A)/np.mean(A))**2))
    
#%%

# # Bland-Altman in Percentages
# mean_diff = np.mean(roi_vm_mean_tests.flatten() - roi_vm_mean_preds.flatten())
# sd_diff = np.std(roi_vm_mean_tests.flatten() - roi_vm_mean_preds.flatten())

# plt.figure()
# for bb in range(16):
#     A = roi_vm_mean_tests[bb,:]
#     B = roi_vm_mean_preds[bb,:]
#     plt.plot(A,(B-A)/A*100,'o')
# plt.plot(pval,np.array([mean_diff,mean_diff])/np.mean(roi_vm_mean_tests.flatten())*100,'k-')
# plt.plot(pval,(np.array([mean_diff,mean_diff])+1.96*sd_diff)/np.mean(roi_vm_mean_tests.flatten())*100,'k--')
# plt.plot(pval,(np.array([mean_diff,mean_diff])-1.96*sd_diff)/np.mean(roi_vm_mean_tests.flatten())*100,'k--')
# plt.xlabel('True Von Mises Stress')
# plt.ylabel('Percentage Difference')
# plt.xlim(1500,21000)
# plt.ylim(-80,80)

# Bland-Altman with FIT
# mean_diff = np.mean(roi_vm_mean_tests.flatten() - fitr1d(fit,roi_vm_mean_preds.flatten()))
# sd_diff = np.std(roi_vm_mean_tests.flatten() - fitr1d(fit,roi_vm_mean_preds.flatten()))

# plt.figure()
# for bb in range(16):
#     A = roi_vm_mean_tests[bb,:]
#     B = fitr1d(fit,roi_vm_mean_preds[bb,:])
#     plt.plot((B+A)/2,B-A,'o')
# plt.plot(pval,[mean_diff,mean_diff],'k-')
# plt.plot(pval,np.array([mean_diff,mean_diff])+1.96*sd_diff,'k--')
# plt.plot(pval,np.array([mean_diff,mean_diff])-1.96*sd_diff,'k--')
# plt.xlabel('Average, True vs Estimate')
# plt.ylabel('Difference, True vs Estimate')
# plt.xlim(1500,21000)