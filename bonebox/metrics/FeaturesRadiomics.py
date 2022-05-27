# -*- coding: utf-8 -*-

"""

Compute radiomic features from images

TODO: https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/batchprocessing_parallel.py

"""

from __future__ import print_function

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import functools

def getDefaultSettings():
    # Default radiomic settings
    # See: https://pyradiomics.readthedocs.io/en/latest/customization.html
    
    settings = {}
    # settings['binWidth'] = 25
    # settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    # settings['interpolator'] = sitk.sitkBSpline
    # settings['imageType'] = {'Original': []}
    settings['featureClass'] = {'firstorder': [],
                                'glcm': [],
                                'glrlm': [],
                                'glszm': [],
                                'gldm': [],
                                'ngtdm': []} # shape is not included
    # settings['verbose'] = True
    
    return settings

def getRadiomicFeatureNames(settings=None):
    # Check which features are computed based on the settings dict
    #
    # TODO: there must a better way of doing this than allocating a test image
    #
    # see example_match_glcm_20211103
    #
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    if settings is None:
        settings = getDefaultSettings()
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
    
    # Extract radiomics from volume
    volume = np.random.rand(3,3,3)*256
    volumeSITK = sitk.GetImageFromArray(volume)
    mask = np.ones(volume.shape).astype(int)
    mask[0,0,0] = 0 # TODO: this is a temporary fix https://github.com/AIM-Harvard/pyradiomics/issues/765
    maskSITK = sitk.GetImageFromArray(mask)
    
    # featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    featureVector = extractor.execute(volumeSITK, maskSITK, label=1)
    # featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    
    featureNames = [feat for feat in featureVector if "diagnostics_" not in feat] # TODO: output diagnostics too. remove diagnostic entries, leave only features
    # featureNames = list(featureVector.keys())
    
    return featureNames

def computeRadiomicFeatures(volume, settings=None):
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    
    if settings is None:
        settings = getDefaultSettings()
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)  
    
    # Extract radiomics from volume
    volumeSITK = sitk.GetImageFromArray(volume)
    mask = np.ones(volume.shape).astype(int)
    mask[0,0,0] = 0 # TODO: this is a temporary fix https://github.com/AIM-Harvard/pyradiomics/issues/765
    maskSITK = sitk.GetImageFromArray(mask)
    
    # featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    featureVector = extractor.execute(volumeSITK, maskSITK)
    featureVector = {key: value for key, value in featureVector.items() if "diagnostics_" not in key} # TODO: output diagnostics too. remove diagnostic entries, leave only features
    featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    featureNamesList = [featureName for featureName in featureVector.keys()]
    
    return featureNamesList, featureVectorArray

def computeRadiomicFeaturesParallel(volumeList, settings=None, numWorkers=None):
    # Extract radiomics features from a list of volumes using the same settings.
    # https://stackoverflow.com/questions/60116458/multiprocessing-pool-map-attributeerror-cant-pickle-local-object
    #
    # Note: The output featureVectorArray is in (Nvolumes, Nfeatures)
    
    from multiprocessing import cpu_count, Pool
    
    if numWorkers is None:
        numWorkers = cpu_count() - 2
    
    featureNamesList = getRadiomicFeatureNames(settings)
    numFeatures = len(featureNamesList)
    numVolumes = len(volumeList)
    featureVectorArray = np.zeros((numFeatures,numVolumes))

    with Pool(numWorkers) as pool:
        results = pool.map(functools.partial(computeRadiomicFeatures, settings=settings),volumeList)
    
    featureVectors = [x[1] for x in results]
    featureVectorArray = np.vstack(featureVectors)

    return featureNamesList, featureVectorArray

def getNameByString(nameString,featureNames):
    # Returns index and names of features from featureNames (list) containing nameString
    inds = []
    names = []
    for ind, name in enumerate(featureNames):
        if f"{nameString}" in name:
            inds.append(ind)
            names.append(name)
    
    return inds, names

def getFeaturesByName(nameList,featureNames,featureMatrix):
    # assumes the dimensions of featureMatrix to be:
    #    (samples,features,...)
    
    names = []
    features = []
    
    # convert to list if className is a string
    if isinstance(nameList,str):
        nameList = [nameList]
    
    if isinstance(nameList,list):
        for cn in nameList:
            ii, nn = getNameByString(cn,featureNames)
            names.extend(nn)
            features.append(np.take(featureMatrix,ii,axis=1))
    
    if len(features)>1:
        features = np.concatenate(features,axis=1)
    else:
        features = features[0]
    
    return names, features

if __name__ == "__main__":
    
    import glob, nrrd
    
    # roi data folder
    roiDir = "../../data/rois/"
    def getROI(number):
        filenameNRRD = glob.glob(roiDir+f"*_roi_{number}.nrrd")[0]
        roi, header = nrrd.read(filenameNRRD)
        return roi
    
    # load 20 rois and compute radiomic features in parallel
    volumeList = []
    for ind in range(20):
        volumeList.append(getROI(ind))
        
    featureNamesList, featureVectorArray = computeRadiomicFeaturesParallel(volumeList)