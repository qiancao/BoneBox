# -*- coding: utf-8 -*-

"""

Compute radiomic features from images

TODO: https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/batchprocessing_parallel.py

"""

from __future__ import print_function

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

def getDefaultSettings():
    # Default radiomic settings
    # See: https://pyradiomics.readthedocs.io/en/latest/customization.html
    
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    settings['imageType'] = ['original']
    
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
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    # Extract radiomics from volume
    volume = np.random.rand(3,3,3)*256
    volumeSITK = sitk.GetImageFromArray(volume)
    mask = np.ones(volume.shape).astype(int)
    mask[0,0,0] = 0 # TODO: this is a temporary fix https://github.com/AIM-Harvard/pyradiomics/issues/765
    maskSITK = sitk.GetImageFromArray(mask)
    
    # featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    featureVector = extractor.execute(volumeSITK, maskSITK, label=1)
    # featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    featureNames = list(featureVector.keys())
    
    return featureNames

def computeRadiomicFeatures(volume, settings=None):
    
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    if settings is None:
        settings = getDefaultSettings()
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  
    
    # Extract radiomics from volume
    volumeSITK = sitk.GetImageFromArray(volume)
    mask = np.ones(volume.shape).astype(int)
    mask[0,0,0] = 0 # TODO: this is a temporary fix https://github.com/AIM-Harvard/pyradiomics/issues/765
    maskSITK = sitk.GetImageFromArray(mask)
    
    # featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    featureVector = extractor.execute(volumeSITK, maskSITK)
    featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    featureNamesList = [featureName for featureName in featureVector.keys()]
    
    return featureNamesList, featureVectorArray