# -*- coding: utf-8 -*-

"""
 partially taken from:
     https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/batchprocessing_parallel.py
     
"""

from __future__ import print_function

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

def getRadiomicFeatureNames(settings=None):
    """
    Generates a list of feature names for computeRadiomicFeatures

    Returns
    -------
    featureNames : list of strings
        features names

    """
    
    #
    # TODO: there must a better way of doing this
    #
    # see example_match_glcm_20211103
    
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    if settings is None:
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        settings['interpolator'] = sitk.sitkBSpline
        settings['imageType'] = ['original','wavelet']
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  
    
    # Extract radiomics from volume
    volume = np.random.rand(3,3,3)*256
    volumeSITK = sitk.GetImageFromArray(volume)
    maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
    featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    # featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    featureNames = list(featureVector.keys())
    
    return featureNames

def computeRadiomicFeatures(volume, settings=None):
    
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    if settings is None:
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        settings['interpolator'] = sitk.sitkBSpline
        settings['imageType'] = ['original','wavelet']
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  
    
    # Extract radiomics from volume
    volumeSITK = sitk.GetImageFromArray(volume)
    maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
    featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    
    return featureVectorArray