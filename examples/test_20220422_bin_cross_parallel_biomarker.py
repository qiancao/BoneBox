#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:55:31 2022

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
    
    