#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qcao
"""

import numpy
import glob
import os

import matplotlib.pyplot as plt

# inputs
segDir = "/data/Segmentations/"

# outputs
outDir = "/data/Segmentations-out/"
os.mkdirs(outDir)

# list of image names
imgNames = []

tmpFiles = glob.glob(segDir+"Normals-*.txt") # get only annotated nrrd files
for ind, tmp in enumerate(tmpFiles):
    name = os.path.split(tmp)[1]
    imgNames.append(name.lstrip("Normals-").rstrip(".txt"))
    
