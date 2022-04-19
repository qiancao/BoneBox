# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:38:49 2022

@author: Qian.Cao
"""

import nrrd
import numpy as np
import os, sys

voxelSize = (0.05, 0.05, 0.05) # mm

filenameNRRD = "../data/rois/isodata_04216_roi_4.nrrd"
roiBone, header = nrrd.read(filenameNRRD)

