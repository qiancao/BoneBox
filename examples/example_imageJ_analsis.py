#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:56:16 2021

@author: qcao
"""

import imagej
# from jnius import autoclass

import nrrd

# ij = imagej.init('sc.fiji:fiji:2.1.1')
ij = imagej.init('/home/qcao/Fiji.app')

roi_dir = "/data/BoneBox/data/rois/"
img_fn = "isodata_04216_roi_5.nrrd"

npimage, header = nrrd.read(roi_dir+img_fn)

jimage = ij.py.to_java(npimage)

ij.ui().showUI()
ij.ui().show(jimage)

# result = ij.py.run_plugin("Thickness")
# outputs = result.getOutputs()

# ijop = ij.op()

