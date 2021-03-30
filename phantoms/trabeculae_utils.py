# -*- coding: utf-8 -*-

"""
phantoms.trabeculae_utils

An analytical trabecular bone phantom.

Utility functions for trabecular phantom code.

Authors: Qin Li, Nada Kamona, Xin Xu, Qian Cao

"""

# Compute rectangular VOI volume from height and radius
def vol(sliceHeight, radius0):
    return sliceHeight*(2*radius0)**2

# Compute number of seed points from volume and trabecular spacing
def Nseed(volume, spacing):
    return ((volume**(1/3))/spacing)**3