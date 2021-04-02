"""
phantoms.trabeculae

An analytical trabecular bone phantom

Authors: Qin Li, Nada Kamona, Xin Xu, Qian Cao

"""

import numpy as np
from trabeculae_utils import vol, Nseed

# Parameters for volume size
vol_params = {"sliceHeight": 36.,
              "radius0": 6.,
              "target_tbsp": 1.,
              }

# Parameters for Centroidal Voronoi Tessellation
numPoints = np.round(Nseed(vol(vol_params["sliceHeight"],vol_params["radius0"]), \
                           vol_params["target_tbsp"]))

cvt_params = {"generator_num": numPoints,
              "iteration_num": 50,
              "samplepoints_num": 5000,
              "randR": 0.4,
              "zoom_factor": 2.5,
              "KK_factor": 1.1,
              "z_xy_density_ratio": 0.7,
              }

def makeVertebralBone(generator_num,
                      iteration_num,
                      samplepoints_num,
                      randR,
                      zoom_factor,
                      KK_factor,
                      z_xy_density_ratio,
                      radius0,
                      sliceHeight):
    
    # Volume and per-seed length
    tmpV = vol(sliceHeight, radius0)
    delta = (tmpV/generator_num)**(1/3)
    
    # Number of layers in Z
    M =  np.round(sliceHeight/delta*z_xy_density_ratio)
    
    # Number of seed points in each plane * 1.1 (TODO: what is this 1.1 for?)
    KK = np.round(zoom_factor**KK_factor*generator_num/M*1.1/z_xy_density_ratio)
    
    
    
    return V, C, idx_bounded, pointsCoordinates