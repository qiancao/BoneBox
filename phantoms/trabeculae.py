"""
phantoms.trabeculae

An analytical trabecular bone phantom

Authors: Qin Li, Nada Kamona, Xin Xu, Qian Cao

"""

import numpy as np

# Parameters for volume size
vol_params = {"sliceHeight": 36.,
              "radius0": 6.,
              "target_tbsp": 1.,
              }

# Parameters for Centroidal Voronoi Tessellation
numPoints = np.round(((vol_params["sliceHeight"] \
                       *(2*vol_params["radius0"])^2)^(1/3) \
                      /vol_params["target_tbsp"])^3) # N = (V^(1/3)/Sp)^3
cvt_params = {"generator_num": numPoints,
              "iteration_num": 50,
              "samplepoints_num": 5000,
              }