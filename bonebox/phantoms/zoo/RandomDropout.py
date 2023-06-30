# -*- coding: utf-8 -*-
"""

Create a simple trabecular bone phantom

"""

import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv

import networkx as nx

from bonebox.phantoms import TrabeculaePhantom, MedialAxisUtils, PVUtils, MeshUtils

def generate(volume_extent = [3,]*3,
            volume_ndim = (120,)*3,
            scaffold_k = 30,
            scaffold_radius = 0.4,
            faces_init_fraction = 0.1,
            edges_init_fraction = 0.3,
            mean_TbTh = 0.08,
            std_TbTh = 0.01,
            scaffold_seed = None,
            faces_seed = None,
            edges_seed = None,
            TbTh_seed = None):
    """
    

    Parameters
    ----------
    volume_extent : TYPE, optional
        DESCRIPTION. The default is [3,]*3.
    volume_ndim : TYPE, optional
        DESCRIPTION. The default is (120,)*3.
    scaffold_k : TYPE, optional
        DESCRIPTION. The default is 30.
    scaffold_radius : TYPE, optional
        DESCRIPTION. The default is 0.4.
    faces_init_fraction : TYPE, optional
        DESCRIPTION. The default is 0.1.
    edges_init_fraction : TYPE, optional
        DESCRIPTION. The default is 0.3.
    mean_TbTh : TYPE, optional
        DESCRIPTION. The default is 0.08.
    std_TbTh : TYPE, optional
        DESCRIPTION. The default is 0.01.
    scaffold_seed : TYPE, optional
        DESCRIPTION. The default is None.
    faces_seed : TYPE, optional
        DESCRIPTION. The default is None.
    edges_seed : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    v0, e0, f0, i0 = TrabeculaePhantom.generatePoissonScaffold(volume_extent=volume_extent,
                                                               radius=scaffold_radius,
                                                               k=scaffold_k,
                                                               seed=scaffold_seed,
                                                               centered=False,
                                                               init=None)
    
    #### Generate Faces
    
    faces_select_rng = np.random.default_rng(faces_seed)
    
    faces_probability = faces_select_rng.uniform(low=0,high=1,size=len(f0))
    faces_mask = faces_probability < faces_init_fraction
    
    #### Generate Edges
    
    edges_select_rng = np.random.default_rng(edges_seed)
    
    edges_probability = edges_select_rng.uniform(low=0,high=1,size=len(e0))
    edges_mask = edges_probability < edges_init_fraction
    
    #### Generate Radius
    
    TbTh_rng = np.random.default_rng(TbTh_seed)
    TbTh = TbTh_rng.normal(loc=mean_TbTh, scale=std_TbTh, size=len(v0))
    TbTh[TbTh<0] = 0 # non-neg constraint

    print(f"Number of plates {np.sum(faces_mask)}")
    print(f"Number of rods {np.sum(edges_mask)}")
    
    e_hat = e0[edges_mask]
    f_hat = f0[faces_mask]
    
    volume = TrabeculaePhantom.vref2volume(v0,TbTh,e_hat,f_hat,volume_extent,
                                           ndim=volume_ndim,
                                           origin=(0,0,0),
                                           theta_resolution=5,
                                           phi_resolution=5)
    
    return volume

if __name__ == "__main__": 
    pass