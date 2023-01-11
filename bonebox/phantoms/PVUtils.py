# -*- coding: utf-8 -*-
"""

Utilities for PyVista

@author: Qian.Cao

"""

import numpy as np
import pyvista as pv

def formatPV(arr):
    """
    
    Convert array of edges of faces to pyvista format for use in pv.PolyData
    
    arr: array in (N, dim)
    
    dim = 2 for edges
    dim = 3 for faces

    """
    
    N, dim = arr.shape
    
    pv_arr = np.hstack(np.concatenate((np.ones((N, 1),dtype=np.uint32)*dim, arr),axis=1).tolist())
    
    return pv_arr

def vef2pd(v,e,f):
    """
    Convert vertices, edges, and faces to pv.PolyData

    Returns
    -------
    None.

    """
    
    e = formatPV(np.array(e))
    f = formatPV(np.array(f))
    pd = pv.PolyData(v, f, lines=e)
    
    return pd

def vef2plotter(v,e,f):
    """
    Visualize vertices, edges, and faces using pyvista.Plotter

    Returns
    -------
    None.

    """
    
    pd = vef2pd(v, e, f)
    
    p = pv.Plotter()
    p.add_mesh(pd)
    
    return p

