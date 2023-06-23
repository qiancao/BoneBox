# -*- coding: utf-8 -*-
"""

Utilities for PyVista

@author: Qian.Cao

"""

import numpy as np
import pyvista as pv

def scalar2rgb(x,palette,vmin,vmax):
    
    import matplotlib as mpl
    from matplotlib import cm
    import seaborn as sns
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = sns.light_palette(palette, as_cmap=True)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = scalarMap.to_rgba(x)
    
    return colors

def formatPV(arr):
    """
    
    Convert array of edges of faces to pyvista format for use in pv.PolyData
    
    dim = 2 for edges
    dim = 3 for faces

    Parameters
    ----------
    arr
        array in (N, dim)

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
    
    if e is not None:
        e = formatPV(np.array(e))
        
    if f is not None:
        f = formatPV(np.array(f))
        
    pd = pv.PolyData(v, f, lines=e)
    
    return pd

def vef2plotter(v,e,f,add_mesh_opts=None):
    """
    Visualize vertices, edges, and faces using pyvista.Plotter

    Returns
    -------
    None.

    """
    
    pd = vef2pd(v, e, f)
    
    p = pv.Plotter()
    
    if add_mesh_opts is None:
        p.add_mesh(pd)
    else:
        p.add_mesh(pd,**add_mesh_opts)
        
    return p

