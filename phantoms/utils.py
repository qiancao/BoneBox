# -*- coding: utf-8 -*-

"""
Phantoms.utils

Utilities module

Authors:  Qian Cao, Xin Xu, Nada Kamona, Qin Li

"""

import numpy as np

def Polar2CartesianEllipsoid(Phi, Lambda, r, h, N):
    """
    Sample (x,y,z) from r=(Phi, Lambda, h)
    
    https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversions

    Note that this is NOT a uniform sampling of the ellipsoid.

    Parameters
    ----------
    Phi : np.ndarray
        Angle from X in XY.
    Lambda : np.ndarray
        Angle from XY plane.
    h : np.ndarray
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    Pxyz : points in XYZ

    """
    raise(NotImplemented,"Currently supports sampleSphere")
    
def Polar2CartesianSphere(r, Theta, Phi):
    """
    Sample from sphere solid.
    
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    
    Note that this is NOT a uniform sampling of the ellipsoid.

    Parameters
    ----------
    r : np.ndarray (1,) floats
        Distance from origin.
    Theta : np.ndarray (1,) floats
        Azimuthal angle in radians.
    Phi : np.ndarray (1,) floats
        Polar angle in radians.

    Returns
    -------
    np.ndarray[N,3]

    """
    
    assert r.ndim == 1
    assert Theta.ndim == 1
    assert Phi.ndim == 1
    
    x = r*np.sin(Theta)*np.cos(Phi)
    y = r*np.sin(Theta)*np.sin(Phi)
    z = r*np.cos(Theta)
    
    return np.array([x,y,z]).T

def MinMaxNorm(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)