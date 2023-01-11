# -*- coding: utf-8 -*-
"""

Utilities for Processing Medial Axis Meshes

@author: Qian.Cao

"""

import numpy as np


def readMAT(pathMAT):
    """
    Reads medial axis transform files generated from QMAT

    Parameters
    ----------
    pathMAT : str
        Path to medial axis file.
    clean_edges : bool, optional
        Removes edges already in faces array. The default is False.

    Returns
    -------
    vertices : np.array
        Vertex positions.
    radii : np.array
        Radius corresponding to each vertex.
    edges : np.array
        Vertex indices of edges.
    faces : np.array
        Vertex indices of faces.

    """

    # length of rows for vertices, edges, and faces
    lenV, lenE, lenF = 4, 2, 3

    with open(pathMAT,"r") as file:
        lines = file.read().splitlines()

    # reads first line for number of vertices, edges and faces
    nv, ne, nf = [int(x) for x in lines.pop(0).rstrip().split()]

    # splits lines into vertices, edges and faces
    lines_v, lines = lines[:nv], lines[nv:]
    lines_e, lines = lines[:ne], lines[ne:]
    lines_f, lines = lines[:nf], lines[nf:]

    # convert lines to numpy arrays
    lines2array = lambda lines, row_prefix, row_type: np.array("".join("".join(lines).split(row_prefix)[1:]).split(" ")[1:]).astype(row_type)
    vertices_radii = lines2array(lines_v,"v", np.double).reshape(nv,lenV)
    vertices, radii = vertices_radii[:,:3], vertices_radii[:,3]
    edges = lines2array(lines_e,"e", np.uint32).reshape(ne,lenE)
    faces = lines2array(lines_f,"f", np.uint32).reshape(nf,lenF)

    return vertices, radii, edges, faces

