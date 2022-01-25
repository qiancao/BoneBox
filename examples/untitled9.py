# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:10:09 2022
@author: Qian.Cao

* install vtk==9.1.0 or else windows won't close

"""

import pyvista as pv
import numpy as np

grid = pv.UniformGrid(
    dims=(10, 10, 10),
    spacing=(1, 1, 1),
    origin=(-5, -5, -5),
)

vertices = np.array([[-4,-4,-4],[2,2,2]]).astype(float)
cells = np.array([[2,0,1]])

mesh = pv.PolyData(vertices, cells)
mesh.cell_data["values"] = 1.2

dargs = dict(cmap="coolwarm", clim=[0,1], scalars="values")

p = pv.Plotter()
p.add_mesh(grid.outline(), color='k')
p.add_mesh(mesh, render_points_as_spheres=True, **dargs)
p.show()

# # sample = grid.sample(mesh)
# interp = grid.interpolate(mesh)
# plotter = pv.Plotter()
# plotter.add_volume(interp)