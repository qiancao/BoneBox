#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:04:53 2022

@author: qcao
"""

import pyvista as pv
from pyvista import examples

# Download the sparse data
probes = examples.download_thermal_probes()

grid = pv.UniformGrid()
grid.origin = (329700, 4252600, -2700)
grid.spacing = (250, 250, 50)
grid.dimensions = (60, 75, 100)

dargs = dict(cmap="coolwarm", clim=[0,300], scalars="temperature (C)")
cpos = [(364280.5723737897, 4285326.164400684, 14093.431895014139),
 (337748.7217949739, 4261154.45054595, -637.1092549935128),
 (-0.29629216102673206, -0.23840196609932093, 0.9248651025279784)]

p = pv.Plotter()
p.add_mesh(grid.outline(), color='k')
p.add_mesh(probes, render_points_as_spheres=True, **dargs)
p.show(cpos=cpos)

interp = grid.interpolate(probes, radius=15000, sharpness=10, strategy='mask_points')

vol_opac = [0, 0, .2, 0.2, 0.5, 0.5]

p = pv.Plotter(shape=(1,2), window_size=[1024*3, 768*2])
p.add_volume(interp, opacity=vol_opac, **dargs)
p.add_mesh(probes, render_points_as_spheres=True, point_size=10, **dargs)
p.subplot(0,1)
p.add_mesh(interp.contour(5), opacity=0.5, **dargs)
p.add_mesh(probes, render_points_as_spheres=True, point_size=10, **dargs)
p.link_views()
p.show(cpos=cpos)