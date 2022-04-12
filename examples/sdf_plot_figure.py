#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:12:58 2022

@author: qcao
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,21)
r = 2.5

plt.plot(x, np.abs(x),'b')
plt.plot(x, np.abs(x) - r, 'r')
plt.plot(x, np.zeros(x.shape),'k--')
plt.plot([0,0],[-5,10],'k--')
plt.xlabel("Spatial Coordinate")
plt.ylabel("SDF")
plt.axis_tight()