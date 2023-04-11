import iris
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy.ma as ma

cubes = iris.load('/jet/home/ding0928/cylc-run/u-cr977/share/cycle/20140729T0000Z/glm/um/umglaa_pc135')

Constraint_for_crii = iris.AttributeConstraint(STASH='m01s38i675') & iris.Constraint(model_level_number=1)
crii = cubes.extract(Constraint_for_crii)
data1 = crii[0].data
lat = crii[0].coord('latitude').points
lon = crii[0].coord('longitude').points

plt.pcolormesh(lon, lat, data1, vmin=1, vmax=40, cmap='magma_r')
# plt.pcolormesh(lon, lat, data1, cmap='magma_r')
plt.colorbar(label='38675 crii_offline', orientation="horizontal")

plt.title('CRII s-1', fontsize=14)  # fig 45
plt.show()
