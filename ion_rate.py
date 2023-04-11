import iris
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy.ma as ma

file_ion = Dataset(
    '/ocean/projects/atm200005p/shared/um-inputs/ancil/atmos/n96e/cmip6-2014-greg/n96e/ion_production/v1'
    '/ukca_cosmic_ray_ionisation_rate.nc')
ion_con = (file_ion.variables['CRII'][:])
print('ion_con.shape', ion_con.shape) #(12, 85, 144, 192)

ion_lat = (file_ion.variables['latitude'][:])
print(np.max(ion_lat))  #89.375
print(np.min(ion_lat))  #-89.375
ion_lon = (file_ion.variables['longitude'][:])
print(np.max(ion_lon))  # 359.0625
print(np.min(ion_lon))  #0.9375
ion_lev = (file_ion.variables['level_height'][:])
print(np.max(ion_lev))  # 85000.0
print(ion_lev)  #
print(np.min(ion_lev))  #19.999998
ion_time = (file_ion.variables['time'][:])
print(np.max(ion_time))  #262584.0
print(np.min(ion_time))  #254568.0  totally 4h??

ion_time0 = (file_ion.variables['CRII'][3, 50, :, :])  # time, model_level,lat, lon
print(np.max(ion_time0))  #
print(np.min(ion_time0))  #

# fig = plt.figure()
# ax = fig.add_subplot(111)
# levels = np.array((1, 30, 5))
plt.pcolormesh(ion_lon, ion_lat, ion_time0,vmin=1, vmax=40, cmap='magma_r')
plt.colorbar(label='ion pairs s-1 ', orientation="horizontal")

plt.title('CRII s-1', fontsize=14)  # fig 45
# plt.xlim(-150, -100)
# plt.ylim(25, 45)
plt.show()
# plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/fig_qual/emission_low.png')
