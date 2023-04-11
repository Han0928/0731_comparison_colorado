import iris
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy.ma as ma


# new updated for p-3b
def merge_obs_method():
    time_NAV = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140731_r0.ict',
                          delimiter=',',
                          skiprows=73, usecols=0)  # (17546,)
    print(time_NAV.shape)  # (17546,)
    temp = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140731_r0.ict', delimiter=',',
                      skiprows=73, usecols=18)  # inC  (17546,)
    print('temp', temp)
    pres = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140731_r0.ict', delimiter=',',
                      skiprows=73, usecols=24)  # in hpa  (17546,)
    print('pres', pres)
    print('temp.shape', temp.shape)
    print('pres.shape', pres.shape)

    # load >100nm UHSAS data for acc+coar
    file_100 = Dataset(
        '/jet/home/ding0928/Colorado/EOLdata/dataaw2NKX/RF05Z.20140731.195400_011100.PNI.nc')
    print('file_100', file_100)

    uhsas_time = (file_100.variables['Time'][:])  # we need time to interpolate?
    print(uhsas_time, 'uhsas_time')  # (19021,)

    cut_off_size_100 = [0.097, 0.105, 0.113, 0.121, 0.129, 0.145, 0.162, 0.182, 0.202, 0.222, 0.242, 0.262,
                        0.282, 0.302, 0.401, 0.57, 0.656, 0.74, 0.833, 0.917, 1.008, 1.148, 1.319, 1.479,
                        1.636, 1.796, 1.955, 2.184, 2.413, 2.661, 2.991]  # 31bin in um
    cut_off_size_100 = np.multiply(cut_off_size_100, 1e-6)

    data_100 = (file_100.variables['CS200_RPI'][:, 0, :])  # ('data_100.shape', (19021, 31))
    print('data_100', data_100)
    print('data_100.shape', data_100.shape)  # ('data_100.shape', (19021, 31))

    # ????? test here whether it's correct??
    # temp = np.concatenate((temp, [0]))  # to go from 19020 to 19021 since data_100 is 19021
    # pres = np.concatenate((pres, [0]))
    # time_NAV_2 = np.concatenate((time_NAV, [0]))

    # temp.shape = 19021, 1
    # pres.shape = 19021, 1
    # time_NAV.shape = 19021, 1

    term1 = np.multiply((temp + 273.15), 1013.25)
    term2 = np.multiply(pres, 293.15)
    print('term2_shape', term2.shape)

    data_100 = ma.masked_where(data_100 < 0, data_100)
    uhsas_time2 = uhsas_time[~data_100[:, 0].mask]
    print('uhsas_time2.shape', uhsas_time2.shape)
    # print('time_NAV_2shape', time_NAV_2.shape)

    interp_time = interp1d(time_NAV, time_NAV, kind='nearest', fill_value='extrapolate')
    time_track = interp_time(np.asarray(uhsas_time))
    print('time_track', time_track)

    interp_term1 = interp1d(time_NAV, term1, kind='nearest', fill_value='extrapolate')
    term1_track = interp_term1(np.asarray(uhsas_time))
    print('term1_track', term1_track)

    interp_term2 = interp1d(time_NAV, term2, kind='nearest', fill_value='extrapolate')
    term2_track = interp_term2(np.asarray(uhsas_time))
    print('term2_track', term2_track)

    gas_track_data2 = []
    interp_gas = interp1d(time_NAV, term2, kind='nearest', fill_value='extrapolate')
    gas_track = interp_gas(np.asarray(uhsas_time))  # term2:17546 interpreted to term2:19021
    gas_track_data2.append(gas_track)

    interp_temp = interp1d(time_NAV, temp, kind='nearest', fill_value='extrapolate')
    temp_track = interp_temp(np.asarray(uhsas_time))
    print('temp_track', temp_track)
    print('temp_track.shape', temp_track.shape)

    interp_pres = interp1d(time_NAV, pres, kind='nearest', fill_value='extrapolate')
    pres_track = interp_pres(np.asarray(uhsas_time))
    print('pres_track.shape', pres_track.shape)

    # data_100_new = ma.mask_rowcols(data_100, axis=0)
    data_100_new2 = data_100[~data_100.mask.any(axis=1)]
    print('data_100_new2.shape', data_100_new2.shape)  # (19000, 31) which means 21 points is masked out
    # data_100_new = data_100_new[~data_100_new.mask]

    time_NAV_2 = (time_track[~(data_100.mask.any(axis=1))])
    temp_2 = (temp_track[~(data_100.mask.any(axis=1))])
    # temp_2 = temp[~data_100_new.mask]  # masked t
    pres_2 = (pres_track[~(data_100.mask.any(axis=1))])

    term1 = (term1_track[~(data_100.mask.any(axis=1))])
    term2 = (term2_track[~(data_100.mask.any(axis=1))])

    print('temp_2.shape', temp_2.shape)
    print('pres_2.shape', pres_2.shape)
    print('term1.shape', term1.shape)
    print('term2.shape', term2.shape)
    print('time_NAV_2.shape', time_NAV_2.shape)
    print('data_100_new2.shape', data_100_new2.shape)


merge_obs_method()
