import pandas as pd
import numpy as np
import math  # for e^x
from netCDF4 import Dataset
import numpy.ma as ma
from scipy.interpolate import interp1d, RegularGridInterpolator
from numpy.ma import exp
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from box_model_rerun_step4 import h2so4_cal_box_model
import seaborn as sns


# measurement
def ams_data():
    time_AMS = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                          '/FRAPPE-AMS_C130_20140731_R0.ict',
                          delimiter=',', skiprows=39, usecols=0)
    data_SO4 = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                          '/FRAPPE-AMS_C130_20140731_R0.ict',
                          delimiter=',', skiprows=39, usecols=1)  # ugpsm3
    print('time_AMS.shape', time_AMS.shape)  # ('time_AMS.shape', (22859,))
    print('data_SO4.shape', data_SO4.shape)  # ('data_SO4.shape', (22859,))

    data_SO4 = ma.masked_where(data_SO4 < 0, data_SO4)
    time_AMS_SO4 = time_AMS[~data_SO4.mask]
    data_SO4_2 = data_SO4[~data_SO4.mask]
    print('data_SO4_2.shape', data_SO4_2.shape)  # (935,)
    print('time_AMS_SO4.shape', time_AMS_SO4.shape)  # ('time_AMS_SO4.shape', (935,))

    data_NO3 = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                          '/FRAPPE-AMS_C130_20140731_R0.ict',
                          delimiter=',', skiprows=39, usecols=2)
    print('data_NO3.shape', data_NO3.shape)  # ('data_NO3.shape', (22859,))
    data_NO3 = ma.masked_where(data_NO3 < 0, data_NO3)
    time_AMS_NO3 = time_AMS[~data_NO3.mask]
    data_NO3_2 = data_NO3[~data_NO3.mask]
    print('data_NO3_2.shape', data_NO3_2.shape)  # ('data_NO3_2.shape', (837,))
    print('time_AMS_NO3.shape', time_AMS_NO3.shape)  # ('time_AMS_NO3.shape', (837,))

    data_NH4 = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                          '/FRAPPE-AMS_C130_20140731_R0.ict',
                          delimiter=',', skiprows=39, usecols=4)
    print('data_NH4.shape', data_NH4.shape)  # ('data_NH4.shape', (22859,))
    data_NH4 = ma.masked_where(data_NH4 < 0, data_NH4)
    time_AMS_NH4 = time_AMS[~data_NH4.mask]
    data_NH4_2 = data_NH4[~data_NH4.mask]
    print('data_NH4_2.shape', data_NH4_2.shape)  # ('data_NH4_2.shape', (628,))
    print('time_AMS_NH4.shape', time_AMS_NH4.shape)  # ('time_AMS_NH4.shape', (628,))
    return data_NH4_2, data_NO3_2, data_SO4_2, time_AMS_NH4, time_AMS_NO3, time_AMS_SO4


def interp_ams():
    ## since 3 dataset has different length, interp?
    # to interpolate the data_SO4_2#(935,) to the data_NH4_2(628,)
    data_NH4_2, data_NO3_2, data_SO4_2, time_AMS_NH4, time_AMS_NO3, time_AMS_SO4 = ams_data()

    interp_SO4_2 = interp1d(np.asarray(time_AMS_SO4), np.asarray(data_SO4_2), kind='nearest', bounds_error=False,
                            fill_value=-9999)
    SO4_2_track = interp_SO4_2(np.asarray(time_AMS_NH4))
    print('SO4_2_track.shape', SO4_2_track.shape)  # =

    # to interpolate the data_NO3_2(837,) to the data_NH4_2(628,)
    interp_NO3 = interp1d(np.asarray(time_AMS_NO3), np.asarray(data_NO3_2), kind='nearest', bounds_error=False,
                          fill_value=-9999)
    NO3_track = interp_NO3(np.asarray(time_AMS_NH4))
    print('NO3_track.shape', NO3_track.shape)  # ('NO3_track.shape', (628,))


def plot():
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.hist(np.divide(np.asarray(time_AMS_NH4), 3600), np.asarray(AMS_neut_coef), label='neut_coef_obs')
    plt.ylabel('neut_coef_obs', fontsize=16)
    plt.title('AMS obser')

    plt.subplot(212)
    plt.scatter(np.divide(np.asarray(time_AMS_NH4), 3600), np.asarray(AMS_neut_coef), label='neut_coef_obs')
    plt.ylabel('neut_coef_obs_zoom_in', fontsize=16)
    plt.ylim(0, 3)
    plt.xlabel('731 UTC', fontsize=16)

    # unit conversion
    NO3_track = NO3_track * 9.7e9  # from ug m-3 to molecule cm-3
    SO4_2_track = SO4_2_track * 6.27e9  # from ug m-3 to molecule cm-3
    data_NH4_2 = data_NH4_2 * 3.34e10  # from ug m-3 to molecule cm-3

    # Now, 3 column has dimension of (935,), we calculate [NO3]+2[SO4]/[NH4]
    AMS_neut_coef = np.divide(NO3_track + np.multiply(SO4_2_track, 2), data_NH4_2)

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.scatter(np.divide(np.asarray(time_AMS_NH4), 3600), np.asarray(AMS_neut_coef), label='neut_coef_obs')
    plt.ylabel('neut_coef_obs', fontsize=16)
    plt.title('AMS obser')

    plt.subplot(212)
    plt.scatter(np.divide(np.asarray(time_AMS_NH4), 3600), np.asarray(AMS_neut_coef), label='neut_coef_obs')
    plt.ylabel('neut_coef_obs_zoom_in', fontsize=16)
    plt.ylim(0, 3)
    plt.xlabel('0731 UTC', fontsize=16)

    plt.savefig(
        '/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs/neutr_coff_0731.png')
    plt.show()
