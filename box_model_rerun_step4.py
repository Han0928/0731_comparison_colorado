# box model for calculating [H2SO4]so2-oh
# [H2SO4]=(k1*[SO2]*[OH])/[CS]
import pandas as pd
import numpy as np
import math  # for e^x
from netCDF4 import Dataset
import numpy.ma as ma
from scipy.interpolate import interp1d, RegularGridInterpolator
from numpy.ma import exp
import datetime
import matplotlib.pyplot as plt
from cs_0731_rerun_step4 import merge_obs_method
import matplotlib.ticker as ticker


def h2so4_cal_box_model():
    temp_0 = 298  # in K
    press_0 = 1e5  # in pa
    R = 8.314  # Universal gas constant(J mol^-1 K^-1)
    temp = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict', delimiter=',',
                      skiprows=132, usecols=29)  # inC  75540
    temp = temp + 273  # in K
    pres = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict', delimiter=',',
                      skiprows=132, usecols=62)  # in hpa
    pres = pres * 100  # in pa
    time_NAV = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',
                          delimiter=',',
                          skiprows=132, usecols=0)  #
    height = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict', delimiter=',',
                        skiprows=132, usecols=4)  # in m
    # the same thing but read in model data
    r_p = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["model_P"])  # in Pa
    r_p = np.asarray(r_p).reshape(318)  # from291(0802) to 318(0729)
    r_T = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["model_T"])  # in K
    r_T = np.asarray(r_T).reshape(318)  # from291(0802) to 318(0729)

    n_0 = 2.55e19  # from P8 in (molecules cm-3)
    ##step1:calculate K1 from S+P p1101
    kl0 = 3.3e-31  # constant #constant
    kl = kl0 * (temp / 300) ** -4.3  # low-pressure limit,in(cm6 molecule-2 s-1)
    kl_r = kl0 * (r_T / 300) ** -4.3  # low-pressure limit,in(cm6 molecule-2 s-1)
    # print('kl', kl)

    ku0 = 1.6e-12  # constant
    ku = ku0 * (temp / 300)  # upper-pressure limit, in(cm3 molecule-1 s-1)
    ku_r = ku0 * (r_T / 300)  # upper-pressure limit, in(cm3 molecule-1 s-1)
    # print('ku', ku)

    # keep on observation part
    M = pres / (R * temp) * 6.022 * 1e17  # in molecule cm-3
    M_r = r_p / (R * r_T) * 6.022 * 1e17  # in molecule cm-3
    A = kl * M / ku
    A_r = kl_r * M_r / ku_r
    B = np.log10(A)
    B_r = np.log10(A_r)
    C = np.power((1 + np.power(B, 2)), -1)
    C_r = np.power((1 + np.power(B_r, 2)), -1)
    d = np.power(0.6, C)
    d_r = np.power(0.6, C_r)
    k_termole = (kl * M / (1 + A)) * d
    k_termole_r = (kl_r * M_r / (1 + A_r)) * d_r
    k1 = k_termole  # cm6 molecule-2 s-1
    k1_r = k_termole_r  # cm6 molecule-2 s-1
    # print('k1', k1)
    # plt.plot(time_NAV, k1)
    # plt.figure()

    # # read in so2 part and mask<0, 0731 does not have SO2
    # so2_ppt = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-GTCIMS_C130_20140802_R2.ict', delimiter=',',
    #                      skiprows=38, usecols=4)  # in ppt
    # print('so2_ppt.shape', so2_ppt.shape)  # ('so2_ppt.shape', (5003,))
    # time_so2 = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-GTCIMS_C130_20140802_R2.ict', delimiter=',',
    #                       skiprows=38, usecols=0)
    # print('time_so2.shape', time_so2.shape)  # ('time_so2.shape', (5003,))
    # # print('time_so2', time_so2)  # start from [75602.5,
    # so2_ppt = ma.masked_where(so2_ppt < 0, so2_ppt)  # 72-75 to eliminate invalid data(<0) in SO2 file
    # time_so2_2 = time_so2[~so2_ppt.mask]
    # so2_ppt_2 = so2_ppt[~so2_ppt.mask]
    # print('so2_ppt_2.shape', so2_ppt_2.shape)  # ('so2_ppt_2.shape', (4428,))
    # # print('so2_ppt_2', so2_ppt_2)  # ('so2_ppt_2.shape', (4428,))
    # print('time_so2_2.shape', time_so2_2.shape)  # ('time_so2_2.shape',  (4428,))
    # # plt.plot(time_so2_2, so2_ppt_2)  # up to 5500ppt??
    #
    # # to interpolate the SO2(3s) to the T/P(1s)
    # interp_so2 = interp1d(np.asarray(time_so2_2), np.asarray(so2_ppt_2), kind='nearest', bounds_error=False,
    #                       fill_value=-9999)
    # so2_track = interp_so2(np.asarray(time_NAV))
    # # print('so2_track', so2_track)
    # print('so2_track.shape', so2_track.shape)  # ('so2_track.shape', (17401,)
    # # plt.plot(time_NAV, so2_track, '-')
    #
    # # ppt to molecule cm-3; Here we need to resolve the different resolutionT/P/so2
    # so2 = (pres * 64) / (8.314 * temp) * so2_track * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    # so2 = so2 * 9.4e9  # from ug m-3 to molecule cm-3
    # # print('so2', so2)
    # print('so2_value', so2)
    # # plt.plot(time_NAV, so2, '-')

    # read in oh part and mask<0
    oh = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-OH-H2SO4_C130_20140731_R0.ict', delimiter=',',
                    skiprows=40, usecols=3)  # in molecule_cm-3 
    # print('oh.shape', oh.shape)  # ('oh.shape', (544,))
    time_oh = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-OH-H2SO4_C130_20140731_R0.ict', delimiter=',',
                         skiprows=40, usecols=0)
    # print('time_oh.shape', time_oh.shape)  # ('time_oh.shape', (544,))
    oh = ma.masked_where(oh < 0, oh)  # 72-75 to eliminate invalid data(<0) in SO2 file
    time_oh_2 = time_oh[~oh.mask]
    oh_2 = oh[~oh.mask]
    # print('oh_2.shape', oh_2.shape)  # ('oh_2.shape', (506,))
    # print('time_oh_2.shape', time_oh_2.shape)  # ('time_oh_2.shape', (506,))

    # to interpolate the oh(30s) to the T/P(1s)
    interp_oh = interp1d(np.asarray(time_oh_2), np.asarray(oh_2), kind='nearest', bounds_error=False,
                         fill_value=-9999)
    oh_track = interp_oh(np.asarray(time_NAV))
    # print('oh_track', oh_track)
    # print('oh_track.shape', oh_track.shape)  #
    # plt.figure()
    # plt.plot(time_NAV, oh_track, '-')

    # link the cs.py to here.
    obe_sumnc, time_cs = merge_obs_method()
    cs = np.asarray(obe_sumnc)
    interp_cs = interp1d(np.asarray(time_cs), np.asarray(cs), kind='nearest', bounds_error=False,
                         fill_value=-9999)
    cs_track = interp_cs(np.asarray(time_NAV))
    # print('cs_track.shape', cs_track.shape)

    # now calculate the box model H2SO4[so2+oh]
    # h2so4_calcu = k1 * np.asarray(so2) * np.asarray(oh_track) / cs_track

    # do the same thing for model
    model_time = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                             '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv',
                             parse_dates=True,
                             usecols=["time"])
    # print('model_time_value', model_time)  # 'model_time.shape', (291, 0))
    model_time = np.asarray(model_time).reshape(318)
    # print('model_time.shape', model_time.shape)
    # print('type(model_time[0])', type(model_time[0]))  # ('type(model_time[0])', <type 'str'>)
    # print('model_time[0]', model_time[0])

    r_cs = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["model_SUMNC2"])
    r_cs = np.asarray(r_cs).reshape(318)
    # read in ozone
    r_o3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["model_O3"])  # #ppbv
    r_o3 = np.asarray(r_o3).reshape(318)  # in ppt, but we need this in different units!!
    r_o3 = (r_p * 48) / (8.314 * r_T) * r_o3 * 1e-3  # ppt*1e-6=ppm
    r_o3 = r_o3 * 1.25e10  # from ug m-3 to molecule cm-3

    # read in c10h16
    r_c10h16 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                           '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                           usecols=["model_C10H16"])  # pptv
    r_c10h16 = np.asarray(r_c10h16).reshape(318)  # in ppt, but we need this in different units!!
    r_c10h16 = (r_p * 136) / (8.314 * r_T) * r_c10h16 * 1e-6  # ppt*1e-6=ppm
    r_c10h16 = r_c10h16 * 4.4e9  # from ug m-3 to molecule cm-3

    # calculate the creegie for regional model
    rate_o3_c10h16_r = 8.05e-16 * 0.6 * exp(np.divide(-640, r_T))  # temp now in K
    r_creegie = np.divide(np.multiply(np.multiply(r_o3, r_c10h16), rate_o3_c10h16_r),
                          r_cs)  # double check the correct unit!!!

    r_oh = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["model_OH"])  # in ppt
    r_oh = np.asarray(r_oh).reshape(318)  # in ppt, but we need this in different units!!
    r_oh = (r_p * 17) / (8.314 * r_T) * r_oh * 1e-6  # ppt*1e-6=ppm
    r_oh = r_oh * 3.5e10  # from ug m-3 to molecule cm-3

    r_so2 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["model_SO2"])  # pptv
    r_so2 = np.asarray(r_so2).reshape(318)
    r_so2 = (r_p * 64) / (8.314 * r_T) * r_so2 * 1e-6  # ppt*1e-6=ppm
    r_so2 = r_so2 * 9.4e9  # from ug m-3 to molecule cm-3

    r_h2so4 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_H2SO4"])  # pptv
    # r_h2so4 = (r_p * 98) / (8.314 * r_T) * r_h2so4 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    # r_h2so4 = r_h2so4 * 6.14e9  # from ug m-3 to molecule cm-3
    r_h2so4 = np.asarray(r_h2so4).reshape(318)
    r_h2so4 = np.multiply(np.multiply(np.divide(np.multiply(r_p, 98), np.multiply(8.314, r_T)), r_h2so4),
                          1e-6)  # ppt*1e-6=ppm
    r_h2so4 = np.multiply(r_h2so4, 6.14e9)  # from ug m-3 to molecule cm-3
    # print('r_h2so4.shape', r_h2so4.shape)  # ('r_h2so4.shape', (291, 0))
    # h2so4_calcu = k1 * np.asarray(so2) * np.asarray(oh_track) / cs_track
    h2so4_calcu_r = k1_r * np.asarray(r_so2) * np.asarray(r_oh) / r_cs
    # print('h2so4_calcu', h2so4_calcu)
    # print('h2so4_calcu_r_value', h2so4_calcu_r)
    # plt.plot((time_NAV / 3600) - 7, h2so4_calcu, '-')
    # plt.yscale('symlog', linthreshy=0.01)

    # plt.figure()
    # # # plt.plot(creegie, 1 - np.divide(h2so4_calcu, h2so4_track), '*')
    # plt.plot(r_creegie, 1 - np.divide(h2so4_calcu_r, r_h2so4), '*')
    # plt.xlabel('[O3]*[C10H16]*rate constant/CS')
    # plt.ylabel('Creegie(1-C/M) %')
    # plt.title('SCI calculation for regional model 731')
    # plt.grid(linestyle='-.')
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/SCI_criegee_0731_r.png')
    #
    # plt.figure()
    # plt.plot(r_creegie, 1 - np.divide(h2so4_calcu_r, r_h2so4), '*')
    # plt.xlabel('[O3]*[C10H16]*rate constant/CS', fontsize=16)
    # plt.ylabel('Creegie(1-C/M) %', fontsize=16)
    # plt.ylim(0, 0.5)
    # plt.xlim(0, 0.5e7)
    # plt.title('SCI calculation for regional model 0731')
    # plt.grid(linestyle='-.')
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/SCI_criegee_0731_zoomin_r.png')

    # read in h2so4 part and mask<0
    h2so4_measurement = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-OH-H2SO4_C130_20140802_R0.ict',
                                   delimiter=',',
                                   skiprows=40, usecols=4)  # in molecule_cm-3  (, )  # in molecule cm-3
    # print('h2so4_measurement.shape', h2so4_measurement.shape)  # ('oh.shape', (,))
    time_h2so4_measurement = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-OH-H2SO4_C130_20140802_R0.ict',
                                        delimiter=',',
                                        skiprows=40, usecols=0)
    # print('time_h2so4_measurement.shape', time_h2so4_measurement.shape)  # ('time_oh.shape', (,))

    h2so4_measurement = ma.masked_where(h2so4_measurement < 0,
                                        h2so4_measurement)  # 72-75 to eliminate invalid data(<0) in SO2 file
    time_h2so4_measurement_2 = time_h2so4_measurement[~h2so4_measurement.mask]
    h2so4_measurement_2 = h2so4_measurement[~h2so4_measurement.mask]
    # print('time_h2so4_measurement_2.shape', time_h2so4_measurement_2.shape)  # ('oh_2.shape', (,))
    # print('h2so4_measurement_2.shape', h2so4_measurement_2.shape)  # ('time_oh_2.shape', (,))

    # interp the 30s resolution into 1s to do the difference calculation
    interp_h2so4 = interp1d(np.asarray(time_h2so4_measurement_2), np.asarray(h2so4_measurement_2), kind='nearest',
                            bounds_error=False,
                            fill_value=-9999)
    h2so4_track = interp_h2so4(np.asarray(time_NAV))  # so final observ is named h2so4_track
    # print('h2so4_track', h2so4_track)
    # print('h2so4_track_value', h2so4_track)  # ('so2_track.shape', (17401,)
    # # plt.plot((time_NAV/3600), h2so4_track, '-')
    # # plt.plot(time_h2so4_measurement_2 / 3600 - 7, h2so4_measurement_2, '-')
    # # plt.plot((time_NAV / 3600), h2so4_calcu, '-')
    # plt.plot((time_NAV / 3600), h2so4_track, '-')
    # plt.yscale('symlog', linthreshy=0.01)

    # # to compare the h2so4_track vs h2so4_calcu difference;
    # # h2so4_diff = h2so4_track - h2so4_calcu
    h2so4_diff_r = r_h2so4 - h2so4_calcu_r
    # print('h2so4_diff', h2so4_diff)
    # plt.plot((time_NAV / 3600), h2so4_diff, '-')
    # # plt.yscale('symlog', linthreshy=0.01)
    # # plt.legend(["h2so4_calcu", "h2so4_track", "h2so4_diff"])
    # plt.ylim(-4e7, 8e7)
    # plt.xlabel('time')
    # plt.ylabel('Concentration (molecule cm-3) for Obs')
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/h2so4_box_0731.png')
    # plt.figure()

    # # twin-y with altitude fig
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # # ax1.plot((time_NAV / 3600) - 7, h2so4_calcu, '-', label='h2so4_calcu')
    # # ax1.plot((time_NAV / 3600) - 7, h2so4_diff, '-', label='h2so4_diff')
    # ax1.plot((time_NAV / 3600) - 7, h2so4_track, '-', label='h2so4_track')
    # ax1.set_ylabel('concentration (molecule cm-3')
    # ax1.set_title("H2SO4 comparison box model&measu 0731")
    # ax1.set_ylim([-4e7, 8e7])
    # ax1.legend(loc=4)
    # ax1.grid()
    #
    # ax2 = ax1.twinx()  # double-y
    # ax2.plot((time_NAV / 3600) - 7, height, '-', label='height')
    # ax2.set_ylabel('height (m)')
    # ax2.legend(loc=0)
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/double_y.png')
    # plt.figure()

    # # plot for regional model
    # fig = plt.figure(figsize=(8, 5))
    # ax = fig.add_subplot(111)
    # ax = plt.gca()
    # ax.plot(np.asarray(model_time), np.asarray(h2so4_calcu_r), '-', label='H$_{2}$SO$_{4}$_calcu_r')
    # ax.plot(np.asarray(model_time), np.asarray(r_h2so4), '-', label='H$_{2}$SO$_{4}$_r')
    # ax.plot(np.asarray(model_time), np.asarray(h2so4_diff_r), '-', label='H$_{2}$SO$_{4}$_diff_r')
    # tick_spacing = 100
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    # plt.title('H$_{2}$SO$_{4}$ comparison in regional model 0731')
    # # plt.xticks(rotation=15)
    # # plt.grid()
    # ax = plt.gca()
    # plt.ylabel('Concentration (molecule cm$-3}$)', fontsize=12)
    # # plt.xlabel('time in UTC',fontsize=14)
    # plt.setp(ax.get_yticklabels(), fontsize=12)
    # plt.legend(["H$_{2}$SO$_{4}$_calcu_r", "H2SO4_r", "H$_{2}$SO$_{4}$_diff_r"], fontsize=14)
    # # plt.xlabel('time', fontsize=16)
    # # plt.yscale('log')
    # # plt.locator_params(nbins=10, axis='x')
    # plt.ylim(-1e7, 3e7)
    # plt.legend(loc=4)
    # plt.grid(linestyle='-.')
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/0112_regional.png')

    # fig = plt.figure()
    # plt.plot(np.asarray(model_time), np.asarray(h2so4_calcu_r), '-', label='h2so4_calcu_r')
    # plt.plot(np.asarray(model_time), np.asarray(r_h2so4), '-', label='r_h2so4')
    # plt.plot(np.asarray(model_time), np.asarray(h2so4_diff_r), '-', label='h2so4_diff_r')
    # plt.ylabel('concentration (molecule cm-3)')
    # plt.title("H2SO4 comparison Regional Model 0731")
    # plt.ylim([-1e7, 3e7])
    # plt.legend(loc=4)
    # plt.grid()
    # # plt.show()

    # # plot fig: fraction vs OH, we expect negative correlation
    # plt.figure()
    # plt.plot(oh_track, 1 - np.divide(h2so4_calcu, h2so4_track), '.')
    # plt.xlabel('OH molecule cm-3')
    # plt.ylabel('Creegie(1-C/M) %')
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/oh_criegee_0731.png')

    # now we confirm the criegee is from VOC=isoprene+monoterpene+benzene+tolune
    # first read in Isoprene
    isop_measurement = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-TOGA_C130_20140802_R2.ict',
                                  delimiter='\t',
                                  skiprows=90, usecols=16)
    # print('isoprene_measurement.shape', isop_measurement.shape)  #
    time_isoprene = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-TOGA_C130_20140802_R2.ict',
                               delimiter='\t', skiprows=90, usecols=0)
    # print('time_isoprene.shape', time_isoprene.shape)

    # mask <0 for isoprene
    isop_measurement = ma.masked_where(isop_measurement < 0, isop_measurement)  #
    time_isoprene_2 = time_isoprene[~isop_measurement.mask]
    isop_measurement_2 = isop_measurement[~isop_measurement.mask]
    # print('time_isoprene_2.shape', time_isoprene_2.shape)  #
    # print('isoprene_measurement_2.shape', isop_measurement_2.shape)  #

    # interp 120s resolution into 1s to do the difference calculation
    interp_isoprene = interp1d(np.asarray(time_isoprene_2), np.asarray(isop_measurement_2), kind='nearest',
                               bounds_error=False,
                               fill_value=-9999)
    isoprene_track = interp_isoprene(np.asarray(time_NAV))
    # print('isoprene_track', isoprene_track)
    # print('isoprene_track.shape', isoprene_track.shape)  #
    # isoprene from ppt to molecule cm-3(step3 in printing paper,isoprene=C5H8=68 )
    isoprene = (pres * 68) / (8.314 * temp) * isoprene_track * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    isoprene = isoprene * 8.9e9  # from ug m-3 to molecule cm-3
    # print('isoprene', isoprene)
    # print('isoprene_shape', isoprene.shape)

    ## read in Alpha_Pinene_measurement
    alpha_pinene_measurement = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-TOGA_C130_20140802_R2.ict',
                                          delimiter='\t',
                                          skiprows=90, usecols=51)  # in ppt, resolution:every 120s
    time_al_pine = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-TOGA_C130_20140802_R2.ict',
                              delimiter='\t', skiprows=90, usecols=0)
    # print('alpha_pinene_measurement.shape', alpha_pinene_measurement.shape)  #

    # mask <0 for alpha_pinene
    alpha_pinene_measurement = ma.masked_where(alpha_pinene_measurement < 0,
                                               alpha_pinene_measurement)  #
    time_al_pine_2 = time_al_pine[~alpha_pinene_measurement.mask]  # same time dimention with isoprene_time
    alpha_pinene_measurement_2 = alpha_pinene_measurement[~alpha_pinene_measurement.mask]
    # print('alpha_pinene_measurement_2.shape', alpha_pinene_measurement_2.shape)  # ('time_oh_2.shape', (,))

    # interp 120s resolution into 1s to do the difference calculation
    interp_alpha_pinene = interp1d(np.asarray(time_al_pine_2), np.asarray(alpha_pinene_measurement_2), kind='nearest',
                                   bounds_error=False,
                                   fill_value=-9999)
    alpha_pinene_track = interp_alpha_pinene(np.asarray(time_NAV))
    # print('alpha_pinene_track', alpha_pinene_track)
    # print('alpha_pinene_track.shape', alpha_pinene_track.shape)  #
    # benzene from ppt to molecule cm-3
    alpha_pinene = (pres * 136) / (
            8.314 * temp) * alpha_pinene_track * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    alpha_pinene = alpha_pinene * 4.4e9  # from ug m-3 to molecule cm-3
    # print('alpha_pinene', alpha_pinene)
    # print('alpha_pinene_shape', alpha_pinene.shape)

    ## read in O3 measurement
    ozone_measurement = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NONO2O3_C130_20140731_R0.ict',
                                   delimiter=',',
                                   skiprows=38, usecols=5)  # in  ppbv, resolution:every 1s
    time_ozone = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NONO2O3_C130_20140731_R0.ict',
                            delimiter=',', skiprows=38, usecols=0)  # in , instead of\t
    # print('ozone_measurement.value', ozone_measurement)  #

    # mask <0 for ozone
    ozone_measurement = ma.masked_where(ozone_measurement < 0,
                                        ozone_measurement)  #
    time_ozone_2 = time_ozone[~ozone_measurement.mask]  # same time dimention with isoprene_time
    ozone_measurement_2 = ozone_measurement[~ozone_measurement.mask]
    # print('alpha_pinene_measurement_2.shape', alpha_pinene_measurement_2.shape)  # ('time_oh_2.shape', (,))

    # interp 1s resolution into 1s to do the difference calculation
    interp_ozone = interp1d(np.asarray(time_ozone_2), np.asarray(ozone_measurement_2), kind='nearest',
                            bounds_error=False,
                            fill_value=-9999)
    ozone_track = interp_ozone(np.asarray(time_NAV))

    # O3 from ppb to molecule cm-3
    ozone = (pres * 48) / (
            8.314 * temp) * ozone_track * 1e-3  # transfer ppb into ug m-3 given T and P, (S+P)P14
    ozone = ozone * 1.25e10  # from ug m-3 to molecule cm-3

    # calculate creegie reaction rate http://mcm.york.ac.uk/browse.htt?species=APINENE
    rate_o3_c10h16 = 8.05e-16 * 0.6 * exp(np.divide(-640, temp))  # temp now in K
    creegie = np.divide(np.multiply(np.multiply(ozone, alpha_pinene), rate_o3_c10h16),
                        cs_track)  # double check the correct unit!!!

    # plt.figure()
    # plt.plot(creegie, 1 - np.divide(h2so4_calcu, h2so4_track), '*')
    # plt.xlabel('[O3]*[C10H16]*rate constant/CS')
    # plt.ylabel('Creegie(1-C/M) %')
    # plt.title('SCI calculation 0731')
    # plt.grid()
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/SCI_criegee_0731.png')
    #
    # plt.figure()
    # plt.plot(creegie, 1 - np.divide(h2so4_calcu, h2so4_track), '*')
    # plt.xlabel('[O3]*[C10H16]*rate constant/CS')
    # plt.ylabel('Creegie(1-C/M) %')
    # plt.ylim(0, 50)
    # plt.xlim(0, 4e11)
    # plt.title('SCI calculation 0731')
    # plt.grid()
    # plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
    #             '/0731_comparison/SCI_criegee_0731_zoomin.png')

    # Now I want to df.csv to draw the colormap in Orgin.
    # d_creegie = {'time_NAV': pd.Series(time_NAV), 'height': pd.Series(height),
    #              'creegie': pd.Series(creegie),
    #              '1-c/m': pd.Series(np.asarray(1 - np.divide(h2so4_calcu, h2so4_track)))}
    # df_cr2 = pd.DataFrame(d_creegie)
    # df_cr2.to_csv('CREEGIE.csv')
    # # df2 = df_cr2.resample('1T', on='time').mean()

    # # read in benzene
    # benz_m = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-TOGA_C130_20140731_R2.ict',
    #                     delimiter='\t',
    #                     skiprows=90, usecols=37)  # in ppt, resolution:every 120s
    # time_benzene = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-TOGA_C130_20140731_R2.ict',
    #                           delimiter='\t', skiprows=90, usecols=0)
    # # print('benz_m.shape', benz_m.shape)  #
    #
    # # mask <0 for benzene
    # benz_m = ma.masked_where(benz_m < 0,
    #                          benz_m)  # 72-75 to eliminate invalid data(<0) in SO2 file
    # time_benzene_2 = time_benzene[~benz_m.mask]  # same time dimention with isoprene_time
    # benz_m_2 = benz_m[~benz_m.mask]
    # # print('benz_m_2.shape', benz_m_2.shape)  # ('time_oh_2.shape', (,))
    #
    # # interp 120s resolution into 1s to do the difference calculation
    # interp_benzene = interp1d(np.asarray(time_benzene_2), np.asarray(benz_m_2), kind='nearest',
    #                           bounds_error=False,
    #                           fill_value=-9999)
    #
    # # print('time_benzene_2.shape', time_benzene_2.shape)
    # benzene_track = interp_benzene(np.asarray(time_NAV))
    # # print('benzene_track', benzene_track)
    # # print('benzene_track.shape', benzene_track.shape)  #
    # # benzene from ppt to molecule cm-3
    # benzene = (pres * 78) / (8.314 * temp) * benzene_track * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    # benzene = benzene * 7.7e9  # from ug m-3 to molecule cm-3
    # # print('benzene', benzene)
    # # print('benzene_shape', benzene.shape)
    #
    # # read in toluene_measurement
    # toluene_measurement = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-TOGA_C130_20140731_R2.ict',
    #                                  delimiter='\t', skiprows=90, usecols=44)  # in ppt, resolution:every 120s
    # time_toluene = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-TOGA_C130_20140731_R2.ict',
    #                           delimiter='\t', skiprows=90, usecols=0)
    # # print('toluene_measurement.shape', toluene_measurement.shape)  #
    #
    # # mask <0 for isoprene
    # toluene_measurement = ma.masked_where(toluene_measurement < 0,
    #                                       toluene_measurement)  # 72-75 to eliminate invalid data(<0) in SO2 file
    # time_toluene_2 = time_toluene[~toluene_measurement.mask]  # same time dimention with isoprene_time
    # toluene_measurement_2 = toluene_measurement[~toluene_measurement.mask]
    # # print('toluene_measurement_2.shape', toluene_measurement_2.shape)  # ('time_oh_2.shape', (,))
    #
    # # interp 120s resolution into 1s to do the difference calculation
    # interp_toluene = interp1d(np.asarray(time_toluene_2), np.asarray(toluene_measurement_2), kind='nearest',
    #                           bounds_error=False,
    #                           fill_value=-9999)
    # toluene_track = interp_toluene(np.asarray(time_NAV))
    # # print('toluene_track', toluene_track)
    # # print('toluene_track.shape', toluene_track.shape)  #
    # # benzene from ppt to molecule cm-3
    # toluene = (pres * 92) / (8.314 * temp) * toluene_track * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    # toluene = toluene * 6.5e9  # from ug m-3 to molecule cm-3
    # print('toluene', toluene)
    # print('toluene_shape', toluene.shape)

    # total VOC now
    # voc = isoprene + alpha_pinene + benzene + toluene
    # plot the twin-y
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(np.asarray(voc), 1 - np.divide(h2so4_calcu, h2so4_track), '.', label='VOC=IP+AP+BENZ+TOL')
    # ax1.set_xlim(0e10, 2e10)
    # ax1.set_ylabel('Criegee(1-C/M) %')
    # ax1.set_title("negative correlation between Criegee and H2SO4")
    # ax1.legend(loc=3)
    # ax1.grid()
    #
    # ax2 = ax1.twinx()  # double-y
    # ax2.plot(np.asarray(voc), height, '-', label='height')
    # ax2.set_ylabel('height (m)')
    # ax2.legend(loc=0)
    # plt.savefig('/jet/home/ding0928/box_model/data/fig_out_boxmodel/double_y_voc.png')
    # plt.figure()

    # plot fig: fraction vs OH, we expect negative correlation
    # plt.figure()
    # plt.plot(oh_track, 1 - np.divide(h2so4_calcu, h2so4_track), '.')
    # plt.xlabel('OH molecule cm-3')
    # plt.ylabel('Creegie(1-C/M) %')
    # plt.savefig('/jet/home/ding0928/box_model/data/fig_out_boxmodel/oh_criegee_0802.png')
    # plt.figure()
    # plt.plot(np.asarray(voc), 1 - np.divide(h2so4_calcu, h2so4_track), '.')
    # plt.xlabel('voc=ip+ap+ben+tol in (molecule cm-3)')
    # plt.ylabel('Creegie(1-C/M) %')
    # plt.savefig('/jet/home/ding0928/box_model/data/fig_out_boxmodel/voc_criegee_0802.png')
    # plt.figure()

    # VOC>0
    # voc = isoprene + alpha_pinene + benzene + toluene
    # # voc = isoprene + alpha_pinene + benzene + toluene
    # plt.plot(np.asarray(voc), 1 - np.divide(h2so4_calcu, h2so4_track), '.')
    # plt.xlim(0e10, 2e10)
    # plt.xlabel('VOC=IP+AP+BENZ+TOL in (molecule cm-3)')
    # plt.ylabel('Creegie(1-C/M) %')
    # plt.savefig('/jet/home/ding0928/box_model/data/fig_out_boxmodel/voc_criegee_0802_zoomin.png')
    # plt.figure()
    # plt.show()

    # For J;
    # since we will link box model with J_nuc.py, so we need to read in NH3+RH data
    # first, read in NH3 data
    nh3_measurement = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NH3_C130_20140731_R0.ict',
                                 delimiter=',',
                                 skiprows=36, usecols=3)  # in ppb
    # print('nh3_measurement.shape', nh3_measurement.shape)  # resolution:~1s,(17350,))
    time_nh3 = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NH3_C130_20140731_R0.ict',
                          delimiter=',',
                          skiprows=36, usecols=0)

    nh3_measurement = ma.masked_where(nh3_measurement < 0,
                                      nh3_measurement)
    time_nh3_2 = time_nh3[~nh3_measurement.mask]
    nh3_measurement_2 = nh3_measurement[~nh3_measurement.mask]

    # after mask it goes to 7170, interpolate back into 17410
    interp_nh3 = interp1d(np.asarray(time_nh3_2), np.asarray(nh3_measurement_2), kind='nearest',
                          bounds_error=False,
                          fill_value=-9999)
    nh3_track = interp_nh3(np.asarray(time_NAV))

    # ppb to molecule cm-3
    nh3 = (pres * 17) / (8.314 * temp) * nh3_track * 1e-3  # ppb*1e-3=ppm
    nh3 = nh3 * 3.5e10  # from ug m-3 to molecule cm-3

    # Now read in RH from NAV file
    rh = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict', delimiter=',',
                    skiprows=132, usecols=66)  # %
    # print('rh_value', rh)  # ('rh', (17401,))

    return temp, h2so4_track, nh3, rh, M, obe_sumnc, time_cs, height


h2so4_cal_box_model()

# make the (75630)time into readable format, but if interp, this part is no need.
# date = '20140802'
# epoch = datetime.datetime(2014, 7, 31)
# unix_epoch_of_31July2014_0001 = 1406764800
# if date == '20140731':
#     trackpdtime = pd.to_datetime(np.asarray(tracktime) + unix_epoch_of_31July2014_0001, unit='s')
# elif date == '20140801':
#     unix_epoch_of_01August2014_0001 = unix_epoch_of_31July2014_0001 + 86400
#     trackpdtime = pd.to_datetime(np.asarray(tracktime) + unix_epoch_of_01August2014_0001, unit='s')
# elif date == '20140802':
#     unix_epoch_of_02August2014_0001 = unix_epoch_of_31July2014_0001 + 86400 * 2
#     trackpdtime_NAV = pd.to_datetime(np.asarray(time_NAV) + unix_epoch_of_02August2014_0001, unit='s')
#     trackpdtime_so2 = pd.to_datetime(np.asarray(time_so2_2) + unix_epoch_of_02August2014_0001, unit='s')
# print(trackpdtime_NAV)  # 2014-08-02 21:00:30
# print(trackpdtime_so2)  # 2014-08-02 19:54:01


# for temp/pres from 1s resolution to 30s resolution
# d_tp = {'temp': pd.Series(temp), 'pres': pd.Series(pres), 'time': pd.Series(trackpdtime_NAV)
#         }  # could everything put into one dic?
# df_tp = pd.DataFrame(d_tp)
# print('df_tp', df_tp)
# df2_tp = df_tp.resample('30S', on='time').mean()
# print('df2_tp', df2_tp)  # [578 rows x 2 columns]  why it's different? how to quote the separate T P
#
# # for temp/pres from 1s resolution to 30s resolution
# d_so2 = {'so2_ppt_2': pd.Series(so2_ppt_2), 'time': pd.Series(trackpdtime_so2)}  # could everything put into one dic?
# df_so2 = pd.DataFrame(d_so2)
# print('df_so2', df_so2)  # [567 rows x 1 columns])
# df2_so2 = df_so2.resample('30S', on='time').mean()
# print('df2_so2', df2_so2)  # [578 rows x 2 columns]


# before we plug in the P1101 k(T,Z) equation, we need to have [M]
# first we calculate the z->p; or just read in height(m)
# def p_z(t_0, p_0, t_x,
#         p_x):  # p->z, from https://baike.baidu.com/item/%E5%8E%8B%E9%AB%98%E5%85%AC%E5%BC%8F/2120153?fr=aladdin
#     z0 = 0
#     t = 1 / 2 * (t_0 + t_x)  # get the mean t as an approximation
#     delta_z = 8000 * (1 + 1 / 273 * t) * np.log(p_0 / p_x)  # in m from the above link solution2 in ln instead of log
#     delta_z = delta_z / 1000  # in km
#     z = delta_z + z0  # pretend surface level is 0
#     scale_height = 7.4
#     h_expo = -z / scale_height
#     n0 = 2.55e19
#     m = n0 * exp(h_expo)
#     return m
# M = p_z(temp_0, press_0, temp, pres)  # under surface=2.46*10^19 from P122 S+P
