# calculating the nucleation rate J based on Gordon 2017 supl
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
import matplotlib.colors as colors
from box_model_rerun_step4 import h2so4_cal_box_model
import seaborn as sns
# import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
# from box_model_rerun_step4 import h2so4_cal_box_model
import seaborn as sns
import os
from windrose import WindroseAxes
import matplotlib.cm as cm

# from colorado_all import colorado_mod_J
fig = plt.figure(constrained_layout=True)
# global fig
gs = GridSpec(2, 20, figure=fig)
plt.rcParams['font.size'] = 12


def read_c130_smps():
    c130_smps_time = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/frappe'
                                '-SMPS_C130_20140731_R0.ict', delimiter=',', skiprows=66, usecols=0)
    c130_smps_time = c130_smps_time - 21600

    c = range(19, 33)
    c130_data_smps = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/frappe'
                                '-SMPS_C130_20140731_R0.ict', delimiter=',', skiprows=66, usecols=c)

    c130_data_smps = ma.masked_where(c130_data_smps < 0, c130_data_smps)

    c130_diam_list_smps = np.array([6, 8.4, 10, 12, 14.5, 17.4, 20, 25, 30, 38, 45, 55, 66, 80, 97])

    date = '20140731'
    unix_epoch_of_31July2014_0001 = 1406764800
    if date == '20140731':
        trackpdtime = pd.to_datetime(np.asarray(c130_smps_time) + unix_epoch_of_31July2014_0001, unit='s')

    ax1 = fig.add_subplot(gs[0, :])
    plt.pcolormesh(trackpdtime, c130_diam_list_smps, c130_data_smps.T, norm=colors.LogNorm(vmin=10, vmax=110000),
                   cmap='RdBu_r')
    plt.colorbar(label='dN/dlog$_{Dp}$\n(# cm$^{-3}$)', extend='max')
    plt.ylabel('Diameter\n (nm)')
    plt.ylim(5, 100)
    plt.yscale('log')
    xmin = '2014-07-31 12:00:00'
    xmax = '2014-07-31 17:30:00'
    plt.xlim(xmin, xmax)
    # plt.show()


def J_rate():
    # Model input: read in the long python script(model 4d interpolation->1d) from csv
    df = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
                     '/0731_comparison/interpolated_korus_dataset_20140731.csv', index_col=0, parse_dates=True)
    import pytz
    mountain = pytz.timezone('US/Mountain')  # define UTC
    df.index = df.index.tz_localize(pytz.utc)
    df.index = df.index.tz_convert(mountain)  # from UTC to mountain time
    # indexed = df.set_index(df['time'], append=True)
    # df = indexed.resample('1T', on='time').mean()
    print('df', df)
    # print(df['OH'])

    # first for reginal model
    model_time = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                             '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv',
                             parse_dates=True,
                             usecols=["time"])
    model_time = np.asarray(model_time).reshape(318)

    r_p = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["model_P"])  # in Pa
    r_p = np.asarray(r_p).reshape(318)
    r_T = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["model_T"])  # in K
    r_T = np.asarray(r_T).reshape(318)

    r_h2so4 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_H2SO4"])  # pptv
    # r_h2so4 = (r_p * 98) / (8.314 * r_T) * r_h2so4 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    # r_h2so4 = r_h2so4 * 6.14e9  # from ug m-3 to molecule cm-3
    r_h2so4 = np.asarray(r_h2so4).reshape(318)
    r_h2so4 = np.multiply(np.multiply(np.divide(np.multiply(r_p, 98), np.multiply(8.314, r_T)), r_h2so4),
                          1e-6)  # ppt*1e-6=ppm
    r_h2so4 = np.multiply(r_h2so4, 6.14e9)

    r_nh3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["model_NH3"])  # pptv
    r_nh3 = np.asarray(r_nh3).reshape(318)
    r_nh3 = np.multiply(np.multiply(np.divide(np.multiply(r_p, 17), np.multiply(8.314, r_T)), r_nh3),
                        1e-6)  # ppt*1e-6=ppm
    r_nh3 = np.multiply(r_nh3, 3.5e10)  # from ug m-3 to molecule cm-3

    r_cs = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["model_SUMNC2"])
    r_cs = np.asarray(r_cs).reshape(318)

    r_oh = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["model_OH"])  # in ppt
    r_oh = np.asarray(r_oh).reshape(318)
    r_oh = (r_p * 17) / (8.314 * r_T) * r_oh * 1e-6  # ppt*1e-6=ppm
    r_oh = r_oh * 3.5e10  # from ug m-3 to molecule cm-3

    r_so2 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["model_SO2"])  # pptv
    r_so2 = np.asarray(r_so2).reshape(318)
    r_so2 = (r_p * 64) / (8.314 * r_T) * r_so2 * 1e-6  # ppt*1e-6=ppm
    r_so2 = r_so2 * 9.4e9  # from ug m-3 to molecule cm-3

    r_model = 80  # %
    R = 8.314
    r_M = r_p / (R * r_T) * 6.022 * 1e17  # box line63

    # then for global model
    g_p = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["glm_P"])  # in Pa
    g_p = np.asarray(g_p).reshape(318)
    g_T = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["glm_T"])  # in K
    g_T = np.asarray(g_T).reshape(318)
    g_h2so4 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_H2SO4"])  # pptv
    g_h2so4 = np.asarray(g_h2so4).reshape(318)
    # g_h2so4 = (g_p * 98) / (8.314 * g_T) * g_h2so4 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
    # g_h2so4 = g_h2so4 * 6.14e9  # from ug m-3 to molecule cm-3
    g_h2so4 = np.multiply(np.multiply(np.divide(np.multiply(g_p, 98), np.multiply(8.314, g_T)), g_h2so4),
                          1e-6)  # ppt*1e-6=ppm
    g_h2so4 = np.multiply(g_h2so4, 6.14e9)

    g_nh3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["glm_NH3"])  # pptv
    g_nh3 = np.asarray(g_nh3).reshape(318)
    # g_nh3 = (g_p * 17) / (8.314 * g_T) * g_nh3 * 1e-6  # ppt*1e-6=ppm
    # g_nh3 = g_nh3 * 3.5e10  # from ug m-3 to molecule cm-3
    g_nh3 = np.multiply(np.multiply(np.divide(np.multiply(g_p, 17), np.multiply(8.314, g_T)), g_nh3),
                        1e-6)  # ppt*1e-6=ppm
    g_nh3 = np.multiply(g_nh3, 3.5e10)  # from ug m-3 to molecule cm-3

    g_oh = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["glm_OH"])  # in ppt
    g_oh = np.asarray(g_oh).reshape(318)
    g_oh = (g_p * 17) / (8.314 * g_T) * g_oh * 1e-6  # ppt*1e-6=ppm
    g_oh = g_oh * 3.5e10  # from ug m-3 to molecule cm-3

    g_so2 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["glm_SO2"])  # pptv
    g_so2 = np.asarray(g_so2).reshape(318)
    g_so2 = (g_p * 64) / (8.314 * g_T) * g_so2 * 1e-6  # ppt*1e-6=ppm
    g_so2 = g_so2 * 9.4e9  # from ug m-3 to molecule cm-3

    g_cs = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["glm_SUMNC2"])
    g_cs = np.asarray(g_cs).reshape(318)
    g_M = g_p / (R * g_T) * 6.022 * 1e17

    # finally we need the altitude
    altitude_obs = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                               '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv',
                               parse_dates=True,
                               usecols=["altitude"])  # m
    altitude_obs = np.asarray(altitude_obs).reshape(318)

    ## Now read in the 8=4 conc+4 dia first for regional
    r_N_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_N_Ait"])  # not sure the unit??
    r_N_Ait = np.asarray(r_N_Ait).reshape(318)
    r_N_Ait_In = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                             '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv',
                             parse_dates=True,
                             usecols=["model_N_Ait_In"])  # not sure the unit??
    r_N_Ait_In = np.asarray(r_N_Ait_In).reshape(318)
    r_N_Acu = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_N_Acu"])  # not sure the unit??
    r_N_Acu = np.asarray(r_N_Acu).reshape(318)
    r_N_Cor = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_N_Cor"])  # not sure the unit??
    r_N_Cor = np.asarray(r_N_Cor).reshape(318)
    r_D_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_D_Ait"])  # in m
    r_D_Ait = np.asarray(r_D_Ait).reshape(318)
    r_D_Ait_In = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                             '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv',
                             parse_dates=True,
                             usecols=["model_D_Ait_In"])  # in m
    r_D_Ait_In = np.asarray(r_D_Ait_In).reshape(318)
    r_D_Cor = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_D_Cor"])  # in m
    r_D_Cor = np.asarray(r_D_Cor).reshape(318)
    r_D_Acu = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_D_Acu"])  # in m
    r_D_Acu = np.asarray(r_D_Acu).reshape(318)
    # Now read in the 8=4 conc+4 dia then for global
    g_N_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_N_Ait"])  # not sure the unit??
    g_N_Ait = np.asarray(g_N_Ait).reshape(318)
    g_N_Ait_In = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                             '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv',
                             parse_dates=True,
                             usecols=["glm_N_Ait_In"])  # not sure the unit??
    g_N_Ait_In = np.asarray(g_N_Ait_In).reshape(318)
    g_N_Acu = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_N_Acu"])  # not sure the unit??
    g_N_Acu = np.asarray(g_N_Acu).reshape(318)
    g_N_Cor = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_N_Cor"])  # not sure the unit??
    g_N_Cor = np.asarray(g_N_Cor).reshape(318)
    g_D_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_D_Ait"])  # in m
    g_D_Ait = np.asarray(g_D_Ait).reshape(318)
    g_D_Ait_In = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                             '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv',
                             parse_dates=True,
                             usecols=["glm_D_Ait_In"])  # in m
    g_D_Ait_In = np.asarray(g_D_Ait_In).reshape(318)
    g_D_Cor = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_D_Cor"])  # in m
    g_D_Cor = np.asarray(g_D_Cor).reshape(318)
    g_D_Acu = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_D_Acu"])  # in m
    g_D_Acu = np.asarray(g_D_Acu).reshape(318)

    ## Observation input:
    ## Link the box_model here
    temp, h2so4_track, nh3, rh, M, cs_obse, time_cs, height = h2so4_cal_box_model()
    nh3 = np.where(np.asarray(nh3) < 0, 1e-1,
                   np.asarray(nh3))  # to make the negative velue to 0, but does not change the shape.
    h2so4_track = np.where(np.asarray(h2so4_track) < 0, 1e-1, np.asarray(h2so4_track))
    # h2so4_calcu = np.where(np.asarray(h2so4_calcu) < 0, 1e-1, np.asarray(h2so4_calcu))

    # for observation
    nh3 = nh3 * 1e-6  # based on Table S1 title From 2017 Gordon.
    h2so4_track = h2so4_track * 1e-6  # table S1 is for 10-6
    # h2so4_calcu = h2so4_calcu * 1e-6
    # for model
    r_nh3 = r_nh3 * 1e-6
    g_nh3 = g_nh3 * 1e-6
    r_h2so4 = r_h2so4 * 1e-6
    g_h2so4 = g_h2so4 * 1e-6

    # divide the J into 4-piece, A[0-3]=Jb,n;A[4-7]=Jb,i;A[8-11]=Jt,n;A[12-15]=Jt,i;
    A = [3.95451, 9.702973, 12.62259, -0.007066146,
         3.373738, -11.48166, 25.49469, 0.1810722,
         2.891024, 182.4495, 1.203451, -4.188065,
         3.138719, -23.8002, 37.03029, 0.227413,
         8.003471, 1.5703478e-6,
         3.071246, 0.00483140]  # these values are from table s1
    # K does not matter for cal or mea! only related with T
    # for k_binary_neutral(observ+regional+global)
    k_b_n = np.exp(A[1] - np.exp(np.multiply(A[2], (np.divide(temp, 1000) - A[3]))))  # ('temp.shape', (17401,)
    k_b_n_r = np.exp(A[1] - np.exp(np.multiply(A[2], (np.divide(r_T, 1000) - A[3]))))  # regional model
    k_b_n_g = np.exp(A[1] - np.exp(np.multiply(A[2], (np.divide(g_T, 1000) - A[3]))))  # global model
    # for k_binary_ion(observe+regional+global)
    k_b_i = np.exp(
        A[5] - np.exp(np.multiply(A[6], (np.divide(temp, 1000) - A[7]))))  # temp in(17401), be care about the dimension
    k_b_i_r = np.exp(
        A[5] - np.exp(np.multiply(A[6], (np.divide(r_T, 1000) - A[7]))))  # temp in(17401), be care about the
    k_b_i_g = np.exp(
        A[5] - np.exp(np.multiply(A[6], (np.divide(g_T, 1000) - A[7]))))  # temp in(17401), be care about the
    # for k_ternary_neutral (observe+regional+global)
    k_t_n = np.exp(A[9] - np.exp(np.multiply(A[10], (np.divide(temp, 1000) - A[11]))))
    k_t_n_r = np.exp(A[9] - np.exp(np.multiply(A[10], (np.divide(r_T, 1000) - A[11]))))
    k_t_n_g = np.exp(A[9] - np.exp(np.multiply(A[10], (np.divide(g_T, 1000) - A[11]))))
    # for k_ternary_ion(observe+regional+global)
    k_t_i = np.exp(A[13] - np.exp(np.multiply(A[14], (np.divide(temp, 1000) - A[15]))))
    k_t_i_r = np.exp(A[13] - np.exp(np.multiply(A[14], (np.divide(r_T, 1000) - A[15]))))
    k_t_i_g = np.exp(A[13] - np.exp(np.multiply(A[14], (np.divide(g_T, 1000) - A[15]))))

    ##step1,get tmp_cs
    pi = 3.1415926
    mu = 1.2e-4
    e_elec = 1.6022e-19
    zboltz = 1.38064852e-23  # Boltzman Constant (kg m2 s-2 K-1 molec-1)(J/K, J=Nm=kg m/s2 m)
    # tmp_cs is in (17401) length,be careful!!
    tmp_cs = np.divide(np.multiply(4, np.multiply(pi, np.multiply(zboltz, np.multiply(temp, mu)))),
                       e_elec)  # corresponding to .f90 line#11
    tmp_cs_r = np.divide(np.multiply(4, np.multiply(pi, np.multiply(zboltz, np.multiply(r_T, mu)))),
                         e_elec)  # corresponding to .f90 line#11
    tmp_cs_g = np.divide(np.multiply(4, np.multiply(pi, np.multiply(zboltz, np.multiply(g_T, mu)))),
                         e_elec)  # corresponding to .f90 line#11

    # step2,cal the csink_ion=get wet_dp->2 dim ND(num density)-?plug in CSINK_ION(p2)
    # uhsas(>100nm) wet_dp,we have 31 data
    cut_off_size_100 = [0.097, 0.105, 0.113, 0.121, 0.129, 0.145, 0.162, 0.182, 0.202, 0.222, 0.242, 0.262,
                        0.282, 0.302, 0.401, 0.57, 0.656, 0.74, 0.833, 0.917, 1.008, 1.148, 1.319, 1.479,
                        1.636, 1.796, 1.955, 2.184, 2.413, 2.661, 2.991]  # 31bin, in um
    cut_off_size_100 = np.multiply(cut_off_size_100, 1e-6)  # um to m

    # get ND term(number of particles in mode, should be 2 dimension)
    file_100 = Dataset(
        '/jet/home/ding0928/Colorado/EOLdata/dataaw2NKX/RF05Z.20140731.195400_011100.PNI.nc')
    data_100 = (file_100.variables['CS200_RPI'][:, 0, :])
    # print('data_100.shape', data_100.shape)  # ('data_100.shape', (17401, 31))
    uhsas_time = (file_100.variables['Time'][:])
    # print('uhsas_time', uhsas_time)
    # print('uhsas_time.shape', uhsas_time.shape)  # ('uhsas_time.shape', (17401,))

    # cal the uhsas_cs_ion
    cs_ion_uhsas = np.zeros(data_100[:, 0].shape)  # size, tmp_cs=17401
    for i in range(31):
        term_uhsas = np.multiply(np.multiply(data_100[:, i], cut_off_size_100[i]), tmp_cs) * 0.5 * 1e6
        # tmp_cs=17401, data_100(17401, 31))
        cs_ion_uhsas = cs_ion_uhsas + term_uhsas  # see cs.py line 204
        # print('cs_ion_uhsas.shape', np.array(cs_ion_uhsas).shape) #(17401)
        # print('cs_ion_uhsas', np.array(cs_ion_uhsas))
    # print('cs_ion_uhsas', np.array(cs_ion_uhsas))  # ('cs_ion_uhsas', array([0.00353061, 0.00314373, 0.00429001, ...,
    # 0.01583359, 0.03460288, 0.04149793]))
    # print('cs_ion_uhsas.shape', np.array(cs_ion_uhsas).shape)  # ('cs_ion_uhsas.shape', (17401,))

    # second, load <100nm smps data for nuc+ai
    c = range(20, 34)  # in excel 21-34
    data_smps = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/frappe-SMPS_C130_20140731_R0.ict', delimiter=',',
                           skiprows=66, usecols=c)  #
    smps_time = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/frappe-SMPS_C130_20140731_R0.ict', delimiter=',',
                           skiprows=66, usecols=0)  # time resolution is different from uhsas,(202,1)
    # print('smps_time_shape', smps_time.shape)  # ('smps_time_shape', (272,))
    # print('data_smps_shape', data_smps.shape)  # ('data_smps_shape', (272, 14))

    # mask out the <0 for data_smps
    data_smps = ma.masked_where(data_smps < 0, data_smps)
    smps_time2 = smps_time[~data_smps[:, 0].mask]
    data_smps_2 = data_smps[~data_smps[:, 0].mask]
    # print('data_smps_2.shape', data_smps_2.shape)  # ('data_smps_2.shape', (220, 14))

    # data_smps_new2 = data_smps_2[~data_smps_2.mask.any(axis=1)]
    # print(data_smps_new2)
    # print('data_smps_new2.shape', data_smps_new2.shape)  # ('data_smps_new2.shape', (218, 14))
    # print('smps_time2.shape', smps_time2.shape)  # ('smps_time2.shape', (220,))

    # read in the time for interp1d??
    time_NAV = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',
                          delimiter=',',
                          skiprows=132, usecols=0)
    # smps wet_dp,we have 14 data
    cut_off_size_smps = [8.4, 10.1, 12.1, 14.5, 17.4, 20.9, 25.1, 30.3, 36.5, 44.1, 53.5, 65, 79, 96.4]  # 15bin,in nm
    cut_off_size_smps = np.multiply(cut_off_size_smps, 1e-9)  # nm to m

    # now calculate the cs_ion_smps
    # since tmp_cs(17401) and smps(212) has different time resolution, so interp first
    gas_track_data = []
    for i in range(14):  # 0-1-2...-13
        # step1:mask
        data_smps_new = data_smps[:, i]
        data_smps_new2 = data_smps_new[~data_smps_new.mask]  # remove the masked
        smps_time2 = smps_time[~data_smps_new.mask]

        # step 2: interp to 17401
        interp_gas = interp1d(smps_time2, data_smps_new2, kind='nearest',
                              fill_value='extrapolate')  # smps_time(202,1)? data_smps(202,15) doesn't match?
        gas_track = interp_gas(np.asarray(time_NAV))  # method borrowed from long script
        gas_track_data.append(gas_track)

    data_smps_trans = np.asarray(gas_track_data).reshape(19021, 14)

    cs_ion_smps = np.zeros(data_100[:, 0].shape)  # now should be 17401
    for i in range(14):
        term_smps = np.multiply(np.multiply(data_smps_trans[:, i], cut_off_size_smps[i]),
                                tmp_cs) * 0.5 * 1e6
        cs_ion_smps = cs_ion_smps + term_smps  # see cs.py line 204

    cs_ion = np.array(cs_ion_smps) + np.array(cs_ion_uhsas)
    # print('cs_ion', cs_ion)

    ## now calculate the cs_ion for reginal model, be careful about the 0.5 yes/no??
    r_cs_ion_Ait = np.multiply(np.multiply(r_N_Ait, r_D_Ait), tmp_cs_r) * 0.5 * 1e6
    r_cs_ion_Ait_in = np.multiply(np.multiply(r_N_Ait_In, r_D_Ait_In), tmp_cs_r) * 0.5 * 1e6
    r_cs_ion_Acu = np.multiply(np.multiply(r_N_Acu, r_D_Acu), tmp_cs_r) * 0.5 * 1e6
    r_cs_ion_Cor = np.multiply(np.multiply(r_N_Cor, r_D_Cor), tmp_cs_r) * 0.5 * 1e6
    cs_ion_r = r_cs_ion_Ait + r_cs_ion_Ait_in + r_cs_ion_Acu + r_cs_ion_Cor
    # print('r_cs_ion_value', cs_ion_r)

    ## now calculate the cs_ion for reginal model
    g_cs_ion_Ait = np.multiply(np.multiply(g_N_Ait, g_D_Ait), tmp_cs_g) * 0.5 * 1e6
    g_cs_ion_Ait_in = np.multiply(np.multiply(g_N_Ait_In, g_D_Ait_In), tmp_cs_g) * 0.5 * 1e6
    g_cs_ion_Acu = np.multiply(np.multiply(g_N_Acu, g_D_Acu), tmp_cs_g) * 0.5 * 1e6
    g_cs_ion_Cor = np.multiply(np.multiply(g_N_Cor, g_D_Cor), tmp_cs_g) * 0.5 * 1e6
    cs_ion_g = g_cs_ion_Ait + g_cs_ion_Ait_in + g_cs_ion_Acu + g_cs_ion_Cor
    # print('g_cs_ion_value', cs_ion_g)

    ###step3, get ion_con[n-]; We do measure fisrt
    crii_g = 5  # (pairs/cm3 s)
    ## now get the alfa,# DOESNT matter either mea or cal
    alfa_term_a = np.multiply(6e-8, np.sqrt(np.divide(300, temp)))
    alfa_term_a_r = np.multiply(6e-8, np.sqrt(np.divide(300, r_T)))
    alfa_term_a_g = np.multiply(6e-8, np.sqrt(np.divide(300, g_T)))

    alfa_term_b = np.multiply(np.multiply(6e-26, np.power(np.divide(300, temp), 4)), M)
    alfa_term_b_r = np.multiply(np.multiply(6e-26, np.power(np.divide(300, r_T), 4)), r_M)
    alfa_term_b_g = np.multiply(np.multiply(6e-26, np.power(np.divide(300, g_T), 4)), g_M)

    alfa = alfa_term_a + alfa_term_b
    alfa_r = alfa_term_a_r + alfa_term_b_r
    alfa_g = alfa_term_a_g + alfa_term_b_g

    # for measurement H2SO4 data(track)+r/g as well
    a_t_n_mea = A[17] + np.divide(np.power(h2so4_track, A[8]), np.power(nh3, A[16]))
    a_t_n_r = A[17] + np.divide(np.power(r_h2so4, A[8]), np.power(r_nh3, A[16]))  # model line
    a_t_n_g = A[17] + np.divide(np.power(g_h2so4, A[8]), np.power(g_nh3, A[16]))

    a_t_i_mea = A[19] + np.divide(np.power(h2so4_track, A[12]), np.power(nh3, A[18]))
    a_t_i_r = A[19] + np.divide(np.power(r_h2so4, A[12]), np.power(r_nh3, A[18]))
    a_t_i_g = A[19] + np.divide(np.power(g_h2so4, A[12]), np.power(g_nh3, A[18]))

    f_t_n_mea = np.divide(nh3, a_t_n_mea)
    f_t_n_r = np.divide(r_nh3, a_t_n_r)
    f_t_n_g = np.divide(g_nh3, a_t_n_g)

    f_t_i_mea = np.divide(nh3, a_t_i_mea)
    f_t_i_r = np.divide(r_nh3, a_t_i_r)
    f_t_i_g = np.divide(g_nh3, a_t_i_g)

    # now get the X_mea term; plus model(same thing but change into model_term)
    sa_bi_mes = np.power(h2so4_track, A[4])  # [H2SO4]^P(b,i),P(b,i)=a4
    sa_bi_r = np.power(r_h2so4, A[4])  # [H2SO4]^P(b,i),P(b,i)=a4
    sa_bi_g = np.power(g_h2so4, A[4])  # [H2SO4]^P(b,i),P(b,i)=a4

    sa_ti_mes = np.power(h2so4_track, A[12])  # [H2SO4]^P(t,i),P(t,i)=a12
    sa_ti_r = np.power(r_h2so4, A[12])  # [H2SO4]^P(t,i),P(t,i)=a12
    sa_ti_g = np.power(g_h2so4, A[12])  # [H2SO4]^P(t,i),P(t,i)=a12

    x_b_mes = np.multiply(sa_bi_mes, k_b_i)
    x_b_r = np.multiply(sa_bi_r, k_b_i_r)
    x_b_g = np.multiply(sa_bi_g, k_b_i_g)

    x_t_mes = np.multiply(np.multiply(sa_ti_mes, k_t_i), f_t_i_mea)
    x_t_r = np.multiply(np.multiply(sa_ti_r, k_t_i_r), f_t_i_r)
    x_t_g = np.multiply(np.multiply(sa_ti_g, k_t_i_g), f_t_i_g)

    x_total_mes = cs_ion + x_b_mes + x_t_mes  # fortran line #211,X = cs_ion + sa_bi * sa_bi_coeff + k_tch * sa_ti
    x_total_r = cs_ion_r + x_b_r + x_t_r  # fortran line #211,X = cs_ion + sa_bi * sa_bi_coeff + k_tch * sa_ti
    x_total_g = cs_ion_g + x_b_g + x_t_g  # fortran line #211,X = cs_ion + sa_bi * sa_bi_coeff + k_tch * sa_ti

    # sqrt_term_mes=X ** 2 + 4. * ALFA * CRII_G
    sqrt_term_mes = np.power(x_total_mes, 2) + np.multiply(np.multiply(alfa, crii_g), 4.0)
    sqrt_term_r = np.power(x_total_r, 2) + np.multiply(np.multiply(alfa_r, crii_g), 4.0)
    sqrt_term_g = np.power(x_total_g, 2) + np.multiply(np.multiply(alfa_g, crii_g), 4.0)
    # get the [n-]~ion_con_mes, first for mes
    ion_conc_mes_a = np.sqrt(sqrt_term_mes)
    ion_conc_r_a = np.sqrt(sqrt_term_r)
    ion_conc_g_a = np.sqrt(sqrt_term_g)

    ion_conc_mes_nume = ion_conc_mes_a - x_total_mes
    ion_conc_r_nume = ion_conc_r_a - x_total_r
    ion_conc_g_nume = ion_conc_g_a - x_total_g

    ion_conc_mes_deno = 2 * alfa
    ion_conc_r_deno = 2 * alfa_r
    ion_conc_g_deno = 2 * alfa_g

    ion_conc_mes = np.divide(ion_conc_mes_nume,
                             ion_conc_mes_deno)  # ION_CONC(JL) = (SQRT(X ** 2 + 4. * ALFA * CRII_G(JL)) - X) / (2 * ALFA)
    ion_conc_r = np.divide(ion_conc_r_nume,
                           ion_conc_r_deno)  # ION_CONC(JL) = (SQRT(X ** 2 + 4. * ALFA * CRII_G(JL)) - X) / (2 * ALFA)
    ion_conc_g = np.divide(ion_conc_g_nume,
                           ion_conc_g_deno)  # ION_CONC(JL) = (SQRT(X ** 2 + 4. * ALFA * CRII_G(JL)) - X) / (2 * ALFA)

    # get the J
    n_con_mes = ion_conc_mes  # will figure this out later and only this term left to get J
    n_con_r = ion_conc_r  #
    n_con_g = ion_conc_g  #

    J_b_mea = np.multiply(k_b_n, np.power(h2so4_track, A[0])) + np.multiply(np.multiply(k_b_i, n_con_mes),
                                                                            sa_bi_mes)
    J_b_r = np.multiply(k_b_n_r, np.power(r_h2so4, A[0])) + np.multiply(np.multiply(k_b_i_r, n_con_r),
                                                                        sa_bi_r)
    J_b_g = np.multiply(k_b_n_g, np.power(g_h2so4, A[0])) + np.multiply(np.multiply(k_b_i_g, n_con_g),
                                                                        sa_bi_g)

    J_t_mea = np.multiply(np.multiply(k_t_n, f_t_n_mea), np.power(h2so4_track, A[8])) \
              + np.multiply(np.multiply(np.multiply(k_t_i, f_t_i_mea), sa_ti_mes), n_con_mes)
    J_t_r = np.multiply(np.multiply(k_t_n_r, f_t_n_r), np.power(r_h2so4, A[8])) \
            + np.multiply(np.multiply(np.multiply(k_t_i_r, f_t_i_r), sa_ti_r), n_con_r)
    J_t_g = np.multiply(np.multiply(k_t_n_g, f_t_n_g), np.power(g_h2so4, A[8])) \
            + np.multiply(np.multiply(np.multiply(k_t_i_g, f_t_i_g), sa_ti_g), n_con_g)

    J_mea = J_t_mea + J_b_mea
    J_r = J_t_r + J_b_r
    J_g = J_t_g + J_b_g
    # print('J_b_mea', J_b_mea)
    # print('J_t_mea', J_t_mea)
    # print('J_mea', J_mea)
    print('J_r.shape', np.asarray(J_r.shape))  # J_r.shapebarray([318])
    print('J_g.shape', np.asarray(J_g.shape))  # ('J_g.shape array([318]))
    print('J_mea.shape', np.asarray(J_mea.shape))  # ('J_mea.shape array([19021]))

    df = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf'
                     '/0731_comparison/interpolated_korus_dataset_20140731.csv', index_col=0, parse_dates=True)
    import pytz
    mountain = pytz.timezone('US/Mountain')  # define UTC
    df.index = df.index.tz_localize(pytz.utc)
    df.index = df.index.tz_convert(mountain)  # from UTC to mountain time
    # indexed = df.set_index(df['time'], append=True)
    # df = indexed.resample('1T', on='time').mean()
    print('df', df)
    # print(df['OH'])

    # plot for qual2_make it high dpi
    date = '20140731'
    unix_epoch_of_31July2014_0001 = 1406764800
    if date == '20140731':
        trackpdtime = pd.to_datetime(np.asarray(time_NAV) + unix_epoch_of_31July2014_0001, unit='s')


    fig = plt.figure(constrained_layout=True)
    ax1 = fig.add_subplot(gs[1, :])
    # fig.add_subplot(gs[1, :])
    d5 = {'time': pd.Series(trackpdtime), 'J_r': pd.Series(np.asarray(J_r)), 'J_g': pd.Series(np.asarray(J_g))}
    df_6 = pd.DataFrame(d5)
    df_6_2 = df_6[['time', 'J_r', 'J_g']]
    indexed = df_6_2.set_index(df_6_2['time'], append=True)
    df_5 = indexed.resample('1T', on='time').mean()
    df_5['J_r'].plot()
    df_5['J_g'].plot()
    plt.grid(True, axis='both')
    plt.ylabel('Nulceation rate \n #cc$^{-3}$s$^{-1}$')
    plt.yscale('log')
    plt.ylim(1e-4, 1e-1)
    ax1.legend(loc='upper left')
    # xmin = '2014-07-31 12:00:00'
    # xmax = '2014-07-31 17:30:00'
    # plt.xlim(xmin, xmax)
    # # now plot it, fig1
    fig = plt.figure(figsize=(8, 8))

    ax2 = ax1.twinx()
    d2 = {'time': pd.Series(trackpdtime), 'J_mea': pd.Series(np.asarray(J_mea))}
    df_3 = pd.DataFrame(d2)
    df_3_2 = df_3[['time', 'J_mea']]
    indexed = df_3_2.set_index(df_3_2['time'], append=True)
    df_2 = indexed.resample('1T', on='time').mean()
    df_2['J_mea'].plot(label="J_mea", color='green')
    # plt.legend(["J_mea"])
    ax2.legend(loc='upper right')
    # plt.xticks(color='w')
    xmin = '2014-07-31 12:00:00'
    xmax = '2014-07-31 17:30:00'
    plt.xlim(xmin, xmax)

    plt.show()


read_c130_smps()
J_rate()
