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
from box_model_rerun_step4 import h2so4_cal_box_model
import seaborn as sns

# from colorado_all import colorado_mod_J

# Model input: read in the long python script(model 4d interpolation->1d) from csv
# first for reginal model
model_time = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                         '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                         usecols=["time"])
# print('model_time_value', model_time)  # 'model_time.shape', (291, 0))
model_time = np.asarray(model_time).reshape(318)
print('model_time.shape', model_time.shape)
print('type(model_time[0])', type(model_time[0]))
print('model_time[0]', model_time[0])

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
# print('r_h2so4.shape', r_h2so4.shape)  # ('r_h2so4.shape', (291, 0))

r_nh3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                    '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                    usecols=["model_NH3"])  # pptv
r_nh3 = np.asarray(r_nh3).reshape(318)
r_nh3 = np.multiply(np.multiply(np.divide(np.multiply(r_p, 17), np.multiply(8.314, r_T)), r_nh3), 1e-6)  # ppt*1e-6=ppm
r_nh3 = np.multiply(r_nh3, 3.5e10)  # from ug m-3 to molecule cm-3
# print('r_nh3.shape', r_nh3.shape)  # ('r_nh3.shape', (291, 0))

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
print('g_h2so4.shape', g_h2so4.shape)  # ('g_h2so4.shape', (291, 0))

g_nh3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                    '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                    usecols=["glm_NH3"])  # pptv
g_nh3 = np.asarray(g_nh3).reshape(318)
# g_nh3 = (g_p * 17) / (8.314 * g_T) * g_nh3 * 1e-6  # ppt*1e-6=ppm
# g_nh3 = g_nh3 * 3.5e10  # from ug m-3 to molecule cm-3
g_nh3 = np.multiply(np.multiply(np.divide(np.multiply(g_p, 17), np.multiply(8.314, g_T)), g_nh3), 1e-6)  # ppt*1e-6=ppm
g_nh3 = np.multiply(g_nh3, 3.5e10)  # from ug m-3 to molecule cm-3
print('g_nh3.shape', g_nh3.shape)  # ('g_nh3.shape', (291, 0))

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
                           '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                           usecols=["altitude"])  # m
altitude_obs = np.asarray(altitude_obs).reshape(318)

## Now read in the 8=4 conc+4 dia first for regional
r_N_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["model_N_Ait"])  # not sure the unit??
r_N_Ait = np.asarray(r_N_Ait).reshape(318)
r_N_Ait_In = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                         '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
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
                         '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
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
                         '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
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
                         '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
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
print('nh3_value', nh3)
print('h2so4_track', h2so4_track)
# print('h2so4_calcu', h2so4_calcu)
# for observation
nh3 = nh3 * 1e-6  # based on Table S1 title From 2017 Gordon.
h2so4_track = h2so4_track * 1e-6  # table S1 is for 10-6
# h2so4_calcu = h2so4_calcu * 1e-6
# for model
r_nh3 = r_nh3 * 1e-6
g_nh3 = g_nh3 * 1e-6
r_h2so4 = r_h2so4 * 1e-6
g_h2so4 = g_h2so4 * 1e-6
# print('r_nh3_value', r_nh3)
# print('r_h2so4_value', r_h2so4)
# print('r_so2_value', r_so2)  # e8~e9
# print('g_nh3_value', g_nh3)
# print('g_h2so4_value', g_h2so4)
# print('g_so2_value', g_so2)

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
print('data_100.shape', data_100.shape)  # ('data_100.shape', (17401, 31))
uhsas_time = (file_100.variables['Time'][:])
print('uhsas_time', uhsas_time)
print('uhsas_time.shape', uhsas_time.shape)  # ('uhsas_time.shape', (17401,))

# cal the uhsas_cs_ion
cs_ion_uhsas = np.zeros(data_100[:, 0].shape)  # size, tmp_cs=17401
for i in range(31):
    term_uhsas = np.multiply(np.multiply(data_100[:, i], cut_off_size_100[i]), tmp_cs) * 0.5 * 1e6
    # tmp_cs=17401, data_100(17401, 31))
    cs_ion_uhsas = cs_ion_uhsas + term_uhsas  # see cs.py line 204
    # print('cs_ion_uhsas.shape', np.array(cs_ion_uhsas).shape) #(17401)
    # print('cs_ion_uhsas', np.array(cs_ion_uhsas))
print('cs_ion_uhsas', np.array(cs_ion_uhsas))  # ('cs_ion_uhsas', array([0.00353061, 0.00314373, 0.00429001, ...,
# 0.01583359, 0.03460288, 0.04149793]))
print('cs_ion_uhsas.shape', np.array(cs_ion_uhsas).shape)  # ('cs_ion_uhsas.shape', (17401,))

# second, load <100nm smps data for nuc+ai
c = range(20, 34)  # in excel 21-34
data_smps = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/frappe-SMPS_C130_20140731_R0.ict', delimiter=',',
                       skiprows=66, usecols=c)  #
smps_time = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/frappe-SMPS_C130_20140731_R0.ict', delimiter=',',
                       skiprows=66, usecols=0)  # time resolution is different from uhsas,(202,1)
print('smps_time_shape', smps_time.shape)  # ('smps_time_shape', (272,))
print('data_smps_shape', data_smps.shape)  # ('data_smps_shape', (272, 14))

# mask out the <0 for data_smps
data_smps = ma.masked_where(data_smps < 0, data_smps)
smps_time2 = smps_time[~data_smps[:, 0].mask]
data_smps_2 = data_smps[~data_smps[:, 0].mask]
print('data_smps_2.shape', data_smps_2.shape)  # ('data_smps_2.shape', (220, 14))

# data_smps_new2 = data_smps_2[~data_smps_2.mask.any(axis=1)]
# print(data_smps_new2)
# print('data_smps_new2.shape', data_smps_new2.shape)  # ('data_smps_new2.shape', (218, 14))
print('smps_time2.shape', smps_time2.shape)  # ('smps_time2.shape', (220,))

# read in the time for interp1d??
time_NAV = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',
                      delimiter=',',
                      skiprows=132, usecols=0)
print('time_NAV.shape', time_NAV.shape)  # ('time_NAV.shape', (17401,))
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
    print('data_smps_new.shape', data_smps_new.shape)  # ('data_smps_new.shape', (272,))
    print('data_smps_new2.shape', data_smps_new2.shape)  # ('data_smps_new2.shape', (220,))
    print('smps_time2.shape', smps_time2.shape)  # ('smps_time2.shape', (220,))

    # step 2: interp to 17401
    interp_gas = interp1d(smps_time2, data_smps_new2, kind='nearest',
                          fill_value='extrapolate')  # smps_time(202,1)? data_smps(202,15) doesn't match?
    gas_track = interp_gas(np.asarray(time_NAV))  # method borrowed from long script
    gas_track_data.append(gas_track)
    # print('gas_track_data', gas_track_data)
    print('gas_track_data_shape', np.asarray(gas_track_data).shape)  # ('gas_track_data_shape', (14, 17401))

data_smps_trans = np.asarray(gas_track_data).reshape(19021, 14)
print('data_smps_trans_shape', np.asarray(data_smps_trans).shape)  # ('data_smps_trans_shape', (17401, 14))

cs_ion_smps = np.zeros(data_100[:, 0].shape)  # now should be 17401
print('cs_ion_smps_shape', cs_ion_smps.shape)
for i in range(14):
    term_smps = np.multiply(np.multiply(data_smps_trans[:, i], cut_off_size_smps[i]),
                            tmp_cs) * 0.5 * 1e6
    cs_ion_smps = cs_ion_smps + term_smps  # see cs.py line 204
    # print('cs_ion_smps.shape', np.array(cs_ion_smps).shape)
    # print('cs_ion_smps', np.array(cs_ion_smps))

# print('cs_ion_smps', np.array(cs_ion_smps))  # ('cs_ion_smps', array([0.02489761, 0.02486868, 0.03833552, ...,
# 0.01414378, 0.01414106,0.01413941]))
print('cs_ion_smps.shape', np.array(cs_ion_smps).shape)  # ('cs_ion_smps.shape', (17401,))

# total cs=uhsas+smps;# doesnt matter either mea or cal
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
# crii_g=5 (pairs/cm3 s)
crii_g = 5
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
n_con_r = ion_conc_r  # will figure this out later and only this term left to get J
n_con_g = ion_conc_g  # will figure this out later and only this term left to get J
# print('n_con_mes', n_con_mes)
# print('n_con_r', n_con_r)
# print('n_con_g', n_con_g)

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
print('J_r.shape', np.asarray(J_r.shape))
print('J_g.shape', np.asarray(J_g.shape))

# before we start plot the fig, need to solve the time format problem:
# first transfer the time into python time format(copied from long python)
# time_NAV = pd.to_datetime(np.asarray(time_NAV),format='%S')
date = '20140731'
unix_epoch_of_31July2014_0001 = 1406764800
if date == '20140731':
    trackpdtime = pd.to_datetime(np.asarray(time_NAV) + unix_epoch_of_31July2014_0001, unit='s')
elif date == '20140801':
    unix_epoch_of_01August2014_0001 = unix_epoch_of_31July2014_0001 + 86400
    trackpdtime = pd.to_datetime(np.asarray(time_NAV) + unix_epoch_of_01August2014_0001, unit='s')
elif date == '20140802':
    unix_epoch_of_02August2014_0001 = unix_epoch_of_31July2014_0001 + 86400 * 2
    trackpdtime = pd.to_datetime(np.asarray(time_NAV) + unix_epoch_of_02August2014_0001, unit='s')
# print(trackpdtime)

##the same thing but for cal
# # for calculated H2SO4 data(calcu)
# a_t_n_cal = A[17] + np.divide(np.power(h2so4_calcu, A[8]), np.power(nh3, A[16]))
# a_t_i_cal = A[19] + np.divide(np.power(h2so4_calcu, A[12]), np.power(nh3, A[18]))
# f_t_n_cal = np.divide(nh3, a_t_n_cal)
# f_t_i_cal = np.divide(nh3, a_t_i_cal)
#
# # now get the X_calc term
# sa_bi_cal = np.power(h2so4_calcu, A[4])  # [H2SO4]^P(b,i)
# sa_ti_cal = np.power(h2so4_calcu, A[12])  # [H2SO4]^P(t,i)
# x_b_cal = np.multiply(sa_bi_cal, k_b_i)
# x_t_cal = np.multiply(np.multiply(sa_ti_cal, k_t_i), f_t_i_cal)
# x_total_cal = cs_ion + x_b_cal + x_t_cal  # fortran line #211,X = cs_ion + sa_bi * sa_bi_coeff + k_tch * sa_ti + org_bi_ch
#
# # sqrt_term_cal=X ** 2 + 4. * ALFA * CRII_G
# sqrt_term_cal = np.power(x_total_cal, 2) + np.multiply(np.multiply(alfa, crii_g), 4.0)
# # get the [n-]~ion_con_cal, then for cal
# ion_conc_cal_a = np.sqrt(sqrt_term_cal)
# ion_conc_cal_nume = ion_conc_cal_a - x_total_cal
# ion_conc_cal_deno = 2 * alfa
# ion_conc_cal = np.divide(ion_conc_cal_nume,
#                          ion_conc_cal_deno)  # ION_CONC(JL) = (SQRT(X ** 2 + 4. * ALFA * CRII_G(JL)) - X) / (2 * ALFA)
#
# n_con_cal = ion_conc_cal  # will figure this out later and only this term left to get J
# J_b_cal = np.multiply(k_b_n, np.power(h2so4_calcu, A[0])) + np.multiply(np.multiply(k_b_i, n_con_cal),
#                                                                         sa_bi_cal)
# J_t_cal = np.multiply(np.multiply(k_t_n, f_t_n_cal), np.power(h2so4_calcu, A[8])) \
#           + np.multiply(np.multiply(np.multiply(k_t_i, f_t_i_cal), sa_ti_cal), n_con_cal)
# J_cal = J_t_cal + J_b_cal
# # print('J_cal.shape', np.asarray(J_cal.shape))

# now plot it, fig1
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.plot(np.asarray(model_time), np.asarray(J_r), color='g')
ax.plot(np.asarray(model_time), np.asarray(J_b_r), color='g', linestyle=':')
ax.plot(np.asarray(model_time), np.asarray(J_g), color='b')
ax.plot(np.asarray(model_time), np.asarray(J_b_g), color='b', linestyle=':')
tick_spacing = 75
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.title('0731 J comparison')
# plt.xticks(rotation=15)
# plt.grid(linestyle='-.')plt.xlabel('time', fontsize=12)
plt.ylabel('Nucleation Rate J,# cm-3-s', fontsize=12)
# plt.setp(ax.get_yticklabels(), fontsize=6)
plt.legend(["J_r", "J_b_r", "J_g", "J_b_g"], fontsize=14)
plt.yscale('log')
# plt.locator_params(nbins=10, axis='x')
plt.ylim(1e-10, 8e0)
plt.grid(linestyle='-.')  # subplot1 is for J(mea+box+r+g)
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/0112_Jbt_model.pdf')

# for J observation
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(J_mea), color='coral')
ax.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(J_b_mea), color='coral', linestyle=':')
plt.ylabel('Nucleation Rate J,# cm-3-s')
# plt.setp(ax.get_yticklabels(), fontsize=6)
plt.legend(["J_mea", "J_b_mea"], fontsize=14)
plt.yscale('log')
# plt.locator_params(nbins=10, axis='x')
plt.ylim(1e-12, 8e0)
plt.grid(linestyle='-.')  # subplot1 is for J(mea+box+r+g)
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/0112_Jbt_observ2.pdf')

plt.figure()
ax = fig.add_subplot(111)
index = pd.Series(trackpdtime)  #
series_J_cal = pd.Series(np.asarray(J_b_mea), index=index)
# print('series_J_cal value', series_J_cal)
series_J_cal.plot(kind='line', grid=True, legend=True, ylim=[1e-10, 8e0], label='J_b_mea', linestyle=':', color='black',
                  logy=True)

series_J_mea = pd.Series(np.asarray(J_mea), index=index)
# print('series_J_mea value', series_J_mea)
series_J_mea.plot(kind='line', grid=True, legend=True, ylim=[1e-10, 8e0], label='J_mea', linestyle=':', color='black',
                  logy=True)
plt.grid(linestyle='-.')
plt.ylabel('Nucleation Rate J', fontsize=12)
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/0112_Jbt_observ.pdf')
# ## for regional+global
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# ax.plot(np.asarray(model_time), np.asarray(J_r))
# ax.plot(np.asarray(model_time), np.asarray(J_g))
# tick_spacing = 80
# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# # plt.title('0802 J r+g')
# # plt.xticks(rotation=45)
# plt.grid(linestyle='-.')# plt.ylabel('Nucleation Rate J')
# plt.legend(["J_r", "J_g"])
# plt.yscale('log')
# # plt.locator_params(nbins=10, axis='x')
# plt.ylim(1e-10, 8e0)
# plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/J_um_model.png')
# plt.show()
# plt.plot(np.asarray(trackpdtime), np.asarray(J_mea))
# plt.yscale('log')
# plt.grid(linestyle='-.')# plt.xlabel('time')
# plt.ylabel('nuclation rate(J), #/(cm3.s)')
# plt.ylim(1e-13, 8e-4)
# # plt.ylim(0, 3e3)
# # plt.legend(["J_mes"])
# plt.plot(np.asarray(trackpdtime), np.asarray(J_cal))
# plt.xlabel('time')
# plt.ylabel('nuclation rate(J), #/(cm3.s)')
# plt.yscale('log')
# # plt.grid(linestyle='-.')# plt.ylim(1e-13, 1)
# plt.title('0802 J')
# plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/J.png')
# plt.figure()

# Now draw the model J+ gas concentration??
# step 1, g+r but only for double check with T
fig = plt.figure(figsize=(8, 5))
plt.yscale('log')
ax = fig.add_subplot(111)
plt.yscale('log')
ax.plot(np.asarray(model_time), np.asarray(J_r), color='green', label='J_r')
ax.plot(np.asarray(model_time), np.asarray(J_g), color='purple', label='J_g')
tick_spacing = 80
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_ylabel('Nucleation Rate J(molecule cm-3 s-1)')
ax.set_title("J & gas concentration(0731) for um11.9")
ax.set_ylim = (1e-6, 8e0)
ax.legend(loc=2)
ax.grid()

ax2 = ax.twinx()  # double-y
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_h2so4), 10), '-.', label='r_h2so4(1e-7)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_nh3), 1e5), '-.', label='r_nh3(1e-11)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_so2), 1e9), '-.', label='r_so2(1e-9)')
ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_T), 100), '-.', label='r_T(1e-2)')
# plt.yscale('log')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_h2so4), 10), '-.', label='g_h2so4(1e-7)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_nh3), 1e6), '-.', label='g_nh3(1e-12)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_so2), 1e9), '-.', label='g_so2(1e-9)')
ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_T), 100), '-.', label='g_T(1e-2)')
ax2.plot(np.asarray(model_time), np.divide(np.asarray(altitude_obs), 2e3), '-.', color='black',
         label='altitude_flight(5e-4)')

# plt.yscale('log')
ax2.set_ylabel('gas concentration(molecule cm-3) T')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.legend(loc=1)
ax2.set_ylim(0, 3.5)
# plt.yscale('log')
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/double_y_T.png')

# step2, only for r
fig = plt.figure(figsize=(8, 5))
plt.yscale('log')
ax = fig.add_subplot(111)
plt.yscale('log')
ax.plot(np.asarray(model_time), np.asarray(J_r), color='green', label='J_r')
# ax.plot(np.asarray(model_time), np.asarray(J_g), color='purple', label='J_g')
tick_spacing = 80
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_ylabel('Nucleation Rate J(molecule cm-3 s-1)')
ax.set_title("J & gas concentration(0731) for um11.9 Regional")
ax.set_ylim = (1e-6, 8e0)
ax.legend(loc=2)
ax.grid()

ax2 = ax.twinx()  # double-y
ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_h2so4), 1e2), '-.', label='r_h2so4(1e-8)')  # already e-6
ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_nh3), 0.25e5), '-.', label='r_nh3(4e-11)')  # already e-6
ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_so2), 1e11), '-.', label='r_so2(1e-11)')
ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_T), 1000), '-.', label='r_T(1e-3)')
ax2.plot(np.asarray(model_time), np.multiply(np.asarray(r_cs), 100), '-.', label='r_cs(1e2)')
# plt.yscale('log')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_h2so4), 10), '-.', label='g_h2so4(1e-7)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_nh3), 1e6), '-.', label='g_nh3(1e-12)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_so2), 1e9), '-.', label='g_so2(1e-9)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_T), 100), '-.', label='g_T(1e-2)')

ax2.set_ylabel('gas concentration(molecule cm-3) T')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.legend(loc=1)
ax2.set_ylim(0, 1)
# plt.yscale('log')
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/double_y_R.png')

# #step3, only for g
# Now draw the model J+ gas concentration??
fig = plt.figure(figsize=(8, 5))
plt.yscale('log')
ax = fig.add_subplot(111)
plt.yscale('log')
# ax.plot(np.asarray(model_time), np.asarray(J_r), color='yellow', label='J_r')
ax.plot(np.asarray(model_time), np.asarray(J_g), color='purple', label='J_g')
tick_spacing = 80
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_ylabel('Nucleation Rate J(molecule cm-3 s-1)')
ax.set_title("J & gas concentration(0731) for um11.9 Global")
ax.set_ylim = (1e-6, 8e0)
ax.legend(loc=2)
ax.grid()

ax2 = ax.twinx()  # double-y
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_h2so4), 10), '-.', label='r_h2so4(1e-7)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_nh3), 1e6), '-.', label='r_nh3(1e-12)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_so2), 1e9), '-.', label='r_so2(1e-9)')
# ax2.plot(np.asarray(model_time), np.divide(np.asarray(r_T), 100), '-.', label='r_T(1e-2)')
# plt.yscale('log')
ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_h2so4), 10), '-.', label='g_h2so4(1e-7)')  ## already e-6
ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_nh3), 0.25e5), '-.', label='g_nh3(4e-11)')  ## already e-6
ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_so2), 1e10), '-.', label='g_so2(1e-10)')
ax2.plot(np.asarray(model_time), np.divide(np.asarray(g_T), 1000), '-.', label='g_T(1e-3)')
ax2.plot(np.asarray(model_time), np.multiply(np.asarray(g_cs), 100), '-.', label='g_cs(1e2)')

ax2.set_ylabel('gas concentration(molecule cm-3) T')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.legend(loc=1)
ax2.set_ylim(0, 1)
# plt.yscale('log')
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/double_y_g.png')

# plt.subplot(222)
# try my own way to resample
# index = pd.Series(trackpdtime)  # here is super useful!!!!!
# series_J_cal = pd.Series(np.asarray(J_cal), index=index)
# # print('series_J_cal value', series_J_cal)
# series_J_cal.plot(kind='line', grid=True, label='J_cal')
# # series_J_cal = series_J_cal.resample('3T')
# # print('series.resample 3min', series_J_cal)
#
# series_J_mea = pd.Series(np.asarray(J_mea), index=index)
# # print('series_J_mea value', series_J_mea)
# series_J_mea.plot(kind='line', grid=True, label='J_mea')
# # series_J_mea = series_J_mea.resample('3T')
# # print('series.resample 3min', series_J_mea)

series_time = pd.Series(np.asarray(trackpdtime), index=index)
# print('series_time value', series_time)
# series_time = series_time.resample('3T')
# print('series.resample 3min', series_time)
plt.figure()
# plt.plot(df3['time'], df3['J_mea'])

# # Hamish resample method
# d = {'time': pd.Series(trackpdtime), 'J_cal': pd.Series(np.asarray(J_cal)),
#      'J_mea': pd.Series(np.asarray(J_mea))}
# # d= {'time':pd.Series(trackpdtime), 'latitude':pd.Series(np.asarray(lat)),'longitude':pd.Series(np.asarray(lon)),
# # 'altitude':pd.Series(np.asarray(alt)), 'OH':pd.Series(ohtrack),'HO2':pd.Series(ho2track),'HO2RO2':pd.Series(
# # ho2ro2track), 'H2SO4':pd.Series(h2so4track)}
# df3 = pd.DataFrame(d)
# # print('df3', df3)
# df3 = df3[['time', 'J_cal', 'J_mea']]
# # print('df3', df3)
# indexed1 = df3.set_index(df3['time'], append=True)
# # print('indexed1_value', indexed1)
# df2 = indexed1.resample('1T', on='time').mean()
# print('df2_shape', np.asarray(df2['J_mea']).shape)
# J_mea_v2 = np.asarray(df2['J_mea']).reshape(291)
# print('J_mea_v2_shape', J_mea_v2.shape)
# # print('df2', df2)
# df2['J_mea'].plot()
# df2['J_cal'].plot()
# # print('J_mea_value', df2['J_mea'])
# # print('J_cal_value', df2['J_cal'])
#
# plt.title('0802 J 1min smooth')
# plt.grid(linestyle='-.')
# plt.ylabel('Nucleation Rate J')
# plt.legend(["J_mes", "J_cal"])
# plt.yscale('log')
# plt.ylim(1e-10, 8e0)
# plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/1min_smooth.png')
# # plt.show()


# Now we need to look at NAV CNTS(count) data
cnts_NAV = np.loadtxt('/jet/home/ding0928/box_model/data/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',
                      delimiter=',',
                      skiprows=132, usecols=3)
print('cnts_NAV.shape', cnts_NAV.shape)
# print('cnts_NAV', cnts_NAV)
plt.figure()
plt.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(np.multiply(cnts_NAV, 1e-5)))
plt.xlabel('time')
plt.ylabel('count*1e-5')
plt.ylim(0, 6)
plt.title('cnts in 0731')
plt.legend(["731 CNTS count"])
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/cnts_1s.png')

# we link the cs to here double-Y figure, but need interp1d first
cs = np.asarray(cs_obse)
interp_cs = interp1d(np.asarray(time_cs), np.asarray(cs), kind='nearest', bounds_error=False,
                     fill_value=-9999)
cs_track = interp_cs(np.asarray(time_NAV))  # from 19397 to 19401

# double y to look at NH3 concentration
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.yscale('log')
# ax1.plot(np.asarray(time_NAV / 3600) - 7, np.asarray(J_mea), color='black', label='J_mea')
ax1.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(J_mea), color='black', label='J_mea')
# ax1.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(J_cal), color='pink', label='J_cal')
plt.yscale('log')
ax1.set_ylabel('Nucleation Rate J(molecule cm-3 s-1)')
ax1.set_title("J CS & gas concentration(0731) for obser")
ax1.set_ylim(5e-13, 5e0)
ax1.legend(loc=2)
ax1.grid()

ax2 = ax1.twinx()  # double-y
ax2.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(nh3 * 1e-6), '-.', label='NH3(1e-12)')
ax2.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(h2so4_track / 10), '-.', label='h2so4_measure(1e-7)')
ax2.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(temp / 100), '-.', label='temp(/100 K)')
ax2.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(np.multiply(cnts_NAV, 1e-5)), '-.', label='cnts(1e-5)')
ax2.plot(np.asarray(pd.Series(trackpdtime)), np.asarray(np.multiply(cs_track, 1e2)), '-.', label='cs_obser(1e2)')

ax2.set_ylabel('gas concentration(molecule cm-3),  CS   T')
ax2.legend(loc=1)
ax2.set_ylim(0, 5)
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/double_y.png')

# to validate the NH3 emission
file_nh3_em = Dataset(
    '/jet/home/ding0928/cylc-run/u-cj832/share/data/ancils/Regn1/resn_1/ukca/out/NH3_anthropogenic_2014_time_slice.nc',
    format='NETCDF4')
print('file_nh3_em', file_nh3_em)
print(file_nh3_em.variables['emissions_NH3'])  # kg m-2 s-1,surface
nh3_emi = (file_nh3_em.variables['emissions_NH3'][0, 0, :, :])  # time,model_level_number,grid_latitude,grid_longitude
nh3_emi_time = (file_nh3_em.variables['time'][:])
nh3_emi_lat = (file_nh3_em.variables['grid_latitude'][:])
nh3_emi_lon = (file_nh3_em.variables['grid_longitude'][:])
plt.figure()
plt.pcolormesh(nh3_emi_lon, nh3_emi_lat, nh3_emi, cmap='magma_r')
plt.colorbar(label='kg m-2 s-1', orientation="horizontal")
plt.title('nh3 emission(kg m-2 s-1) u-cj832 UM11.9', fontsize=14)  # fig 45
plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/fig_qual/emission_low.png')

# try to the correlation between obs vs model
# print('J_mea_v2.value', np.asarray(J_mea_v2))
# correlation = np.corrcoef(np.asarray(J_mea_v2), np.asarray(J_r))
# print('correlation', correlation)
# plt.figure()
# plt.scatter(np.asarray(J_mea_v2), np.asarray(J_r))
# plt.xlim(0, 0.01)
# plt.ylim(0, 0.01)
# plt.xlabel('J_mea')
# plt.ylabel('J_r')
# plt.title('J_mea vs J_r')
# d5 = {'J_mea_v2': np.asarray(J_mea_v2), 'J_r': np.asarray(J_r)}
# # d= {'time':pd.Series(trackpdtime), 'latitude':pd.Series(np.asarray(lat)),'longitude':pd.Series(np.asarray(lon)),
# # 'altitude':pd.Series(np.asarray(alt)), 'OH':pd.Series(ohtrack),'HO2':pd.Series(ho2track),'HO2RO2':pd.Series(
# # ho2ro2track), 'H2SO4':pd.Series(h2so4track)}
# df_5 = pd.DataFrame(d5)
# sns.lmplot(x='np.asarray(J_mea_v2)', y='np.asarray(J_r)', data=df_5)

###3min resample for other variables
plt.figure()
d2 = {'time': pd.Series(trackpdtime), 'cs_obse': pd.Series(np.asarray(np.multiply(cs_track, 1e2))),
      'cnts_NAV': pd.Series(np.asarray(np.multiply(cnts_NAV, 1e-5))),
      'nh3_obse': pd.Series(np.asarray(np.multiply(nh3, 1e-6))),
      'h2so4_obse': pd.Series(np.asarray(np.multiply(h2so4_track, 1e-1)))}
# d= {'time':pd.Series(trackpdtime), 'latitude':pd.Series(np.asarray(lat)),'longitude':pd.Series(np.asarray(lon)),
# 'altitude':pd.Series(np.asarray(alt)), 'OH':pd.Series(ohtrack),'HO2':pd.Series(ho2track),'HO2RO2':pd.Series(
# ho2ro2track), 'H2SO4':pd.Series(h2so4track)}
df_3 = pd.DataFrame(d2)
# print('df3', df3)
df_3_2 = df_3[['time', 'cs_obse', 'cnts_NAV', 'nh3_obse', 'h2so4_obse']]
# print('df3', df3)
indexed = df_3_2.set_index(df_3_2['time'], append=True)
# print('indexed1_value', indexed1)
df_2 = indexed.resample('1T', on='time').mean()
# print('df2', df2)
df_2['cs_obse'].plot()
df_2['cnts_NAV'].plot()
df_2['nh3_obse'].plot()
df_2['h2so4_obse'].plot()
# print('J_mea_value', df2['J_mea'])
# print('J_cal_value', df2['J_cal'])

plt.title('0731 gas 1min smooth')
plt.grid(linestyle='-.')
plt.ylabel('gas concentration')
plt.legend(["cs_obse(1e2)", "cnts_NAV(1e-5)", "nh3_obse(1e-12)", "h2so4_obse(1e-7)"])
# plt.yscale('log')
plt.ylim(0.1, 5)
plt.yscale('log')
plt.savefig('/jet/home/ding0928/box_model/data/J_nuc/1min_smooth_gas.png')
plt.show()

# neutralization for accumulation+Aitken mode
# 1.nitrate in accumulation mode
r_NO3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                    '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                    usecols=["model_NO3"])  # pptv
r_NO3 = np.asarray(r_NO3).reshape(318)
r_NO3 = (r_p * 62) / (8.314 * r_T) * r_NO3 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
r_NO3 = r_NO3 * 9.7e9  # from ug m-3 to molecule cm-3
# nitrate in Aitken mode
r_NO3_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["model_NO3_Ait"])  # pptv
r_NO3_Ait = np.asarray(r_NO3_Ait).reshape(318)
r_NO3_Ait = (r_p * 62) / (8.314 * r_T) * r_NO3_Ait * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
r_NO3_Ait = r_NO3_Ait * 9.7e9  # from ug m-3 to molecule cm-3

# 2.sulfate in accumulation mode
r_SO4_2 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["model_SO4_2"])
r_SO4_2 = np.asarray(r_SO4_2).reshape(318)
r_SO4_2 = (r_p * 96) / (8.314 * r_T) * r_SO4_2 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
r_SO4_2 = r_SO4_2 * 6.27e9  # from ug m-3 to molecule cm-3
# sulfate in Aitken mode
r_SO4_2_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["model_SO4_Ait"])  # ppt
r_SO4_2_Ait = np.asarray(r_SO4_2_Ait).reshape(318)
r_SO4_2_Ait = (r_p * 96) / (8.314 * r_T) * r_SO4_2_Ait * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
r_SO4_2_Ait = r_SO4_2_Ait * 6.27e9  # from ug m-3 to molecule cm-3

# 3.ammonium in accumulation mode
r_NH4 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                    '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                    usecols=["model_NH4"])  # ppt
r_NH4 = np.asarray(r_NH4).reshape(318)
r_NH4 = (r_p * 18) / (8.314 * r_T) * r_NH4 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
r_NH4 = r_NH4 * 3.34e10  # from ug m-3 to molecule cm-3
# ammonium in Aitken mode
r_NH4_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["model_NH4_Ait"])  # ppt
r_NH4_Ait = np.asarray(r_NH4_Ait).reshape(318)
r_NH4_Ait = (r_p * 18) / (8.314 * r_T) * r_NH4_Ait * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
r_NH4_Ait = r_NH4_Ait * 3.34e10  # from ug m-3 to molecule cm-3

r_neut_coef = np.divide(r_NO3 + np.multiply(r_SO4_2, 2), r_NH4)  # neutralization coef for regional model
r_neut_coef_Ait = np.divide(r_NO3_Ait + np.multiply(r_SO4_2_Ait, 2),
                            r_NH4_Ait)  # neutralization coef in Ait for regional model

# same neutralization in accumulation+Aitken mode but for global model
# 1.nitrate in accumulation mode
g_NO3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                    '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                    usecols=["glm_NO3"])  # pptv
g_NO3 = np.asarray(g_NO3).reshape(318)
g_NO3 = (g_p * 62) / (8.314 * g_T) * g_NO3 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
g_NO3 = g_NO3 * 9.7e9  # from ug m-3 to molecule cm-3
# nitrate in Aitken mode
g_NO3_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["glm_NO3_Ait"])  # pptv
g_NO3_Ait = np.asarray(g_NO3_Ait).reshape(318)
g_NO3_Ait = (g_p * 62) / (8.314 * g_T) * g_NO3_Ait * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
g_NO3_Ait = g_NO3_Ait * 9.7e9  # from ug m-3 to molecule cm-3

# 2.sulfate in accumulation mode
g_SO4_2 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["glm_SO4_2"])  # pptv
g_SO4_2 = np.asarray(g_SO4_2).reshape(318)
g_SO4_2 = (g_p * 96) / (8.314 * g_T) * g_SO4_2 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
g_SO4_2 = g_SO4_2 * 6.27e9  # from ug m-3 to molecule cm-3

# sulfate in Aitken mode
g_SO4_2_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                          '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                          usecols=["glm_SO4_Ait"])  # pptv
g_SO4_2_Ait = np.asarray(g_SO4_2_Ait).reshape(318)
g_SO4_2_Ait = (g_p * 96) / (8.314 * g_T) * g_SO4_2_Ait * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
g_SO4_2_Ait = g_SO4_2_Ait * 6.27e9  # from ug m-3 to molecule cm-3

# 3.ammonium in accumulation mode
g_NH4 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                    '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                    usecols=["glm_NH4"])  # pptv
g_NH4 = np.asarray(g_NH4).reshape(318)
g_NH4 = (g_p * 18) / (8.314 * g_T) * g_NH4 * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
g_NH4 = g_NH4 * 3.34e10  # from ug m-3 to molecule cm-3
# ammonium in Aitken mode
g_NH4_Ait = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                        '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                        usecols=["glm_NH4_Ait"])  # pptv
g_NH4_Ait = np.asarray(g_NH4_Ait).reshape(318)
g_NH4_Ait = (g_p * 18) / (8.314 * g_T) * g_NH4_Ait * 1e-6  # transfer ppt into ug m-3 given T and P, (S+P)P14
g_NH4_Ait = g_NH4_Ait * 3.34e10  # from ug m-3 to molecule cm-3

g_neut_coef = np.divide(g_NO3 + np.multiply(g_SO4_2, 2), g_NH4)
g_neut_coef_Ait = np.divide(g_NO3_Ait + np.multiply(g_SO4_2_Ait, 2), g_NH4_Ait)

#
fig = plt.figure(figsize=(10, 8))
plt.ylim(0.5, 2.5)
ax = fig.add_subplot(211)
ax.plot(np.asarray(model_time), np.asarray(r_neut_coef), label='neut_coef_r')
ax.plot(np.asarray(model_time), np.asarray(g_neut_coef), label='neut_coef_g')
tick_spacing = 80
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_ylabel('neut_coef_Accu')
ax.legend(loc=2)
ax.set_title('uptake coef=0.001')
plt.ylim(0.5, 2.5)

ax2 = fig.add_subplot(212)
ax2.plot(np.asarray(model_time), np.asarray(r_neut_coef_Ait), label='neut_coef_r_Ait')
ax2.plot(np.asarray(model_time), np.asarray(g_neut_coef_Ait), label='neut_coef_g_Ait')
tick_spacing = 80
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.set_ylabel('neut_coef_Ait')
ax2.legend(loc=2)
plt.ylim(0.5, 2)
plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/neutr_coff.png')

# neutralization for oberv_AMS
time_AMS = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                      '/FRAPPE-AMS_C130_20140731_R0.ict',
                      delimiter=',', skiprows=39, usecols=0)
data_SO4 = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                      '/FRAPPE-AMS_C130_20140731_R0.ict',
                      delimiter=',', skiprows=39, usecols=1)  # ugpsm3,(22859,)
print('time_AMS.shape', time_AMS.shape)  # ('time_AMS.shape', (23792,))
print('data_SO4.shape', data_SO4.shape)  # ('data_SO4.shape', (23792,))

data_SO4 = ma.masked_where(data_SO4 < 0, data_SO4)
time_AMS_SO4 = time_AMS[~data_SO4.mask]
data_SO4_2 = data_SO4[~data_SO4.mask]
print('data_SO4_2.shape', data_SO4_2.shape)  # ('data_SO4_2.shape', (1060,))
print('time_AMS_SO4.shape', time_AMS_SO4.shape)  #

data_NO3 = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                      '/FRAPPE-AMS_C130_20140731_R0.ict',
                      delimiter=',', skiprows=39, usecols=2)
print('data_NO3.shape', data_NO3.shape)  # (22859,)
data_NO3 = ma.masked_where(data_NO3 < 0, data_NO3)
time_AMS_NO3 = time_AMS[~data_NO3.mask]
data_NO3_2 = data_NO3[~data_NO3.mask]
print('data_NO3_2.shape', data_NO3_2.shape)  # ('data_NO3_2.shape', (853,))
print('time_AMS_NO3.shape', time_AMS_NO3.shape)  #

data_NH4 = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs'
                      '//FRAPPE-AMS_C130_20140731_R0.ict',
                      delimiter=',', skiprows=39, usecols=4)
print('data_NH4.shape', data_NH4.shape)  # (22859,)
data_NH4 = ma.masked_where(data_NH4 < 0, data_NH4)
time_AMS_NH4 = time_AMS[~data_NH4.mask]
data_NH4_2 = data_NH4[~data_NH4.mask]
print('data_NH4_2.shape', data_NH4_2.shape)  # ('data_NH4_2.shape', (706,))
print('time_AMS_NH4.shape', time_AMS_NH4.shape)  #

interp_SO4_2 = interp1d(np.asarray(time_AMS_SO4), np.asarray(data_SO4_2), kind='nearest', bounds_error=False,
                        fill_value=-9999)
SO4_2_track = interp_SO4_2(np.asarray(time_AMS_NH4))
print('SO4_2_track.shape', SO4_2_track.shape)  # =

interp_NO3 = interp1d(np.asarray(time_AMS_NO3), np.asarray(data_NO3_2), kind='nearest', bounds_error=False,
                      fill_value=-9999)
NO3_track = interp_NO3(np.asarray(time_AMS_NH4))
print('NO3_track.shape', NO3_track.shape)  # ('NO3_track.shape', (628,))

# unit conversion
NO3_track = NO3_track * 9.7e9  # from ug m-3 to molecule cm-3
SO4_2_track = SO4_2_track * 6.27e9  # from ug m-3 to molecule cm-3
data_NH4_2 = data_NH4_2 * 3.34e10  # from ug m-3 to molecule cm-3

# Now, 3 column has dimension of (935,), we calculate [NO3]+2[SO4]/[NH4]
AMS_neut_coef = np.divide(NO3_track + np.multiply(SO4_2_track, 2), data_NH4_2)
# draw the observ neutralization coeff
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.hist(np.divide(np.asarray(time_AMS_NH4), 3600), np.asarray(AMS_neut_coef), label='neut_coef_obs')
plt.ylabel('neut_coef_obs', fontsize=16)
plt.title('AMS obser')

plt.subplot(212)
plt.scatter(np.divide(np.asarray(time_AMS_NH4), 3600), np.asarray(AMS_neut_coef), label='neut_coef_obs')
plt.ylabel('neut_coef_obs_zoom_in', fontsize=16)
plt.ylim(0, 3)
plt.xlabel('731 UTC', fontsize=16)

plt.savefig(
    '/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/AMS_obs/neutr_coff_0731.png')
plt.show()

# Histogram for comparison
obs_nh3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["NH3"])  # pptv
obs_nh3 = np.asarray(obs_nh3).reshape(318)  # remember to reshape
obs_oh = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                     '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                     usecols=["OH"])  # pptv
obs_oh = np.asarray(obs_oh).reshape(318)
obs_so2 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                      '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                      usecols=["SO2"])  # pptv
obs_so2 = np.asarray(obs_so2).reshape(318)
obs_o3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                     '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                     usecols=["O3"])  # pptv
obs_o3 = np.asarray(obs_o3).reshape(318)
r_o3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                   '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                   usecols=["model_O3"])  # pptv
r_o3 = np.asarray(r_o3).reshape(318)
g_o3 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                   '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                   usecols=["glm_O3"])  # pptv
g_o3 = np.asarray(g_o3).reshape(318)
obs_c5h8 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                       '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                       usecols=["C5H8"])  # pptv
obs_c5h8 = np.asarray(obs_c5h8).reshape(318)
r_c5h8 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                     '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                     usecols=["model_C5H8"])  # pptv
r_c5h8 = np.asarray(r_c5h8).reshape(318)
g_c5h8 = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                     '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                     usecols=["glm_C5H8"])  # pptv
g_c5h8 = np.asarray(g_c5h8).reshape(318)

obs_co = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                     '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                     usecols=["CO"])  # pptv
obs_co = np.asarray(obs_co).reshape(318)
r_co = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                   '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                   usecols=["model_CO"])  # pptv
r_co = np.asarray(r_co).reshape(318)
g_co = pd.read_csv('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization'
                   '/ground_obs_npf/0731_comparison/interpolated_korus_dataset_20140731.csv', parse_dates=True,
                   usecols=["glm_CO"])  # pptv
g_co = np.asarray(g_co).reshape(318)

fig = plt.figure()
ax1 = fig.add_subplot(331)
ax1.hist(r_c5h8, bins=40)
ax1.hist(g_c5h8, bins=40)
ax1.hist(obs_c5h8, bins=40)

ax2 = fig.add_subplot(332)
ax2.hist(r_so2, bins=40)
ax2.hist(g_so2, bins=40)
ax2.hist(obs_so2, bins=40)

ax3 = fig.add_subplot(333)
ax3.hist(r_h2so4, bins=40)
ax3.hist(g_h2so4, bins=40)
ax3.hist(h2so4_track, bins=40)

ax4 = fig.add_subplot(334)
ax4.hist(r_oh, bins=40)
ax4.hist(g_oh, bins=40)
ax4.hist(obs_oh, bins=40)

ax5 = fig.add_subplot(335)
ax5.hist(r_nh3, bins=40)
ax5.hist(g_nh3, bins=40)
ax5.hist(obs_nh3, bins=40)

ax6 = fig.add_subplot(336)
ax6.hist(r_o3, bins=40)
ax6.hist(g_o3, bins=40)
ax6.hist(obs_o3, bins=40)

ax7 = fig.add_subplot(337)
ax7.hist(r_co, bins=40)
ax7.hist(g_co, bins=40)
ax7.hist(obs_co, bins=40)

ax.set_ylabel('Number of samples')
plt.show()