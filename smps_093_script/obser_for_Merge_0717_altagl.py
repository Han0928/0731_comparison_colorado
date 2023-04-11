import iris
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy.ma as ma
from matplotlib.gridspec import GridSpec
from windrose import WindroseAxes
import matplotlib.cm as cm

fig = plt.figure(constrained_layout=True)
gs = GridSpec(4, 20, figure=fig)
plt.rcParams['font.size'] = 12


def read_p3b_smps():
    p3b_smps_time = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE-SMPS_P3B_20140717_R0.ict',
                               delimiter=',', skiprows=72, usecols=0)
    c = range(6, 35)  # SMPS data, dN/dlogDp,from 7-37
    p3b_data_smps = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE-SMPS_P3B_20140717_R0.ict',
                               delimiter=',', skiprows=72, usecols=c)

    p3b_data_smps = ma.masked_where(p3b_data_smps < 0, p3b_data_smps)

    p3b_smps_time = p3b_smps_time - 21600
    date = '20140717'
    unix_epoch_of_17July2014_0001 = 1405555200
    if date == '20140717':
        trackpdtime = pd.to_datetime(np.asarray(p3b_smps_time) + unix_epoch_of_17July2014_0001, unit='s')

    p3b_diam_list = np.array(
        [10.0, 11.2, 12.6, 14.1, 15.8, 17.8, 20, 22.4, 25.1, 28.2, 31.6, 35.5, 39.8, 44.7, 50.1, 56.2, 63.1, 70.8,
         79.4,
         89.1, 100, 112.2, 125.9, 141.3, 158.5, 177.8, 199.5, 223.9, 251.2, 281.8])
    plt.rcParams['font.size'] = 12

    ax3 = fig.add_subplot(gs[0, :])
    plt.pcolormesh(trackpdtime, p3b_diam_list, p3b_data_smps.T, norm=colors.LogNorm(vmin=10, vmax=110000),
                   cmap='RdBu_r')

    plt.colorbar(label='dN/dlogDp\n(#/cm-3)', extend='max')
    # plt.xlabel('Time(21th Jul, LT)')
    plt.ylabel('Diameter\n(nm)')
    plt.ylim(10, 300)
    plt.yscale('log')
    plt.text(0, 0, 'b).p-3b')
    xmin = '2014-07-17 10:30:00'
    xmax = '2014-07-17 15:30:00'
    plt.xlim(xmin, xmax)


# new updated for p-3b
def merge_obs_method():
    time_NAV = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140717_r0.ict',
                          delimiter=',',
                          skiprows=73, usecols=0)
    temp = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140717_r0.ict', delimiter=',',
                      skiprows=73, usecols=18)  # inC
    pres = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140717_r0.ict', delimiter=',',
                      skiprows=73, usecols=24)  # in hpa

    # load >100nm LAS data for acc+coar
    c = range(5, 31)
    data_las = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE-LAS_P3B_20140717_R0.ict',
                          delimiter=',', skiprows=69, usecols=c)
    data_las = ma.masked_where(data_las < 0, data_las)
    las_time = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE'
                          '-LAS_P3B_20140717_R0.ict', delimiter=',', skiprows=69, usecols=0)
    cut_off_size_100 = [112.2, 125.9, 141.3, 158.5, 177.8, 199.5, 223.9, 251.2, 281.8, 316.2, 354.8, 398.1, 446.7,
                        501.2, 562.3, 631, 707.9, 794.3, 891.3, 1000, 1258.9, 1584.9, 1995.3, 2511.9, 3162.3, 3981.1,
                        5011.9]
    cut_off_size_100 = np.multiply(cut_off_size_100, 1e-9)
    term1 = np.multiply(pres, 293.15)
    term2 = np.multiply(temp + 273.15, 1013.25)

    # gas_track_data2 = []
    # for i in range(26):
    #     data_las_new = data_las[:, i]
    #     # print('data_las_new.shape', data_las_new.shape)  # ('data_las_new.shape', (16617,))
    #     # print('data_las.shape', data_las.shape)  # ('data_las.shape', (16617, 26))
    #     # print('las_time.shape', las_time.shape)  # ('las_time.shape', (17291,))???
    #
    #     # data_las_new = ma.masked_where(data_las_new < 0, data_las_new)
    #     data_las_new2 = data_las_new[~data_las_new.mask]  # remove the masked
    #     las_time2 = las_time[~data_las_new.mask]
    #     # print('data_las_new.shape', data_las_new.shape)
    #     # print('data_las_new2.shape', data_las_new2.shape)  # ('data_las_new2.shape', (16611,))
    #     # print('las_time2.shape', las_time2.shape)  # ('las_time2.shape', (16611,))
    #
    #     interp_gas = interp1d(las_time2, data_las_new2, kind='nearest', fill_value='extrapolate')
    #     gas_track = interp_gas(np.asarray(time_NAV))
    #     gas_track_data2.append(gas_track)
    #
    #     term3 = np.multiply(gas_track_data2, term1)
    #     data_la = np.divide(term3, term2)

    t = temp + 273.15
    pmid = pres * 100
    tsqrt = np.sqrt(t)  # Square-root of mid-level air temperature (K)

    # define some constant
    se = 1.0  # Sticking efficiency for soluble modes;0.3 for insolu
    rgas = 287.05  # Dry air gas constant =(Jkg^-1 K^-1)
    rr = 8.314  # Universal gas constant(J mol^-1 K^-1)
    pi = 3.1415927
    zboltz = 1.38064852e-23  # (J/K)
    avc = 6.02e23  # Avogadros conirnt (mol-1)
    mm_da = avc * zboltz / rgas  # constant,molar mass of air (kg mol-1)
    mmcg = 0.098  # molar mass of H2SO4(kg per mole)
    dmol = 4.5e-10  # Molecular diameter of condensable (m)
    difvol = 51.96
    # sinkarr = 0.0
    # s_cond_s = 0.0  # Condensation sink

    term1 = np.sqrt(8.0 * rr / (pi * mmcg))  # constant; used in calcn of thermal velocity of condensable gas
    zz = mmcg / mm_da  # constant;
    term2 = 1.0 / (pi * np.sqrt(1.0 + zz) * dmol * dmol)
    # constant;used in calcn of mfp of condensable gas(s & p, pg 457, eq 8.11)
    term3 = (3.0 / (8.0 * avc * dmol * dmol))  # constant; Molecular diameter of condensable (m)
    term4 = np.sqrt((rgas * mm_da * mm_da / (2.0 * pi)) * ((mmcg + mm_da) / mmcg))
    term5 = term3 * term4  # used in calcnof diffusion coefficient of condensable gas
    term6 = 4.0e6 * pi  # used in calculation of condensation coefficient /cc-/m
    term7 = np.sqrt((1.0 / (mm_da * 1000.0)) + (1.0 / (mmcg * 1000.0)))
    dair = 19.7  # diffusion volume of air molecule(fuller et al, reid et al)
    term8 = (dair ** (1.0 / 3.0) + difvol ** (1.0 / 3.0)) ** 2  # used in new culation of diffusion coefficient

    # cc = 0.0  # condensation coeff for cpt onto pt(m3/s)
    vel_cp = np.multiply(term1, tsqrt)  # list=cons*list; Calculate diffusion coefficient of condensable gas
    dcoff_cp = np.divide(np.multiply(np.multiply(1.0e-7, np.power(t, 1.75)), term7),
                         (np.multiply(np.array(pmid / 101325.0), term8)))

    mfp_cp = np.divide(np.multiply(3.0, dcoff_cp), vel_cp)  # mfp_cp is a list; Mann[55] whenidcmfp==2

    # kn_las = []
    # fkn_las = []
    # akn_las = []
    # cc_las = []
    # nc_las = []
    # sumnc_las = []
    #
    # for i in range(26):
    #     sumnc_las = 0
    #     kn_las.append((np.divide(mfp_cp, cut_off_size_100[i])))  # for <100nm only aitken
    #     fkn_las.append(np.divide((1.0 + kn_las[i]), (
    #             1.0 + np.multiply(1.71, kn_las[i]) + 1.33 * np.multiply(kn_las[i],
    #                                                                     kn_las[i]))))  # calc.corr.factor Mann[52]
    #     akn_las.append(
    #         np.divide(1, (1.0 + 1.33 * np.multiply(kn_las[i], fkn_las[i]) * (1.0 / se - 1.0))))  # se=0?? Mann[53]
    #     cc_las.append(term6 * dcoff_cp * np.multiply(np.multiply(cut_off_size_100[i], fkn_las[i]),
    #                                                  akn_las[i]))  # Calc condensation coefficient Mann[51]
    #     nc_las.append(np.multiply(data_la[i], cc_las[i]))  # here is data_la instead of data_las
    #     # print('nc_las).shape', np.array(nc_las).shape)
    #     sumnc_las = sumnc_las + nc_las[i]
    #     # print(sumnc_las)

    c = range(6, 26)  # for diameter<100nm, 21-30 is discarded(>100nm)due to overlap with LAS data
    data_smps = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE-SMPS_P3B_20140717_R0.ict',
                           delimiter=',', skiprows=72, usecols=c)  # from col 18-33
    data_smps = ma.masked_where(data_smps < 0, data_smps)

    smps_time = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE-SMPS_P3B_20140717_R0.ict',
                           delimiter=',', skiprows=72, usecols=0)

    # data_smps = (data_smps * pres_smps * 293.15) / ((temp_smps + 273.15) * 1013.25)
    term_1 = np.multiply(pres, 293.15)
    term_2 = np.multiply(temp + 273.15, 1013.25)

    gas_track_data = []
    for i in range(20):
        data_smps_new = data_smps[:, i]
        data_smps_new2 = data_smps_new[~data_smps_new.mask]  # remove the masked
        smps_time2 = smps_time[~data_smps_new.mask]

        interp_gas = interp1d(smps_time2, data_smps_new2, kind='nearest',
                              fill_value='extrapolate')
        gas_track = interp_gas(np.asarray(time_NAV))
        gas_track_data.append(gas_track)

        term_3 = np.multiply(gas_track_data, term_1)  # (202,15) can't multiply (202,1)?
        data_smp = np.divide(term_3, term_2)

    cut_off_size_1 = [10.0, 11.2, 12.6, 14.1, 15.8, 17.8, 20, 22.4, 25.1, 28.2, 31.6, 35.5, 39.8, 44.7, 50.1, 56.2,
                      63.1, 70.8, 79.4,
                      89.1, 100]  # 21bin
    cut_off_size_1 = np.multiply(cut_off_size_1, 1e-9)  # make sure the units are matched with uhsas(in um)

    # all other constants are the same so no need to re-define
    t_smps = temp + 273.15
    pmid_smps = pres * 100
    tsqrt_smps = np.sqrt(t_smps)

    # cc = 0.0  # condensation coeff for cpt onto pt(m3/s)
    vel_cp = np.multiply(term1, tsqrt_smps)  # list=cons*list; Calculate diffusion coefficient of condensable gas
    dcoff_cp = np.divide(np.multiply(np.multiply(1.0e-7, np.power(t_smps, 1.75)), term7),
                         (np.multiply(np.array(pmid_smps / 101325.0), term8)))

    mfp_cp = np.divide(np.multiply(3.0, dcoff_cp), vel_cp)  # mfp_cp is a list; Mann[55] whenidcmfp==2

    kn_smps = []
    fkn_smps = []
    akn_smps = []
    cc_smps = []
    nc_smps = []
    sumnc_smps = []

    for i in range(20):
        sumnc_smps = 0
        kn_smps.append(
            (np.divide(mfp_cp, cut_off_size_1[i])))  # for <100nm only aitken
        fkn_smps.append(np.divide((1.0 + kn_smps[i]), (
                1.0 + np.multiply(1.71, kn_smps[i]) + 1.33 * np.multiply(kn_smps[i],
                                                                         kn_smps[i]))))  # calc.corr.factor Mann[52]
        akn_smps.append(
            np.divide(1, (1.0 + 1.33 * np.multiply(kn_smps[i], fkn_smps[i]) * (1.0 / se - 1.0))))  # se=0?? Mann[53]
        cc_smps.append(term6 * dcoff_cp * np.multiply(np.multiply(cut_off_size_1[i], fkn_smps[i]),
                                                      akn_smps[i]))  # Calc condensation coefficient Mann[51]
        nc_smps.append(np.multiply(data_smp[i], cc_smps[i]))  # notice here is data_smp instead of data_smps
        sumnc_smps = sumnc_smps + nc_smps[i]  # sumnc_smps is for CS(Aitken)

    # since I wanna make twin plot
    alt_p3b = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140717_r0.ict',
                         delimiter=',',
                         skiprows=73, usecols=4)
    # plt.tight_layout()
    time_NAV = time_NAV - 21600

    date = '20140717'
    unix_epoch_of_17July2014_0001 = 1405555200
    if date == '20140717':
        trackpdtime = pd.to_datetime(np.asarray(time_NAV) + unix_epoch_of_17July2014_0001, unit='s')

    ax1 = fig.add_subplot(gs[1, :])
    d5 = {'time': pd.Series(trackpdtime), 'CS': pd.Series(np.asarray(sumnc_smps))}
    df_6 = pd.DataFrame(d5)
    df_6_2 = df_6[['time', 'CS']]
    indexed = df_6_2.set_index(df_6_2['time'], append=True)
    df_5 = indexed.resample('1T', on='time').mean()
    df_5['CS'].plot()
    plt.grid(True, axis='both')
    plt.ylabel('Condensation \n Sink s$^{-1}$')
    plt.yscale('log')
    plt.ylim(1e-4, 1e-1)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    d2 = {'time': pd.Series(trackpdtime), 'p3b_alt': pd.Series(np.asarray(alt_p3b))}
    df_3 = pd.DataFrame(d2)
    df_3_2 = df_3[['time', 'p3b_alt']]
    indexed = df_3_2.set_index(df_3_2['time'], append=True)
    df_2 = indexed.resample('1T', on='time').mean()
    df_2['p3b_alt'].plot(color='green')
    ax2.set_ylabel('Aircraft \n Altitude(m)')
    ax2.legend(loc='upper right')
    plt.text(0, 0, 'a).p-3b')
    xmin = '2014-07-17 10:30:00'
    xmax = '2014-07-17 15:30:00'
    plt.xlim(xmin, xmax)
    print('mean:sumnc_las+sumnc_smps', np.nanmean(sumnc_smps))
    return sumnc_smps


def aerosol_3_10nm():
    p3b_cNgtnm_smps_time = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE'
                                      '-CNC_P3B_20140717_R0.ict', delimiter=',', skiprows=47, usecols=0)
    p3b_cNgtnm_smps_time = p3b_cNgtnm_smps_time - 21600

    p3b_data_CNgt3 = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE'
                                '-CNC_P3B_20140717_R0.ict', delimiter=',', skiprows=47, usecols=1)
    p3b_data_CNgt3 = ma.masked_where(p3b_data_CNgt3 < 0, p3b_data_CNgt3)

    p3b_data_CNgt10 = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE'
                                 '-CNC_P3B_20140717_R0.ict', delimiter=',', skiprows=47, usecols=2)
    p3b_data_CNgt10 = ma.masked_where(p3b_data_CNgt10 < 0, p3b_data_CNgt10)

    date = '20140717'
    unix_epoch_of_17July2014_0001 = 1405555200
    if date == '20140717':
        trackpdtime = pd.to_datetime(np.asarray(p3b_cNgtnm_smps_time) + unix_epoch_of_17July2014_0001, unit='s')

    ax3 = fig.add_subplot(gs[2, :])
    d2 = {'time': pd.Series(trackpdtime), 'N_3nm': pd.Series(np.asarray(p3b_data_CNgt3)),
          'N_10nm': pd.Series(np.asarray(p3b_data_CNgt10)),
          'N_3~10nm': pd.Series(np.asarray(p3b_data_CNgt3) - np.asarray(p3b_data_CNgt10))}
    df_3 = pd.DataFrame(d2)
    df_3_2 = df_3[['time', 'N_3nm', 'N_10nm', 'N_3~10nm']]
    indexed = df_3_2.set_index(df_3_2['time'], append=True)
    df_2 = indexed.resample('1T', on='time').mean()
    df_2['N_3nm'].plot()
    df_2['N_10nm'].plot()
    df_2['N_3~10nm'].plot()
    plt.grid(linestyle='-')
    plt.ylabel('N\n (# cm$^{-3}$)')
    ax3.legend(loc='lower left')
    plt.legend(["N(>3nm)", "N(>10nm)", "Nuc$_{3-10nm}$"])
    plt.yscale('log')
    plt.ylim(1e1, 1e5)
    xmin = '2014-07-17 10:30:00'
    xmax = '2014-07-17 15:30:00'
    plt.xlim(xmin, xmax)
    plt.xticks(color='w')
    print('mean:Nuc$_{3-10nm}$', np.nanmean(p3b_data_CNgt3 - p3b_data_CNgt10))


def read_bao_tower_smps():  # notice this is MDT!!
    bao_tower_smps_time = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/frappe-ground'
                                     '-smps_GROUND-BAO-TOWER_20140717_R1.ict', delimiter=',', skiprows=88,
                                     usecols=0)

    c = range(30, 55)
    bao_tower_smps = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/frappe-ground'
                                '-smps_GROUND-BAO-TOWER_20140717_R1.ict', delimiter=',', skiprows=88, usecols=c)

    bao_tower_smps = ma.masked_where(bao_tower_smps < 0, bao_tower_smps)

    baotower_diam_list = np.array([5.0, 5.6, 6.3, 7.1, 8.0, 9.0, 10.1, 11.3, 12.8, 14.3, 16.1, 18.1, 20.4, 23.0, 25.9,
                                   29.2, 33.0, 37.2, 42.1, 47.6, 53.8, 61.1, 69.3, 78.8, 89.8, 102.4])

    date = '20140717'
    unix_epoch_of_17July2014_0001 = 1405555200
    if date == '20140717':
        trackpdtime = pd.to_datetime(np.asarray(bao_tower_smps_time) + unix_epoch_of_17July2014_0001, unit='s')

    plt.rcParams['font.size'] = 12
    ax4 = fig.add_subplot(gs[3, :])
    plt.pcolormesh(trackpdtime, baotower_diam_list, bao_tower_smps.T, norm=colors.LogNorm(vmin=10, vmax=110000),
                   cmap='RdBu_r')
    plt.colorbar(label='dN/dlog$_{Dp}$\n(# cm$^{-3}$)', extend='max')
    plt.ylabel('Diameter\n (nm)')
    plt.ylim(4.5, 120)
    plt.yscale('log')
    plt.text(0, 0, 'c).BAO Tower')
    xmin = '2014-07-17 10:30:00'
    xmax = '2014-07-17 15:30:00'
    plt.xlim(xmin, xmax)


def new_axes():
    fig1 = plt.figure(figsize=(4, 4), dpi=180, facecolor='w', edgecolor='w')
    rect = [0, 0, 1, 1]
    # ax = WindroseAxes(fig, rect)
    ax = WindroseAxes(fig1, rect, facecolor='w')
    fig1.add_axes(ax)
    return ax


# ...and adjust the legend box
def set_legend(ax):
    l = ax.legend(shadow=False, bbox_to_anchor=[1, 0])
    plt.setp(l.get_texts(), fontsize=12)


def bao_tower_windrose():
    bao_tower_ws10 = np.loadtxt(
        '/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/FRAPPE-TowerMet_GROUND'
        '-BAO-TOWER_20140717_RB.ict', delimiter=',', skiprows=677,
        usecols=3)  # =
    bao_tower_wd10 = np.loadtxt(
        '/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/FRAPPE-TowerMet_GROUND'
        '-BAO-TOWER_20140717_RB.ict', delimiter=',', skiprows=677,
        usecols=4)  # =
    bao_tower_time10 = np.loadtxt(
        '/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/FRAPPE-TowerMet_GROUND'
        '-BAO-TOWER_20140717_RB.ict', delimiter=',', skiprows=677, usecols=0)
    bao_tower_wd10 = ma.masked_where(bao_tower_wd10 < 0, bao_tower_wd10)
    sl = [0, 0.8, 1.6, 2.6, 3.8, 4.6, 5.5, 6.5]
    ax = new_axes()
    ax.contourf(bao_tower_wd10, bao_tower_ws10, bins=sl, normed=True, cmap=cm.cool)
    # ax.set_title(mysheet.name, fontsize=15, loc='right')
    set_legend(ax)
    print('mean:bao_tower_wd10', np.nanmean(bao_tower_wd10[0:300]))
    print('bao_tower_wd10[300]', bao_tower_wd10[300])  # 246.58
    plt.show()


read_p3b_smps()
merge_obs_method()
aerosol_3_10nm()
read_bao_tower_smps()
bao_tower_windrose()
