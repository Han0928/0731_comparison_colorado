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
    print(time_NAV)  # (17546,) starting from 65599
    temp = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140731_r0.ict', delimiter=',',
                      skiprows=73, usecols=18)  # inC  (17546,)
    # print('temp', temp)
    pres = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140731_r0.ict', delimiter=',',
                      skiprows=73, usecols=24)  # in hpa  (17546,)
    # print('pres', pres)
    # print('temp.shape', temp.shape)
    # print('pres.shape', pres.shape)

    # load >100nm UHSAS data for acc+coar
    file_100 = Dataset(
        '/jet/home/ding0928/Colorado/EOLdata/dataaw2NKX/RF05Z.20140731.195400_011100.PNI.nc')
    # print('file_100', file_100)

    uhsas_time = (file_100.variables['Time'][:])  # we need time to interpolate?
    print('uhsas_time', uhsas_time)  # (19021,) starts 71640
    print('uhsas_time.shape', uhsas_time.shape)  # (19021,)

    cut_off_size_100 = [0.097, 0.105, 0.113, 0.121, 0.129, 0.145, 0.162, 0.182, 0.202, 0.222, 0.242, 0.262,
                        0.282, 0.302, 0.401, 0.57, 0.656, 0.74, 0.833, 0.917, 1.008, 1.148, 1.319, 1.479,
                        1.636, 1.796, 1.955, 2.184, 2.413, 2.661, 2.991]  # 31bin in um
    cut_off_size_100 = np.multiply(cut_off_size_100, 1e-6)

    data_100 = (file_100.variables['CS200_RPI'][:, 0, :])  # ('data_100.shape', (19021, 31))
    # print('data_100', data_100)
    # print('data_100.shape', data_100.shape)  # ('data_100.shape', (19021, 31))

    term1 = np.multiply((temp + 273.15), 1013.25)
    term2 = np.multiply(pres, 293.15)
    # print('term2_shape', term2.shape)

    data_100 = ma.masked_where(data_100 < 0, data_100)
    uhsas_time2 = uhsas_time[~data_100[:, 0].mask]
    # print('uhsas_time2.shape', uhsas_time2.shape)

    interp_time = interp1d(time_NAV, time_NAV, kind='nearest', fill_value='extrapolate')
    time_track = interp_time(np.asarray(uhsas_time))
    # print('time_track', time_track)

    interp_term1 = interp1d(time_NAV, term1, kind='nearest', fill_value='extrapolate')
    term1_track = interp_term1(np.asarray(uhsas_time))
    # print('term1_track', term1_track)

    interp_term2 = interp1d(time_NAV, term2, kind='nearest', fill_value='extrapolate')
    term2_track = interp_term2(np.asarray(uhsas_time))
    # print('term2_track', term2_track)

    gas_track_data2 = []
    interp_gas = interp1d(time_NAV, term2, kind='nearest', fill_value='extrapolate')
    gas_track = interp_gas(np.asarray(uhsas_time))  # term2:17546 interpreted to term2:19021
    gas_track_data2.append(gas_track)

    interp_temp = interp1d(time_NAV, temp, kind='nearest', fill_value='extrapolate')
    temp_track = interp_temp(np.asarray(uhsas_time))
    # print('temp_track', temp_track)
    # print('temp_track.shape', temp_track.shape)

    interp_pres = interp1d(time_NAV, pres, kind='nearest', fill_value='extrapolate')
    pres_track = interp_pres(np.asarray(uhsas_time))
    # print('pres_track.shape', pres_track.shape)

    time_NAV_2 = (time_track[~(data_100.mask.any(axis=1))])
    temp_2 = (temp_track[~(data_100.mask.any(axis=1))])
    # temp_2 = temp[~data_100_new.mask]  # masked t
    pres_2 = (pres_track[~(data_100.mask.any(axis=1))])

    term1 = (term1_track[~(data_100.mask.any(axis=1))])
    term2 = (term2_track[~(data_100.mask.any(axis=1))])

    # print('temp_2.shape', temp_2.shape)  # 'temp_2.shape', (19000,)
    # print('pres_2.shape', pres_2.shape)  # 'pres_2.shape', (19000,)
    # print('term1.shape', term1.shape)  # ('term1.shape', (19000,))
    # print('term2.shape', term2.shape)  # ('term2.shape', (19000,))
    print('time_NAV_2', time_NAV_2)  # ('time_NAV_2.shape', (19000,))

    data_100_new2 = data_100[~data_100.mask.any(axis=1)]
    # print('data_100_new2.shape', data_100_new2.shape)  # (19000, 31) which means 21 points is masked out

    data_100_new3 = []
    data_100_new3 = np.zeros(data_100_new2.shape)

    for i in range(31):
        data_100_new4 = np.multiply(data_100_new2[:, i], term2)
        data_100_new3[:, i] = np.divide(data_100_new4, term1)
        # data_100_new4 = data_100_new3[:, 31]

    data_100_new2 = data_100_new3

    se = 1.0  # Sticking efficiency for soluble modes;0.3 for insolu
    rgas = 287.05  # Dry air gas constant =(Jkg^-1 K^-1)
    rr = 8.314  # Universal gas constant(K mol^-1 K^-1)
    pi = 3.1415927
    zboltz = 1.38064852e-23  # (kg m2 s-2 K-1 molec-1)(J/K, J=Nm=kg m/s2 m)
    avc = 6.02e23  # Avogadros conirnt (mol-1)
    mm_da = avc * zboltz / rgas  # constant,molar mass of air (kg mol-1)
    t = temp_2 + 273.15  # no longer constant since we loaded the obser_file
    pmid = pres_2 * 100  # no longer constant since we loaded the obser_file
    tsqrt = np.sqrt(t)  # Square-root of mid-level air temperature (K)
    mmcg = 0.098  # molar mass of H2SO4(kg per mole)
    dmol = 4.5e-10  # Molecular diameter of condensable (m)
    difvol = 51.96
    # sinkarr = 0.0
    # s_cond_s = 0.0  # Condensation sink

    # several constants defined
    term1 = np.sqrt(8.0 * rr / (pi * mmcg))  # constant; used in calcn of thermal velocity of condensable gas
    zz = mmcg / mm_da  # constant;
    term2 = 1.0 / (
            pi * np.sqrt(
        1.0 + zz) * dmol * dmol)  # constant;used in calcn of mfp of condensable gas(s & p, pg 457, eq 8.11)
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
    # define before use)
    # kn = []
    # fkn = []
    # akn = []
    # cc = []
    # nc = []
    sumnc = np.zeros(data_100_new2[:, 0].shape)  # avoid initializing each time,previously at 141

    for i in range(31):
        # print(i)

        # print('data_100_new2[i]', data_100_new2[i])
        kn = (np.divide(mfp_cp, cut_off_size_100[i]))  # for i<21 nc for accu; i>21 nc for coarse
        # print('kn.shape', kn.shape)

        fkn = np.divide((1.0 + kn), (
                1.0 + np.multiply(1.71, kn) + 1.33 * np.multiply(kn, kn)))  # calc.corr.factor Mann[52]
        akn = np.divide(1, (1.0 + 1.33 * np.multiply(kn, fkn) * (1.0 / se - 1.0)))  # se=0?? Mann[53]
        cc = term6 * dcoff_cp * np.multiply(np.multiply(cut_off_size_100[i], fkn),
                                            akn)  # Calc condensation coefficient Mann[51]
        # print('cc.shape', cc.shape)
        nc = np.multiply(data_100_new2[:, i], cc)
        # print('nc.shape', nc.shape)
        # nc[3] = acc_num_conc * cc[3]
        # nc[4] = cor_num_conc * cc[4]
        sumnc = sumnc + nc
        # np.array(sumnc[i])
        # print('nc.shape', np.array(nc).shape)
        # print('sumnc', sumnc)

    c = range(6, 26)  # in smpsc for diameter<100nm is the first 21 columns, 21-30 is discarded(>100nm)
    data_smps = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE'
                           '-SMPS_P3B_20140731_R0.ict', delimiter=',', skiprows=72, usecols=c)  # from col 18-33
    data_smps = ma.masked_where(data_smps < 0, data_smps)

    smps_time = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE'
                           '-SMPS_P3B_20140731_R0.ict', delimiter=',', skiprows=72, usecols=0)
    # print('smps_time_shape', smps_time.shape)  # 202,

    # data_smps = (data_smps * pres_smps * 293.15) / ((temp_smps + 273.15) * 1013.25)
    term_1 = np.multiply(pres_2, 293.15)
    term_2 = np.multiply(temp_2 + 273.15, 1013.25)
    gas_track_data = []

    for i in range(20):
        data_smps_new = data_smps[:, i]
        data_smps_new2 = data_smps_new[~data_smps_new.mask]  # remove the masked
        smps_time2 = smps_time[~data_smps_new.mask]

        # print('data_smps_new.shape', data_smps_new.shape)
        # print('data_smps_new2.shape', data_smps_new2.shape)
        # print('smps_time2.shape', smps_time2.shape)

        interp_gas = interp1d(smps_time2, data_smps_new2, kind='nearest',
                              fill_value='extrapolate')
        gas_track = interp_gas(np.asarray(time_NAV_2))
        gas_track_data.append(gas_track)

        term_3 = np.multiply(gas_track_data, term_1)  # (202,15) can't multiply (202,1)?
        # np.multiply(data_smps, term_1)
        data_smp = np.divide(term_3, term_2)  # based on line 48 it should be divide?

    cut_off_size_1 = [10.0, 11.2, 12.6, 14.1, 15.8, 17.8, 20, 22.4, 25.1, 28.2, 31.6, 35.5, 39.8, 44.7, 50.1, 56.2,
                      63.1, 70.8, 79.4,
                      89.1, 100]  # 21bin
    cut_off_size_1 = np.multiply(cut_off_size_1, 1e-9)  # make sure the units are matched with uhsas(in um)

    # interp_gas = interp1d(smps_time, data_smps, kind='linear',
    #                       fill_value='extrapolate')  # to synchronize the time resolution
    # gas_track = interp_gas(np.asarray(uhsas_time))  # method borrowd from long script

    # all other constants are the same so no need to re-define
    t_smps = temp_2 + 273.15
    pmid_smps = pres_2 * 100
    tsqrt_smps = np.sqrt(t_smps)

    # cc = 0.0  # condensation coeff for cpt onto pt(m3/s)
    vel_cp = np.multiply(term1, tsqrt_smps)  # list=cons*list; Calculate diffusion coefficient of condensable gas
    dcoff_cp = np.divide(np.multiply(np.multiply(1.0e-7, np.power(t_smps, 1.75)), term7),
                         (np.multiply(np.array(pmid_smps / 101325.0), term8)))

    mfp_cp = np.divide(np.multiply(3.0, dcoff_cp), vel_cp)  # mfp_cp is a list; Mann[55] whenidcmfp==2
    # defien before use
    kn_smps = []
    fkn_smps = []
    akn_smps = []
    cc_smps = []
    nc_smps = []

    for i in range(20):
        # print(i)
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
        nc_smps.append(np.multiply(data_smp[i], cc_smps[i]))
        # nc[3] = acc_num_conc * cc[3]
        # nc[4] = cor_num_conc * cc[4]
        sumnc_smps = sumnc_smps + nc_smps[i]  # sumnc_smps is for CS(Aitken)
        # print(sumnc_smps)
        # print('sumnc_smps.shape', sumnc_smps.shape)

    plt.subplot(2, 2, 1)
    plt.tight_layout()
    time_NAV_2 = time_NAV_2 - 21600
    date = '20140731'
    unix_epoch_of_31July2014_0001 = 1406764800
    if date == '20140731':
        trackpdtime = pd.to_datetime(np.asarray(time_NAV_2) + unix_epoch_of_31July2014_0001, unit='s')
    elif date == '20140801':
        unix_epoch_of_01August2014_0001 = unix_epoch_of_31July2014_0001 + 86400
        trackpdtime = pd.to_datetime(np.asarray(time_NAV_2) + unix_epoch_of_01August2014_0001, unit='s')
    elif date == '20140802':
        unix_epoch_of_02August2014_0001 = unix_epoch_of_31July2014_0001 + 86400 * 2
        trackpdtime = pd.to_datetime(np.asarray(time_NAV_2) + unix_epoch_of_02August2014_0001, unit='s')

    # print('trackpdtime',pd.Series(trackpdtime))
    d2 = {'time': pd.Series(trackpdtime), 'sumnc_smps': pd.Series(np.asarray(sumnc_smps))}
    df_3 = pd.DataFrame(d2)
    df_3_2 = df_3[['time', 'sumnc_smps']]
    indexed = df_3_2.set_index(df_3_2['time'], append=True)
    df_2 = indexed.resample('5T', on='time').mean()
    df_2['sumnc_smps'].plot()
    plt.title('0731 p3b_CS(<100nm)')
    plt.grid(linestyle='-.')
    plt.ylabel('s-1')
    # plt.yscale('log')
    # plt.savefig('0731_P3B_NH3(ppbv).png')

    plt.subplot(2, 2, 2)
    d3 = {'time': pd.Series(trackpdtime), 'sumnc': pd.Series(np.asarray(sumnc))}
    df_4 = pd.DataFrame(d3)
    df_4_2 = df_4[['time', 'sumnc']]
    indexed = df_4_2.set_index(df_4_2['time'], append=True)
    df_3 = indexed.resample('5T', on='time').mean()
    df_3['sumnc'].plot()
    plt.title('0731 p3b_CS(>100nm)')
    plt.grid(linestyle='-.')
    plt.ylabel('s-1')
    plt.yscale('log')
    plt.ylim(0.0001, 0.03)

    plt.subplot(2, 1, 2)
    d5 = {'time': pd.Series(trackpdtime), 'sumnc_all': pd.Series(np.asarray(sumnc_smps+sumnc))}
    df_6 = pd.DataFrame(d5)
    df_6_2 = df_6[['time', 'sumnc_all']]
    indexed = df_6_2.set_index(df_6_2['time'], append=True)
    df_5 = indexed.resample('5T', on='time').mean()
    df_5['sumnc_all'].plot()
    plt.title('0731 p3b_CS(total)')
    plt.grid(linestyle='-.')
    plt.ylabel('s-1')
    # plt.yscale('log')
    plt.ylim(0.0001, 0.03)
    plt.tight_layout()

    # plt.show()
    plt.savefig('/jet/home/ding0928/Colorado/Colorado/p-3b/obser_fig_test.jpg')
    # print('(sumnc_smps+sumnc).shape', (sumnc_smps + sumnc).shape)

    return sumnc_smps + sumnc


# merge_obs_method()
