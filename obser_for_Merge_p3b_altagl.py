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
    pres = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140731_r0.ict', delimiter=',',
                      skiprows=73, usecols=24)  # in hpa  (17546,)

    # load >100nm LAS data for acc+coar
    c = range(5, 31)  # 112nm=6th colomn; 5012nm=32th colomn
    data_las = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE'
                          '-LAS_P3B_20140731_R0.ict', delimiter=',', skiprows=69, usecols=c)
    data_las = ma.masked_where(data_las < 0, data_las)
    print('data_las.shape', data_las.shape)
    las_time = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE'
                          '-LAS_P3B_20140731_R0.ict', delimiter=',', skiprows=69, usecols=0)
    cut_off_size_100 = [112.2, 125.9, 141.3, 158.5, 177.8, 199.5, 223.9, 251.2, 281.8, 316.2, 354.8, 398.1, 446.7,
                        501.2, 562.3, 631, 707.9, 794.3, 891.3, 1000, 1258.9, 1584.9, 1995.3, 2511.9, 3162.3, 3981.1,
                        5011.9]
    cut_off_size_100 = np.multiply(cut_off_size_100, 1e-9)
    term1 = np.multiply(pres, 293.15)
    term2 = np.multiply(temp + 273.15, 1013.25)

    # gas_track_data3 = []
    gas_track_data2 = []
    # gas_las_data3 = []  #
    for i in range(26):
        data_las_new = data_las[:, i]
        # data_las_new = ma.masked_where(data_las_new < 0, data_las_new)
        data_las_new2 = data_las_new[~data_las_new.mask]  # remove the masked
        las_time2 = las_time[~data_las_new.mask]
        print('data_las_new.shape', data_las_new.shape)
        print('data_las_new2.shape', data_las_new2.shape)
        print('las_time2.shape', las_time2.shape)

        interp_gas = interp1d(las_time2, data_las_new2, kind='nearest', fill_value='extrapolate')
        gas_track = interp_gas(np.asarray(time_NAV))  # term2:17546 interpreted to term2:19021
        gas_track_data2.append(gas_track)

        term3 = np.multiply(gas_track_data2, term1)
        data_la = np.divide(term3, term2)
        # np.multiply(data_smps, term_1)
        # gas_las_data3.append(np.divide(term3, term2))

    # gas_las_data3 = np.array(gas_las_data3)
    # print('gas_las_data3.shape', gas_las_data3.shape)  # ('gas_las_data3.shape', (26,))

    t = temp + 273.15
    pmid = pres * 100
    tsqrt = np.sqrt(t)  # Square-root of mid-level air temperature (K)

    # define some constant
    se = 1.0  # Sticking efficiency for soluble modes;0.3 for insolu
    rgas = 287.05  # Dry air gas constant =(Jkg^-1 K^-1)
    rr = 8.314  # Universal gas constant(K mol^-1 K^-1)
    pi = 3.1415927
    zboltz = 1.38064852e-23  # (kg m2 s-2 K-1 molec-1)(J/K, J=Nm=kg m/s2 m)
    avc = 6.02e23  # Avogadros conirnt (mol-1)
    mm_da = avc * zboltz / rgas  # constant,molar mass of air (kg mol-1)
    mmcg = 0.098  # molar mass of H2SO4(kg per mole)
    dmol = 4.5e-10  # Molecular diameter of condensable (m)
    difvol = 51.96
    # sinkarr = 0.0
    # s_cond_s = 0.0  # Condensation sink

    # several constants defined
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

    kn_las = []
    fkn_las = []
    akn_las = []
    cc_las = []
    nc_las = []

    for i in range(26):
        sumnc_las = 0
        kn_las.append((np.divide(mfp_cp, cut_off_size_100[i])))  # for <100nm only aitken
        fkn_las.append(np.divide((1.0 + kn_las[i]), (
                1.0 + np.multiply(1.71, kn_las[i]) + 1.33 * np.multiply(kn_las[i],
                                                                        kn_las[i]))))  # calc.corr.factor Mann[52]
        akn_las.append(
            np.divide(1, (1.0 + 1.33 * np.multiply(kn_las[i], fkn_las[i]) * (1.0 / se - 1.0))))  # se=0?? Mann[53]
        cc_las.append(term6 * dcoff_cp * np.multiply(np.multiply(cut_off_size_100[i], fkn_las[i]),
                                                     akn_las[i]))  # Calc condensation coefficient Mann[51]
        nc_las.append(np.multiply(data_la[i], cc_las[i]))
        # nc[3] = acc_num_conc * cc[3]
        # nc[4] = cor_num_conc * cc[4]
        print('nc_las).shape', np.array(nc_las).shape)
        sumnc_las = sumnc_las + nc_las[i]
        print(sumnc_las)

    print('sumnc_las.shape', sumnc_las.shape)  # ('sumnc_las.shape', (26, 17546))
    print('nc_las.shape', np.array(nc_las).shape)  # ('nc_las.shape', (26,))
    print('kn_las.shape', np.array(kn_las).shape)  # ('kn_las.shape', (26, 17546))
    print('cc_las.shape', np.array(cc_las).shape)  # ('cc_las.shape', (26, 17546))
    print('fkn_las.shape', np.array(fkn_las).shape)  # ('fkn_las.shape', (26, 17546))

    c = range(6, 26)  # in smpsc for diameter<100nm is the first 21 columns, 21-30 is discarded(>100nm)
    data_smps = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE'
                           '-SMPS_P3B_20140731_R0.ict', delimiter=',', skiprows=72, usecols=c)  # from col 18-33
    data_smps = ma.masked_where(data_smps < 0, data_smps)

    smps_time = np.loadtxt('/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE'
                           '-SMPS_P3B_20140731_R0.ict', delimiter=',', skiprows=72, usecols=0)
    # print('smps_time_shape', smps_time.shape)  # 202,

    # data_smps = (data_smps * pres_smps * 293.15) / ((temp_smps + 273.15) * 1013.25)
    term_1 = np.multiply(pres, 293.15)
    term_2 = np.multiply(temp + 273.15, 1013.25)

    gas_track_data = []
    gas_smps_data3 = []
    for i in range(20):
        data_smps_new = data_smps[:, i]
        data_smps_new2 = data_smps_new[~data_smps_new.mask]  # remove the masked
        smps_time2 = smps_time[~data_smps_new.mask]

        # print('data_smps_new.shape', data_smps_new.shape)
        # print('data_smps_new2.shape', data_smps_new2.shape)
        # print('smps_time2.shape', smps_time2.shape)

        interp_gas = interp1d(smps_time2, data_smps_new2, kind='nearest',
                              fill_value='extrapolate')
        gas_track = interp_gas(np.asarray(time_NAV))
        gas_track_data.append(gas_track)

        term_3 = np.multiply(gas_track_data, term_1)  # (202,15) can't multiply (202,1)?
        data_smp = np.divide(term_3, term_2)

        # np.multiply(data_smps, term_1)
        # data_smp = np.divide(term_3, term_2)  # based on line 48 it should be divide?
        # gas_smps_data3.append(np.divide(term_3, term_2))

    # gas_smps_data3 = np.array(gas_smps_data3)
    # print('gas_smps_data3.shape', gas_smps_data3.shape)  # ('gas_smps_data3.shape', (20,))

    cut_off_size_1 = [10.0, 11.2, 12.6, 14.1, 15.8, 17.8, 20, 22.4, 25.1, 28.2, 31.6, 35.5, 39.8, 44.7, 50.1, 56.2,
                      63.1, 70.8, 79.4,
                      89.1, 100]  # 21bin
    cut_off_size_1 = np.multiply(cut_off_size_1, 1e-9)  # make sure the units are matched with uhsas(in um)

    # interp_gas = interp1d(smps_time, data_smps, kind='linear',
    #                       fill_value='extrapolate')  # to synchronize the time resolution
    # gas_track = interp_gas(np.asarray(uhsas_time))  # method borrowd from long script

    # all other constants are the same so no need to re-define
    t_smps = temp + 273.15
    pmid_smps = pres * 100
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
    sumnc_smps = []

    for i in range(20):
        # for j in range(17546):
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

    print('sumnc_smps.shape', sumnc_smps.shape)  # ('sumnc_smps.shape', (17546,))
    print('nc_smps.shape', np.array(nc_smps).shape)  # ('nc_smps.shape', (2311, 1, 17546))
    print('cc_smps.shape', np.array(cc_smps).shape)  # ('cc_smps.shape', (2311, 17546))
    print('fkn_smps.shape', np.array(fkn_smps).shape)  # ('fkn_smps.shape', (2310, 17546))

    plt.subplot(2, 2, 1)
    plt.tight_layout()
    time_NAV = time_NAV - 21600
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
    d3 = {'time': pd.Series(trackpdtime), 'sumnc_las': pd.Series(np.asarray(sumnc_las))}
    df_4 = pd.DataFrame(d3)
    df_4_2 = df_4[['time', 'sumnc_las']]
    indexed = df_4_2.set_index(df_4_2['time'], append=True)
    df_3 = indexed.resample('5T', on='time').mean()
    df_3['sumnc_las'].plot()
    plt.title('0731 p3b_CS(>100nm)')
    plt.grid(linestyle='-.')
    plt.ylabel('s-1')
    plt.yscale('log')
    plt.ylim(0.0001, 0.03)

    plt.subplot(2, 1, 2)
    d5 = {'time': pd.Series(trackpdtime), 'sumnc_all': pd.Series(np.asarray(sumnc_smps + sumnc_las))}
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

    return sumnc_smps + sumnc_las

# merge_obs_method()
