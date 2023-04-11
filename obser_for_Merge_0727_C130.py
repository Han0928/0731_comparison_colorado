import iris
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy.ma as ma


def merge_obs_method():
    # load >100nm UHSAS data
    file_100 = Dataset(
        '/jet/home/ding0928/Colorado/EOLdata/dataaw2NKX/RF02Z.20140727.171300_214500.PNI.nc')  # (16381, 31)
    print('file_100', file_100)
    temp = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140727_R4.ict', delimiter=',',
                      skiprows=133, usecols=29)  # inC
    pres = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140727_R4.ict', delimiter=',',
                      skiprows=133, usecols=62)  # in hpa  (19020, )
    print('temp.shape', temp.shape)  # 16320
    print('pres.shape', pres.shape)  # 16320
    uhsas_time = (file_100.variables['Time'][:])  # we need time to interpolate(19021, 1, 31)
    cut_off_size_100 = [0.097, 0.105, 0.113, 0.121, 0.129, 0.145, 0.162, 0.182, 0.202, 0.222, 0.242, 0.262,
                        0.282, 0.302, 0.401, 0.57, 0.656, 0.74, 0.833, 0.917, 1.008, 1.148, 1.319, 1.479,
                        1.636, 1.796, 1.955, 2.184, 2.413, 2.661, 2.991]  # 31bin in um
    cut_off_size_100 = np.multiply(cut_off_size_100, 1e-6)
    time_NAV = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140727_R4.ict',
                          delimiter=',',
                          skiprows=133, usecols=0)

    temp = np.concatenate((temp, [0]))  # to go from 16320 to 16321
    pres = np.concatenate((pres, [0]))
    time_NAV_2 = np.concatenate((time_NAV, [0]))

    data_100 = (file_100.variables['CS200_RPI'][:, 0, :])  # ('data_100.shape', (16321, 1, 31)
    print('data_100', data_100)
    print('data_100.shape', data_100.shape)
    term1 = np.multiply((temp + 273.15), 1013.25)
    term2 = np.multiply(pres, 293.15)
    data_100 = ma.masked_where(data_100 < 0, data_100)
    uhsas_time2 = uhsas_time[~data_100[:, 0].mask]

    gas_track_data2 = []
    interp_gas = interp1d(time_NAV_2, term2, kind='nearest', fill_value='extrapolate')
    gas_track = interp_gas(np.asarray(uhsas_time))
    gas_track_data2.append(gas_track)
    data_100_new3 = []
    data_100_new2 = data_100[~data_100.mask.any(axis=1)]
    print('data_100_new2.shape', data_100_new2.shape)

    # for i in range(31):
    #     # data_100_new = data_100[:, i]
    #     data_100_temporay = data_100_new[~data_100_new.mask]
    #     print('data_100_temporay.shape', data_100_temporay.shape)
    #     # uhsas_time2 = uhsas_time[~data_100_new.mask]
    #     # print('uhsas_time2.shape', uhsas_time2.shape)
    #     # term2 = term2[~data_100_new.mask]
    #     term4 = term2[~data_100_new.mask]  # gordon new 2&4
    #     # to remove the invalid value
    #     # term1 = term1[~data_100_new.mask]
    #     term5 = term1[~data_100_new.mask]  # gordon new 1&5
    #     term3 = np.multiply(data_100_temporay, term4)  # 293.15 term
    #     print('term3', term3) #3 and temporay is transient term.
    #     data_100_new3.append(np.divide(term3, term5))  # divide?multiply
    #     # print(data_100_new3.shape)

    time_NAV_2 = (time_NAV_2[~(data_100.mask.any(axis=1))])
    temp_2 = (temp[~(data_100.mask.any(axis=1))])
    # temp_2 = temp[~data_100_new.mask]
    pres_2 = (pres[~(data_100.mask.any(axis=1))])

    term1 = (term1[~(data_100.mask.any(axis=1))])
    term2 = (term2[~(data_100.mask.any(axis=1))])
    print('temp_2.shape', temp_2.shape)
    print('pres_2.shape', pres_2.shape)
    print('term1.shape', term1.shape)
    print('term2.shape', term2.shape)

    print('time_NAV_2.shape', time_NAV_2.shape)

    # data_100_new2 = np.array(data_100_new3)
    print('data_100_new2.shape', data_100_new2.shape)

    data_100_new3 = np.zeros(data_100_new2.shape)

    for i in range(31):
        data_100_new4 = np.multiply(data_100_new2[:, i], term2)
        data_100_new3[:, i] = np.divide(data_100_new4, term1)
        # data_100_new4 = data_100_new3[:, 31]

    data_100_new2 = data_100_new3
    # data_100 = term3/term1
    # data_100 = np.divide((np.dot(data_100, np.multiply(pres, 293.15))), (np.multiply((temp + 273.15), 1013.25)))
    # print('data_100', data_100)
    # # data_100 = np.divide((np.multiply(pres, 293.15) * data_100[:, np.newaxis], np.multiply(pres, 293.15))), (np.multiply((temp + 273.15), 1013.25)))
    #
    # print('data_100.shape', data_100.shape)

    # codes transfered from f90 to calculate CS
    se = 1.0  # Sticking efficiency for soluble modes;0.3 for insolu
    rgas = 287.05  # Dry air gas constant =(Jkg^-1 K^-1)
    rr = 8.314  # Universal gas constant(K mol^-1 K^-1)
    pi = 3.1415927
    zboltz = 1.38064852e-23  # (kg m2 s-2 K-1 molec-1)(J/K, J=Nm=kg m/s2 m)
    avc = 6.02e23  # Avogadros conirnt (mol-1)
    mm_da = avc * zboltz / rgas  # constant,molar mass of air (kg mol-1)
    t = temp_2 + 273.15  # C to K
    pmid = pres_2 * 100  # hpa to pa
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
        print(i)

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
        print('nc.shape', nc.shape)
        # nc[3] = acc_num_conc * cc[3]
        # nc[4] = cor_num_conc * cc[4]
        sumnc = sumnc + nc
        # np.array(sumnc[i])
        print('nc.shape', np.array(nc).shape)
        # print('sumnc', sumnc)

    # from now is smps less than 100nm for ait
    # temp_smps = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',
    #                        delimiter=',',
    #                        skiprows=133, usecols=29)  # C
    # pres_smps = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',
    #                        delimiter=',',
    #                        skiprows=133, usecols=62)  # in hpa
    # time_NAV = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',
    #                       delimiter=',',
    #                       skiprows=133, usecols=0)  # time
    # time_NAV = np.concatenate((time_NAV, [0]))
    # print('temp ', temp_smps)  # (16380,))
    # print('pres ', pres_smps)  # (16380,))
    # print('time_NAV shape', time_NAV.shape)  # (16380,))

    # temp_smps.shape = 16380, 1
    # pres_smps.shape = 16380, 1

    c = range(20, 34)  # column in smpsc=15*1
    data_smps = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/frappe-SMPS_C130_20140727_R0.ict', delimiter=',',
                           skiprows=66, usecols=c)  # from col 18-33
    data_smps = ma.masked_where(data_smps < 0, data_smps)
    smps_time = np.loadtxt('/jet/home/ding0928/Colorado/Colorado/frappe-SMPS_C130_20140727_R0.ict', delimiter=',',
                           skiprows=66, usecols=0)  # time resolution is different from uhsas,(202,1)
    # smps_time = smps_time[:, np.newaxis]

    # print('data_smps', data_smps)  # (258*14), 2d
    print('smps_time_shape', smps_time.shape)  # 258
    # data_smps = (data_smps * pres_smps * 293.15) / ((temp_smps + 273.15) * 1013.25)
    term_1 = np.multiply(pres_2, 293.15)
    term_2 = np.multiply(temp_2 + 273.15, 1013.25)
    gas_track_data = []

    for i in range(14):
        data_smps_new = data_smps[:, i]
        data_smps_new2 = data_smps_new[~data_smps_new.mask]  # remove the masked
        smps_time2 = smps_time[~data_smps_new.mask]

        print('data_smps_new.shape', data_smps_new.shape)
        print('data_smps_new2.shape', data_smps_new2.shape)
        print('smps_time2.shape', smps_time2.shape)

        interp_gas = interp1d(smps_time2, data_smps_new2, kind='nearest',
                              fill_value='extrapolate')  # smps_time(202,1)? data_smps(202,15) doesn't match?
        gas_track = interp_gas(np.asarray(time_NAV_2))  # method borrowed from long script
        gas_track_data.append(gas_track)

        # term_3 = np.multiply(data_smps[:, i], term_1)  # (202,15) can't multiply (202,1)?
        term_3 = np.multiply(gas_track_data, term_1)  # (202,15) can't multiply (202,1)?
        # np.multiply(data_smps, term_1)
        data_smp = np.divide(term_3, term_2)  # based on line 48 it should be divide?
        # interp_gas = interp1d(smps_time, data_smp[:], kind='linear',
        #                       fill_value='extrapolate')  # smps_time? data_smps doesn't match?
        # gas_track = interp_gas(np.asarray(uhsas_time))  # method borrowd from long script
        # gas_track_data.append(gas_track)

    # term_1 = np.multiply(pres_smps, 293.15)
    # term_2 = np.multiply(temp_smps + 273.15, 1013.25)
    # term_3 = data_smps * term_1[:, np.newaxis]  # (202,15) can't multiply (202,1)?
    # # np.multiply(data_smps, term_1)
    # data_smp = np.multiply(term_3, term_2)

    # nucl_data = data_smps[:, 0:2]  # mode3
    # ait_data = data_100[:, 3:14]

    cut_off_size_1 = [8.4, 10.1, 12.1, 14.5, 17.4, 20.9, 25.1, 30.3, 36.5, 44.1, 53.5, 65, 79, 96.4]  # 15bin
    cut_off_size_1 = np.multiply(cut_off_size_1, 1e-9)  # make sure the units are matched with uhsas(in um)

    # interp_gas = interp1d(smps_time, data_smps, kind='linear',
    #                       fill_value='extrapolate')  # to synchronize the time resolution
    # gas_track = interp_gas(np.asarray(uhsas_time))  # method borrowd from long script

    # all other constants are the same so no need to re-define
    t_smps = temp_2 + 273.15  # (16000,)
    pmid_smps = pres_2 * 100
    tsqrt_smps = np.sqrt(t_smps)

    # cc = 0.0  # condensation coeff for cpt onto pt(m3/s)
    vel_cp = np.multiply(term1, tsqrt_smps)  # list=cons*list; Calculate diffusion coefficient of condensable gas
    # dcoff_cp = np.divide(np.multiply((np.multiply(1.0e-7, (i ** 1.75 for i in t_smps)), term7), (
    #     np.multiply(np.divide(pmid, 101325.0),
    #                 term8))))  # list**1.75; Mann[56] when idcmfp==2,pmid=Centre level pressure (Pa)
    dcoff_cp = np.divide(np.multiply(np.multiply(1.0e-7, np.power(t_smps, 1.75)), term7),
                         (np.multiply(np.array(pmid_smps / 101325.0), term8)))

    mfp_cp = np.divide(np.multiply(3.0, dcoff_cp), vel_cp)  # mfp_cp is a list; Mann[55] whenidcmfp==2
    # defien before use
    kn_smps = []
    fkn_smps = []
    akn_smps = []
    cc_smps = []
    nc_smps = []

    for i in range(14):
        print(i)
        sumnc_smps = 0
        # mfp_cp_array = np.full(cut_off_size_100[i].shape, mfp_cp)
        # akn_array = np.full(cut_off_size_100[i].shape, 1)
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
        print(sumnc_smps)
        print('sumnc_smps.shape', sumnc_smps.shape)

    plt.subplot(2, 2, 1)
    plt.plot(np.divide(time_NAV_2, 3600), sumnc_smps)
    plt.title('sumnc_smps')

    plt.subplot(2, 2, 2)
    plt.plot(np.divide(time_NAV_2, 3600), sumnc)
    plt.ylim(0, 0.05)
    plt.title('sumnc')

    plt.subplot(2, 1, 2)
    plt.plot(np.divide(time_NAV_2, 3600), (sumnc_smps + sumnc))
    plt.ylim(0, 0.1)
    plt.title('0731_test_sum smps+sumnc')

    plt.savefig('/jet/home/ding0928/Colorado/Colorado/obser_fig_test.jpg')
    print('(sumnc_smps+sumnc).shape', (sumnc_smps + sumnc).shape)

    return (sumnc_smps + sumnc)


merge_obs_method()
