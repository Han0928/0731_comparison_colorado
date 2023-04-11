#from typing import Dict

import iris,sys,glob
import iris.quickplot as qplt
import iris.plot as iplt
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from iris.util import unify_time_units
from iris.experimental.equalise_cubes import equalise_attributes
import iris.coord_systems as cs
import matplotlib.patches as patches
import iris.coord_systems as cs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.colors as cols
import matplotlib.cm as cmx
#import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from pandas import Series
from scipy.interpolate import interp1d,RegularGridInterpolator
from gridded_interpolation import _RegularGridInterpolator # This is from cis.
import matplotlib as mpl
import datetime
import time
#from cis.data_io.ungridded_data import UngriddedData
#from cis.data_io.hyperpoint import HyperPoint
import pandas as pd

#flags for whether to make and write intermediate files or use them
new_diam_calc=0
make_dataset=1
justso2=0 # only works for plotting, can't use with make_dataset=1
##plotting routines
def add_times_to_plot(df,date):
    for itime in range(0,len(df.altitude)):
        hours = df.index.hour
        minutes= df.index.minute
        seconds = df.index.second
        if minutes[itime]==0 and seconds[itime]==0:
            plt.plot(df.longitude[itime],df.latitude[itime],marker='|',markersize=3,color='black')
            if hours[itime] < 10:
                timestring = '0'+str(hours[itime])+':00'
            else:
                timestring = str(hours[itime])+':00'
            if (date=='20160511' and hours[itime]%2==0) or date=='20160510' or date=='20160512':
                plt.text(df.longitude[itime],df.latitude[itime],timestring, fontsize=12)
        if minutes[itime]%4==0 and seconds[itime]==0:
            plt.plot(df.longitude[itime],df.latitude[itime],marker='|',markersize=3,color='black')
def add_flight_track_to_latlon(ax,norm,cmap,variable,dataframe,date):
    points = np.array([dataframe.longitude, dataframe.latitude]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap,norm=norm)
    lc.set_array(np.asarray(variable))
    lc.set_linewidth(2)
    ax.add_collection(lc)
    add_times_to_plot(dataframe,date)

def horizontalplot(ax, cube,minv,maxv,norm):
    if norm !=None:
        pl =  iplt.pcolormesh(cube,vmin=minv,vmax=maxv,norm=norm)
    else:
        pl =  iplt.pcolormesh(cube,vmin=minv,vmax=maxv)
    pl.cmap.set_under('k')
    plt.gca().stock_img()
    plt.gca().coastlines(resolution='50m')
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='-',draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter(LONGITUDE_FORMATTER)
    gl.yformatter(LATITUDE_FORMATTER)

#plot the comparison figure but black observation is missing
def plot_comparison(df,obs_col,lam_col,glm_col,lam2d,glm2d,ylabel,var_min,var_max,map_min,map_max,is_log,norm_array,saving_name,date):
    plt.figure(figsize=(8,4.8))
    plt.subplot(212)
    obs_col.plot(logy=is_log,ylim=(var_min,var_max),label='Obs.', color='black',linewidth=1.5)
    lam_col.plot(logy=is_log,ylim=(var_min,var_max),label='Regn.' )
    ax2 = glm_col.plot(logy=is_log,ylim=(var_min,var_max),label='Global')
    ax2.set_ylabel(ylabel)
    ax3 = df['altitude'].plot(secondary_y=True,label='Altitude')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_ylim(0,10000)

    h2,l2 = ax2.get_legend_handles_labels()
    h3,l3 = ax3.get_legend_handles_labels()
    plt.legend(h2+h3,l2+l3,bbox_to_anchor=(1.0, 1.1),ncol=2)

    ax = plt.subplot(221, projection=ccrs.PlateCarree())
    if is_log:
        horizontalplot(ax, lam2d, map_min, map_max, LogNorm())
    else:
        horizontalplot(ax, lam2d, map_min, map_max,None)
    plt.colorbar(label=ylabel)
    cmap=plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    norm = BoundaryNorm(norm_array, cmap.N,clip=False)
    add_flight_track_to_latlon(ax, norm,cmap,obs_col,df,date)

    ax = plt.subplot(222, projection=ccrs.PlateCarree())
    if is_log:
        horizontalplot(ax, glm2d, map_min, map_max, LogNorm())
    else:
        horizontalplot(ax, glm2d, map_min, map_max, None)
    plt.colorbar(label=ylabel)
    add_flight_track_to_latlon(ax, norm,cmap,obs_col,df,date)
    plt.savefig(path_model+saving_name)


def get_vertical_profiles(model_height_coord,interp_altitudes,interp_result):
    totals_in_altitude_bins=np.zeros(len(model_height_coord.points))
    numbers_in_altitude_bins=np.zeros(len(model_height_coord.points))
    totals_squared_in_altitude_bins=np.zeros(len(model_height_coord.points))
    model_altitude_bounds = model_height_coord.bounds
    ibin=0
    for height_interval in model_altitude_bounds:
        ialt=0
        for altitude in interp_altitudes:
            #if ialt==1:
            #    print 'altitude for vp',altitude
            if 0.001*altitude > height_interval[0] and 0.001*altitude < height_interval[1]:
                totals_in_altitude_bins[ibin]=totals_in_altitude_bins[ibin]+interp_result[ialt]
                numbers_in_altitude_bins[ibin]=numbers_in_altitude_bins[ibin]+1
                totals_squared_in_altitude_bins[ibin]=totals_squared_in_altitude_bins[ibin]+interp_result[ialt]*interp_result[ialt]
            ialt=ialt+1
        ibin=ibin+1
    means_in_bins = totals_in_altitude_bins/numbers_in_altitude_bins
    stds_in_bins = np.sqrt(totals_squared_in_altitude_bins/numbers_in_altitude_bins - np.power((totals_in_altitude_bins/numbers_in_altitude_bins),2))
    #print 'means_in_bins'
    #print means_in_bins
    return means_in_bins,stds_in_bins


def plot_vertical(df,model_height_coord,obs,lamvar,glmvar,xmax,xlabel,saving_name):
    #print model_height_coord
    means_obs,stds_obs = get_vertical_profiles(model_height_coord,df.altitude,obs)
    means_lam,stds_lam = get_vertical_profiles(model_height_coord,df.altitude,lamvar)
    means_glm,stds_glm = get_vertical_profiles(model_height_coord,df.altitude,glmvar)
    plt.figure()
    plt.semilogy(means_obs,model_height_coord.points,label='Obs', color='black',linewidth=1.5,basey=1.8 )
    plt.fill_betweenx(model_height_coord.points,means_obs-stds_obs,means_obs+stds_obs, color='silver', alpha=0.5)
    plt.semilogy(means_lam,model_height_coord.points,label='Regn.',basey=1.8,linewidth=1.5)
    plt.fill_betweenx(model_height_coord.points,means_lam-stds_lam,means_lam+stds_lam, color='lightskyblue',alpha=0.5)
    plt.semilogy(means_glm,model_height_coord.points,label='Global',basey=1.8,linewidth=1.5)
    plt.xlim(0,xmax)
    plt.ylim(1.5,6)
    plt.gca().yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks([],[])
    plt.tick_params(left=False)
    plt.yticks([1.5,2.0,3.0,4.0,5.0], [1.5,2.0,3.0,4.0,5.0])
    plt.xlabel(xlabel,fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylabel('Altitude (km)',fontsize=16)
    plt.legend()
    plt.savefig(path_model+saving_name)

##helpers to process model data
def add_orog(cube,surfcoord, dims):
    cube.add_aux_coord(surfcoord,dims)
    cube.coord('level_height').convert_units('km')
    #print 'sigma',cube.coord('sigma')
    factory = iris.aux_factory.HybridHeightFactory(delta=cube.coord('level_height'),sigma=cube.coord('sigma'),orography=cube.coord("surface_altitude"))
    cube.add_aux_factory(factory)
    return cube


def bbox_extract_2Dcoords(cube, bbox):
    minmax = lambda x: (np.min(x), np.max(x))
    lons = cube.coord('longitude').points
    lats = cube.coord('latitude').points
    inregion = np.logical_and(np.logical_and(lons > bbox[0],
                                             lons < bbox[1]),
                              np.logical_and(lats > bbox[2],
                                             lats < bbox[3]))
    region_inds = np.where(inregion)
    imin, imax = minmax(region_inds[0])
    jmin, jmax = minmax(region_inds[1])
    return cube[..., imin:imax+1, jmin:jmax+1]
def add_lat_lon(cube,bbox):
    polelat = cube.coord('grid_longitude').coord_system.grid_north_pole_latitude
    polelon = cube.coord('grid_longitude').coord_system.grid_north_pole_longitude
    source_lon = cube.coord('grid_longitude').points
    source_lat = cube.coord('grid_latitude').points
    lat2d = np.transpose(np.tile(source_lat,[len(source_lon),1]))
    lon2d = np.tile(source_lon,[len(source_lat),1])

    lons, lats = iris.analysis.cartography.unrotate_pole(lon2d, lat2d, polelon, polelat)

    longit = iris.coords.AuxCoord(lons,'longitude','longitude', units='degrees',coord_system=cs.GeogCS(6371229.0))
    latit =  iris.coords.AuxCoord(lats,'latitude','latitude', units='degrees',coord_system=cs.GeogCS(6371229.0))
    #print longit
    #print latit
    cube.add_aux_coord(longit, (2,3))
    cube.add_aux_coord(latit, (2,3))
    return bbox_extract_2Dcoords(cube, bbox)

def load_um_cube(timeindexes,surfcoord, inputfile,cubename, is_lam):
    cubelist = iris.cube.CubeList()
    for time_index in timeindexes:
        try:
            cubelist.append(iris.load_cube(inputfile + time_index, cubename))
        except Exception:
            # print cubename
            cubelist.append(iris.load(inputfile + time_index, cubename)[0])

    if cubelist[0].ndim==4:
        cube = cubelist.concatenate_cube()
    elif cubelist[0].ndim==3:
        cube = cubelist.merge_cube()
    add_orog(cube,surfcoord, (2,3,))
    # print(cube)
    if is_lam:
        add_lat_lon(cube,bbox)
    else:
        cube1 =cube.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
        cube = cube1.extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
    # print(cube)
    return cube

def calc_diameter(num_mixing,so4,bc,oc,ss=None):
    vol_so4_per_particle = (0.029*so4/1769.0)/(num_mixing*6.02e23)
    vol_bc_per_particle = (0.029*bc/1500.0)/(num_mixing*6.02e23)
    vol_oc_per_particle = (0.029*oc/1500.0)/(num_mixing*6.02e23)
    vol_per_particle = vol_so4_per_particle+vol_bc_per_particle+vol_oc_per_particle
    if ss !=None:
        vol_ss_per_particle = (0.029*ss/2165.0)*(num_mixing*6.02e23)
        vol_per_particle= vol_per_particle+vol_ss_per_particle
    vol_per_particle.units='m3'
    diam = iris.analysis.maths.exponentiate((6.0/3.14159)*vol_per_particle, (1.0/3.0))
    return diam
import scipy as sp
def lognormal_cumulative_forcubes(N,r,rbar,sigma):
    total=(N.data/2)*(1+sp.special.erf(np.log(r/rbar.data)/np.sqrt(2)/np.log(sigma)))
    return N.copy(total)

#get list of 4D cubes from either global or regional model
def load_model(timeindexes,path,date,is_lam,justso2):
    if is_lam:
        prefix = path+'umnsaa'
        denslabel='_pb'
        timeindexes_den=timeindexes
    else:
        prefix = path+'umglaa'
        denslabel='_pe'
        timeindexes_den = ['063','066','069']

    orog = iris.load_cube(prefix+'_pa000',iris.AttributeConstraint(STASH='m01s00i033'))
    surfcoord = iris.coords.AuxCoord(1e-3*orog.data,'surface_altitude', units='km')
    if not justso2:
        aird = (1.0/(6371000.0*6371000))*load_um_cube(timeindexes_den,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i253'),is_lam)

    so2_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_sulfur_dioxide_in_air', is_lam)
    so2_conc = so2_mixing*0.029/0.064*1e12 #pptv   
    if justso2:
        return iris.cube.CubeList([so2_conc])
    o3_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_ozone_in_air', is_lam)
    o3_conc = o3_mixing*0.029/0.048*1e9 #ppbv

    c5h8_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_isoprene_in_air',is_lam)
    c5h8_conc = c5h8_mixing*0.029/0.068*1e12 #pptv
    # c5h8_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pc', iris.AttributeConstraint(STASH='m01s34i027'),
    #                          is_lam)
    # c5h8_conc = c5h8_mixing * 0.029 / 0.068 * 1e12  # pptv
    mtlabel='_pd' #I dont have mt
    mt_mixing = load_um_cube(timeindexes,surfcoord,prefix+mtlabel,iris.AttributeConstraint(STASH='m01s34i091'),is_lam)
    mt_conc = mt_mixing*0.029/0.136*1e12 #pptv  
    oh_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pd',iris.AttributeConstraint(STASH='m01s34i081'),is_lam)
    oh_conc = oh_mixing * 0.029 / 0.017 * 1e12  # pptv
    h2so4_mixing = load_um_cube(timeindexes,surfcoord,prefix + '_pc', iris.AttributeConstraint(STASH='m01s34i073'),is_lam)
    h2so4_conc = h2so4_mixing * 0.029 / 0.098 * 1e12  # pptv


    # ho2_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pd',iris.AttributeConstraint(STASH='m01s34i082'),is_lam)
    # ho2_conc = ho2_mixing*0.029/0.033*1e12 #pptv
    nolabel='_pd'
    if is_lam:
        nolabel='_pc'
    no_mixing = load_um_cube(timeindexes,surfcoord,prefix+nolabel,iris.AttributeConstraint(STASH='m01s34i002'),is_lam)
    no_conc = no_mixing*0.029/0.030*1e12
    numlabel='_pb'
    if is_lam:
        numlabel='_pe'
        
    ait_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i103'),is_lam)
    acc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i107'),is_lam)
    cor_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i113'),is_lam)
    n10_num_mixing = ait_num_mixing+acc_num_mixing+cor_num_mixing
    if is_lam:
        ait_diam = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s38i402'),is_lam)
        acc_diam = load_um_cube(timeindexes, surfcoord, prefix+'_pd',iris.AttributeConstraint(STASH='m01s38i403'),is_lam)
    else:
        if new_diam_calc:
            ait_so4 =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i104'),is_lam)
            ait_bc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i105'),is_lam)
            ait_oc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i106'),is_lam)
            acc_so4 =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i108'),is_lam)
            acc_bc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i109'),is_lam)
            acc_oc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i110'),is_lam)
            acc_ss =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i111'),is_lam)
            ait_diam = calc_diameter(ait_num_mixing,ait_so4,ait_bc,ait_oc)
            acc_diam = calc_diameter(acc_num_mixing,acc_so4,acc_bc,acc_oc, acc_ss)
            iris.save(ait_diam,path+'Aitken_diameter_'+date+'.nc')
            iris.save(acc_diam,path+'Accumulation_diameter_'+date+'.nc')
            print(ait_diam )
            print('saved diameters')
        else:
            ait_diam = iris.load(path+'Aitken_diameter_'+date+'.nc')[0]
            acc_diam = iris.load(path+'Accumulation_diameter_'+date+'.nc')[0]
            print(ait_diam)
    n100_mixing = ait_num_mixing-lognormal_cumulative_forcubes(ait_num_mixing, 1.0e-7,ait_diam,1.59)
    n100_mixing = n100_mixing+acc_num_mixing - lognormal_cumulative_forcubes(acc_num_mixing, 1.0e-7,acc_diam,1.59)
    n100_mixing = n100_mixing+cor_num_mixing
    if is_lam:
        cdnc_in_cloud = 1e-6*load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s38i479'),is_lam) #num/m3->num/cc
        lwc_prefix='_pb'
    else:
        cdnc_in_cloud = 1e-6*load_um_cube(timeindexes,surfcoord,prefix+'_pc',iris.AttributeConstraint(STASH='m01s34i968'),is_lam) #num/m3->num/cc
        lwc_prefix='_pe'
    #lwc1 = 1e3*load_um_cube(timeindexes_den,surfcoord,prefix+lwc_prefix,iris.AttributeConstraint(STASH='m01s00i254'),is_lam)
    #lwc =lwc1.copy(lwc1.data*aird.data)
    # want number concentration at STP per cc. Good, because I can't easily multiply by density anyway as pe is written every 6 hours. Change this!
    pref=1.013E5
    tref=293.0 #Assume STP is 20C
    zboltz=1.3807E-23
    staird=pref/(tref*zboltz*1.0E6)
    n10_num_conc = n10_num_mixing*staird
    n100_num_conc = n100_mixing*staird
    print(so2_conc)
    #return iris.cube.CubeList([so2_conc,oh_conc,co_conc,c5h8_conc,mt_conc,no_conc,n10_num_conc,n100_num_conc,cdnc_in_cloud])#,lwc])
    return iris.cube.CubeList([oh_conc,h2so4_conc,c5h8_conc,no_conc,o3_conc]) #,lwc])

# at 50m/s, 1 minute of flight to cover 1 grid cell
def read_flight_track(flight_path,headerlen, col1,col2,col3):   #output 3 column variable
    var1_timeseries=[]
    var2_timeseries=[]
    var3_timeseries=[]
    utctime=[] # seconds from midnight on July 29 2014                                                                      
    iline=0
    with open(flight_path) as fp:
        line = fp.readline()
        while line:
            if iline < headerlen:#length of header                                                                                                            
                line = fp.readline()
                iline+=1
                continue
            line = fp.readline()
            data =line.split(',')
            try:
                utctime.append(float(data[0]))
            except Exception:
                break
            var1_timeseries.append(float(data[col1]))
            var2_timeseries.append(float(data[col2]))
            var3_timeseries.append(float(data[col3]))
            #if iline%1000==0:
            #    print iline,float(data[0]),float(data[col1])
            iline=iline+1
    return utctime,var1_timeseries,var2_timeseries,var3_timeseries #time and 3 variables


def interp_flight_data(track_time,gas_time,gas_conc,l_o_d):
    gas_a = ma.masked_where(np.asarray(gas_conc) < l_o_d, np.asarray(gas_conc))
    gas_time_a = np.asarray(gas_time)[~gas_a.mask]
    gas_a = gas_a[~gas_a.mask]
    print gas_time_a.shape,gas_a.shape
    if(len(gas_time_a.shape) > 1):
        print 'strange shape of array'
        gas_time_a = gas_time_a[0]
        gas_a = gas_a[0]
        print gas_time_a.shape,gas_a.shape
    interp_gas = interp1d(gas_time_a,gas_a,kind='nearest', fill_value='extrapolate')
    gas_track = interp_gas(np.asarray(track_time))
    return gas_track
                           
def read_flight_data(date):
    flight_path='/jet/home/ding0928/Colorado/Colorado/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict'
    headerlen=131
    tracktime,alt,lat,lon = read_flight_track(flight_path,headerlen, 4,5,6)  #lat/lon/alt
    tracktime,temp,pres,pres2 = read_flight_track(flight_path, headerlen, 4, 29, 62)  # I get the pressure wrong
    print 'filter',len(tracktime)
    tracktime = ma.masked_where(np.asarray(lat) < -90,np.array(tracktime))
    tracktime2 = ma.masked_where(np.asarray(lon) < -180,np.array(tracktime))
    print 'filter',len(tracktime[~tracktime.mask])
    tracktime = tracktime2[~tracktime.mask]
    lon = np.array(lon)[~tracktime2.mask]
    lat = np.array(lat)[~tracktime2.mask]
    temp = np.array(temp)[~tracktime2.mask]
    pres=np.array(pres)[~tracktime2.mask]
    pres2=np.array(pres2)[~tracktime2.mask]
    temptrack = interp_flight_data(tracktime, tracktime, temp, -99)
    prestrack = interp_flight_data(tracktime, tracktime, pres, -99)
    pres2track = interp_flight_data(tracktime, tracktime, pres2, -99)


    h2so4_path='/jet/home/ding0928/Colorado/Colorado/FRAPPE-OH-H2SO4_C130_20140731_R0.ict'
    h2so4time,oh,h2so4,dummy = read_flight_track(h2so4_path,100,3,4,1)
    ohtrack = interp_flight_data(tracktime,h2so4time,oh, -99)
    h2so4track = interp_flight_data(tracktime, h2so4time, h2so4, -99)
    # ohtrack=ohtrack/2.5
    # h2so4track=h2so4track/2.5

    c5h8_path = '/jet/home/ding0928/Colorado/Colorado/FRAPPE-TOGA_C130_20140731_R2.ict'
    c5h8time, c5h8, c10h16, dummy = read_flight_track(c5h8_path, 92, 16, 51, 1)
    c5h8track = interp_flight_data(tracktime, c5h8time, c5h8, -99)
    c10h16track = interp_flight_data(tracktime, c5h8time, c10h16, -99)

    no_o3_path = '/jet/home/ding0928/Colorado/Colorado/FRAPPE-NONO2O3_C130_20140731_R0.ict'
    notime, no, no2, o3 = read_flight_track(no_o3_path, 40, 3, 4, 5)
    notrack = interp_flight_data(tracktime, notime, no, -99)
    o3track = interp_flight_data(tracktime, notime, o3, -99)


    n0=(pres2track*100*6.02e23)/(8.314*(temptrack+273.15))
    coef=n0*1e-18
    ohtrack= ohtrack / coef
    h2so4track=h2so4track/coef
    # c5h8track=c5h8track/coef
    # c10h16track=c10h16track/coef
    print('pres2track:',pres2track)
    print('prestrack:',prestrack)
    print('temptrack:',temptrack)
    # print('pres2',pres2)


    ho2ro2_path ='/jet/home/ding0928/Colorado/Colorado/FRAPPE-HO2RO2_C130_20140731_R1.ict'
    ho2ro2time,ho2ro2,ho2,dummy = read_flight_track(ho2ro2_path,100,5,3,1)
    ho2ro2track = interp_flight_data(tracktime,ho2ro2time,ho2ro2,-10)
    ho2track = interp_flight_data(tracktime,ho2ro2time,ho2,-10)


    unix_epoch_of_31July2014_0001 = 1406764800
    if date == '20140801':
        unix_epoch_of_01August2014_0001 = unix_epoch_of_31July2014_0001 + 86400
    elif date == '20160512':
        unix_epoch_of_02August2014_0001 = unix_epoch_of_31July2014_0001 + 86400 * 2
    trackpdtime = pd.to_datetime(np.asarray(tracktime) + unix_epoch_of_31July2014_0001, unit='s')



    d= {'time':pd.Series(trackpdtime), 'latitude':pd.Series(np.asarray(lat)),'longitude':pd.Series(np.asarray(lon)),
        'altitude':pd.Series(np.asarray(alt)),'OH':pd.Series(ohtrack),'H2SO4':pd.Series(h2so4track),'C5H8':pd.Series(c5h8track),'NO':pd.Series(notrack),'O3':pd.Series(o3track)}
    # d= {'time':pd.Series(trackpdtime), 'latitude':pd.Series(np.asarray(lat)),'longitude':pd.Series(np.asarray(lon)),
    #     'altitude':pd.Series(np.asarray(alt)), 'OH':pd.Series(ohtrack),'HO2':pd.Series(ho2track),'HO2RO2':pd.Series(ho2ro2track),
    #     'H2SO4':pd.Series(h2so4track)}
    df3 = pd.DataFrame(d)
    df2 = df3.resample('1T',on='time').mean()
    print df2.head()
    return df2

# also works for 3D cubes with no time coordinate
def do_4d_interpolation(cubelist,flight_coords, is_lam):
    cube = cubelist[0]
    if is_lam:
        input_projection=cube.coord('grid_longitude').coord_system.as_cartopy_projection()
    else:
        input_projection=cube.coord('longitude').coord_system.as_cartopy_projection()
    #my_interpolating_function = RegularGridInterpolator((cube.coord('grid_longitude').points, cube.coord('grid_latitude').points, cube.coord('level_height').points), cube.data.T) #This is the simple version that does not account for orography
    lats_of_flight = flight_coords['latitude']
    lons_of_flight = flight_coords['longitude']
    alts_of_flight = flight_coords['altitude']
    #times_of_flight = flight_coords.index
    epoch = datetime.datetime(2014, 7, 31)
    times_of_flight=[(d - epoch).total_seconds() for d in flight_coords.index]
    points=[]
    if cube.ndim==4:
        cube_times1 = [cell.point for cell in cube.coord('time').cells()]
        cube_times = [(d - epoch).total_seconds() for d in cube_times1]
        print cube_times
    if is_lam:
        print 'center of grid,lon,lat',cube.coord('grid_longitude').points[150],cube.coord('grid_latitude').points[150]
        print 'offset-lon of grid,lon,lat',cube.coord('grid_longitude').points[120],cube.coord('grid_latitude').points[150]
        print 'offset-lat of grid,lon,lat',cube.coord('grid_longitude').points[150],cube.coord('grid_latitude').points[120]
    for ipoint in range(0,len(lats_of_flight)):
        rotated_lon,rotated_lat = input_projection.transform_point(lons_of_flight[ipoint],lats_of_flight[ipoint],crs_latlon)
        if ipoint < 20:
            print 'rotated_pole,time,lon,lat',times_of_flight[ipoint],lons_of_flight[ipoint],lats_of_flight[ipoint],rotated_lon,rotated_lat
        if cube.ndim==4:
            if is_lam:
                points.append([times_of_flight[ipoint],rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])
            else:
                points.append([times_of_flight[ipoint],rotated_lon,rotated_lat,1e-3*alts_of_flight[ipoint]])
        else:
            if is_lam:
                points.append([times_of_flight[ipoint],rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])
            else:
                points.append([times_of_flight[ipoint],rotated_lon,rotated_lat,1e-3*alts_of_flight[ipoint]])
    cubedimcoords=[]
    cubedatasets=[]
    if cube.ndim==4:
        cubedimcoords.append(cube_times)
        for cube_to_interp in cubelist:
            cubedatasets.append(np.transpose(cube_to_interp.data,(0,3,2,1)))
            print cube_to_interp
    else:
        for cube_to_interp in cubelist:
            cubedatasets.append(cube_to_interp.data.T)
    if is_lam:
        cubedimcoords.append(cube.coord('grid_longitude').points)
        cubedimcoords.append(cube.coord('grid_latitude').points) 
    else:
        cubedimcoords.append(cube.coord('longitude').points)
        cubedimcoords.append(cube.coord('latitude').points)
    cubedimcoords.append(cube.coord('level_height').points)
    
    print 'interpolator hybrid_dim args',cube.coord_dims(cube.coord('altitude'))
    # complicated version of interpolator that handles topography. Credit to Duncan Watson-Parris (cis) and scitools
    interpolator=_RegularGridInterpolator(cubedimcoords,np.asarray(points).T, hybrid_coord =cube.coord('altitude').points.T,hybrid_dims=cube.coord_dims(cube.coord('altitude')),method='nn')
    print 'setup interpolation'
    #interpolated_values=my_interpolating_function(points)
    interp_results=[]
    for dataset_to_interp in cubedatasets:
        interp_results.append(np.asarray(interpolator(dataset_to_interp, fill_value=None)))
    #print interpolated_values
    return interp_results

#plt.plot(np.asarray(lat),so2track)


#plt.plot(df.latitude,df.SO2)
##read in model data
# path_model='/ocean/projects/atm200005p/ding0928/cylc-run/u-cc346/share/cycle/20140729T0000Z/Regn1/resn_1/RA1M/um/'
path_model='/jet/home/ding0928/cylc-run/u-cl893/share/cycle/20140729T0000Z/Regn1/resn_1/RA2M/um/'
# path_model='/jet/home/ding0928/cylc-run/u-cl893/share/cycle/20140729T0000Z/Regn1/resn_1/RA2M/um/'
# path_glm = '/ocean/projects/atm200005p/ding0928/cylc-run/u-cc346/share/cycle/20140729T0000Z/glm/um/'
path_glm = '/jet/home/ding0928/cylc-run/u-cl893/share/cycle/20140729T0000Z/glm/um/'

date= '20140731'
if date=='20140731':
    timeindexes_glm=['063','066','069']
    timeindexes_lam=['060','066']
    lonmin=-110
    lonmax=-100
    latmin=35
    latmax=45

bbox = [lonmin,lonmax,latmin,latmax]
crs_latlon = ccrs.PlateCarree()
#annoyingly only minor speedup
if make_dataset==1:
    df = read_flight_data(date)
    conc_list = load_model(timeindexes_lam, path_model,date,1,0)
    array_of_model_data= do_4d_interpolation(conc_list,df, 1)
    # print len(array_of_model_data[0]),len(array_of_model_data[1])
    # print array_of_model_data[1]
    # print len(df['SO2'])
   # [df['model_SO2'],df['model_OH'],df['model_CO'],df['model_C5H8'],
    # df['model_C10H16'],df['model_NO'],df['model_N10'],df['model_N100'],df['model_CDNC']] = array_of_model_data
    # [df['model_SO2'],df['model_OH'],df['model_H2SO4'],df['model_C5H8'],df['model_NO']] = array_of_model_data
    [df['model_OH'],df['model_H2SO4'],df['model_C5H8'],df['model_NO'],df['model_O3']] = array_of_model_data

    conc_list_glm = load_model(timeindexes_glm, path_glm,date,0,0)
    # [df['glm_SO2'],df['glm_OH'],df['glm_CO'],df['glm_C5H8'],
    #  df['glm_C10H16'],df['glm_NO'],df['glm_N10'],df['glm_N100'],df['glm_CDNC']] = do_4d_interpolation(conc_list_glm,df, 0)
    # [df['glm_SO2'],df['glm_OH'],df['glm_H2SO4'],df['glm_C5H8'],df['glm_NO']] = do_4d_interpolation(conc_list_glm,df, 0)
    [df['glm_OH'],df['glm_H2SO4'],df['glm_C5H8'],df['glm_NO'],df['glm_O3']] = do_4d_interpolation(conc_list_glm,df, 0)

    # print df.head()
    df.to_csv(path_model+'interpolated_korus_dataset_'+date+'.csv')
else:
    df = pd.read_csv(path_model+'interpolated_korus_dataset_'+date+'.csv',index_col=0, parse_dates=True)
    # print df.head()
    conc_list = load_model(timeindexes_lam, path_model,date,1,justso2)
    conc_list_glm = load_model(timeindexes_glm, path_glm,date,0,justso2)


# norm_array_so2 = [2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000, 50000,100000]
# plot_comparison(df,df['SO2'],df['model_SO2'],df['glm_SO2'],(conc_list[0])[2,3,:,:],(conc_list_glm[0])[1,1,:,:],
#                 'SO$_{2}$ (pptv)',2.0,100000,2.0,100000,True,norm_array_so2,'korus_so2_'+date+'_bt116.png', date)
# plot_vertical(df,conc_list[0].coord('level_height'),df['SO2'],df['model_SO2'],df['glm_SO2'],5000, 'SO$_{2}$ (pptv)','korus_so2_vp_'+date+'_bt116.png')
# if justso2==1:
#     plt.show()
#     sys.exit()

norm_array_oh=[0,0.1,0.2,0.3,0.4,0.5,1,2,5]
plot_comparison(df,df['OH'],df['model_OH'],df['glm_OH'],(conc_list[0])[2,3,:,:],(conc_list_glm[0])[1,1,:,:],
                'OH (pptv)', 0,5,0,1,False,norm_array_oh,'korus_oh_'+date+'_bt116.png', date)
plot_vertical(df,conc_list[0].coord('level_height'),df['OH'],df['model_OH'],df['glm_OH'],6, 'OH (pptv)','korus_oh_vp_'+date+'_bt116.png')


# norm_array_ho2=[0,5,10,20,30,40,50,60]
# plot_comparison(df,df['HO2'],df['model_HO2'],df['glm_HO2'],(conc_list[0])[2,3,:,:],(conc_list_glm[0])[1,1,:,:],
#                 'HO2 (pptv)', 0,60,0,60,False,norm_array_ho2,'korus_ho2_'+date+'_bt116.png', date)
# plot_vertical(df,conc_list[0].coord('level_height'),df['HO2'],df['model_HO2'],df['glm_HO2'],60, 'HO2 (pptv)','korus_ho2_vp_'+date+'_bt116.png')
# #
# norm_array_ho2ro2=[0,5,10,40,60,80,100,150]
# plot_comparison(df,df['HO2RO2'],df['model_HO2RO2'],df['glm_HO2RO2'],(conc_list[2])[2,3,:,:],(conc_list_glm[2])[1,1,:,:],
#                 'HO2RO2 (pptv)', 0,150,0,150,False,norm_array_ho2ro2,'korus_ho2ro2_'+date+'_bt116.png', date)
# plot_vertical(df,conc_list[0].coord('level_height'),df['HO2RO2'],df['model_HO2RO2'],df['glm_HO2RO2'],150, 'HO2RO2 (pptv)','korus_ho2ro2_vp_'+date+'_bt116.png')
#
#
norm_array_h2so4=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,5.0]
plot_comparison(df,df['H2SO4'],df['model_H2SO4'],df['glm_H2SO4'],(conc_list[1])[2,3,:,:],(conc_list_glm[1])[1,1,:,:],
                'H2SO4 (pptv)', 0,5,0,1,False,norm_array_h2so4,'korus_h2so4_'+date+'_bt116.png', date)
plot_vertical(df,conc_list[0].coord('level_height'),df['H2SO4'],df['model_H2SO4'],df['glm_H2SO4'],4, 'H2SO4 (pptv)','korus_h2so4_vp_'+date+'_bt116.png')

# norm_array_co = [0,50,100,150,200,250,300,350]
# plot_comparison(df,df['CO'],df['model_CO'],df['glm_CO'],(conc_list[2])[2,3,:,:],(conc_list_glm[2])[1,1,:,:],
#                 'CO (ppbv)', 0,350,0,350,False,norm_array_co,'korus_co_'+date+'_bt116.png', date)
# plot_vertical(df,conc_list[0].coord('level_height'),df['CO'],df['model_CO'],df['glm_CO'],350, 'CO (ppbv)','korus_co_vp_'+date+'_bt116.png')

norm_array_c5h8=[0.01,0.03,1.0,3.0,10,30,100,300,1000.0]
plot_comparison(df,df['C5H8'],df['model_C5H8'],df['glm_C5H8'],(conc_list[2])[2,3,:,:],(conc_list_glm[2])[1,1,:,:],
                'C5H8 (pptv)',1,1000,0.01,1000.0,False,norm_array_c5h8,'korus_c5h8_'+date+'_bt116.png', date)
plot_vertical(df,conc_list[0].coord('level_height'),df['C5H8'],df['model_C5H8'],df['glm_C5H8'],800, 'C5H8 (pptv)','korus_c5h8_vp_'+date+'_bt116.png')

# norm_array_c10h16=[0,2,4,6,8,10,12,14,16,18,20]
# plot_comparison(df,df['C10H16'],df['model_C10H16'],df['glm_C10H16'],(conc_list[4])[2,3,:,:],(conc_list_glm[4])[1,1,:,:],
#                 'C10H16 (pptv)', 1,140,0.01,140.0,True,norm_array_c10h16,'korus_c10h16_'+date+'_bt116.png', date)
# print conc_list_glm[3].coord('level_height')[0:30]

norm_array_no = [5,10,20,50,100,200,300,500,700,1000,1500,2500]
plot_comparison(df,df['NO'],df['model_NO'],df['glm_NO'],(conc_list[3])[2,3,:,:],(conc_list_glm[3])[1,1,:,:],
                'NO (pptv)', 5,2000,20,500,True,norm_array_no,'korus_no_'+date+'_bt116.png', date)
plot_vertical(df,conc_list[0].coord('level_height'),df['NO'],df['model_NO'],df['glm_NO'],1000, 'NO (pptv)','korus_no_vp_'+date+'_bt116.png')

norm_array_o3=[0,0.1,0.2,0.3,0.4,0.5,1,10,20,50,70,80,100]
plot_comparison(df,df['O3'],df['model_O3'],df['glm_O3'],(conc_list[4])[2,3,:,:],(conc_list_glm[4])[1,1,:,:],'O3 (ppbv)', 0,100,0,100,False,norm_array_o3,'korus_O3_'+date+'_bt116.png', date)
plot_vertical(df,conc_list[0].coord('level_height'),df['O3'],df['model_O3'],df['glm_O3'],100, 'O3 (ppbv)','korus_O3_vp_'+date+'_bt116.png')


# norm_array_num= [500,1000,2000,5000,10000,20000,50000]
# plot_comparison(df,df['N10'],df['model_N10'],df['glm_N10'],(conc_list[6])[2,3,:,:],(conc_list_glm[6])[1,1,:,:],
#                 'N10 (cm-3 stp)', 500,50000,500,50000.0,True,norm_array_num,'korus_n10_'+date+'_bt116.png', date)
# plot_vertical(df,conc_list[0].coord('level_height'),df['N10'],df['model_N10'],df['glm_N10'],14000, 'N10 (cm-3 stp)','korus_n10_vp_'+date+'_bt116.png')
# norm_array_num= [50,100,200,500,1000,2000,5000,10000]
# plot_comparison(df,df['N100'],df['model_N100'],df['glm_N100'],(conc_list[7])[2,3,:,:],(conc_list_glm[7])[1,1,:,:],
#                 'N100 (cm-3 stp)', 50,10000,50,10000.0,True,norm_array_num,'korus_n100_'+date+'_bt116.png', date)
# plot_vertical(df,conc_list[0].coord('level_height'),df['N100'],df['model_N100'],df['glm_N100'],5000, 'N100 (cm-3 stp)','korus_n100_vp_'+date+'_bt116.png')
# norm_array_cdnc = [0,100,200,300,400,500,600,700,800,900,1000]
# print conc_list[8].coord('level_height')[0:30]
# plot_comparison(df,df['CPSPD'],df['model_CDNC'],df['glm_CDNC'],(conc_list[8])[2,20,:,:],(conc_list_glm[8])[1,13,:,:],
#                 'CDNC (cm-3 amb)', 0,1000,0,1000.0,False,norm_array_cdnc,'korus_cdnc_'+date+'_bt116.png', date)
# #plot_comparison(df,df['CPSLWC'],df['model_LWC'],df['glm_LWC'],(conc_list[9])[2,20,:,:],(conc_list_glm[9])[1,13,:,:],
# #                'LWC (g/kg)', 0,1000.0,False,norm_array_cdnc,'korus_lwc_'+date+'_bt116.png')
plt.show()
