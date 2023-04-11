import iris,sys,glob
import iris.quickplot as qplt
import iris.plot as iplt
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from iris.util import unify_time_units
from iris.experimental.equalise_cubes import equalise_attributes
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
# from obser_for_Merge_0721_altagl import merge_obs_method
from obser_for_Merge_0722_altagl import merge_obs_method
import matplotlib as mpl
import datetime
import time
import matplotlib.dates as mdates
from pytz import timezone
#from cis.data_io.ungridded_data import UngriddedData
#from cis.data_io.hyperpoint import HyperPoint
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator
import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
# from J_nuc_rate_v2 import J_cal


#flags for whether to make and write intermediate files or use them
new_diam_calc=0 #try 0 or 1(old is 0)
make_dataset=1
plt.rcParams['font.size'] = 16
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
                plt.text(df.longitude[itime],df.latitude[itime],timestring, fontsize=16)
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
    gl.xlocator = mticker.FixedLocator([lonmin,lonmin+1,0.5*(lonmin+lonmax),lonmax-1,lonmax])
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter(LONGITUDE_FORMATTER)
    gl.yformatter(LATITUDE_FORMATTER)

#plot the comparison figure but black observation is missing
def plot_comparison(df,obs_col,lam_col,glm_col,lam2d,glm2d,ylabel,var_min,var_max,map_min,map_max,is_log,norm_array,saving_name,date,plot_obs=1):
    plt.figure(figsize=(8,4.8))
    plt.rcParams['font.size'] = 16
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

    # plt.rcParams['font.size'] = 24
    plt.subplot(212)
    if plot_obs==1:
        ax = obs_col.plot(logy=is_log,ylim=(var_min,var_max),label='Obs.', color='black',linewidth=1.5)
        lam_col.plot(logy=is_log,ylim=(var_min,var_max),label='Regn.' )
    else:
        ax = lam_col.plot(logy=is_log,ylim=(var_min,var_max),label='Regn.' )
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # fix the bug of UTC-Mountain in line 973, so now we have local time(Mountain) again
    #and we put this time conversion after 1. sample all the data 2. finish interpolation
    # then last step is time conversion
    date_form = mdates.DateFormatter("%H:%M")
    date_form.set_tzinfo(timezone('US/Mountain'))
    ax.xaxis.set_major_formatter(date_form)
    ax2 = glm_col.plot(logy=is_log,ylim=(var_min,var_max),label='Global')
    ax2.set_ylabel(ylabel)
    ax3 = df['model_Altitude_AGL'].plot(secondary_y=True,label='Altitude')
    ax3.set_ylabel('AGL (m)')
    ax3.set_ylim(0,8000)

    h2,l2 = ax2.get_legend_handles_labels()
    h3,l3 = ax3.get_legend_handles_labels()
    plt.legend(h2+h3,l2+l3,bbox_to_anchor=(1.0, 1.1),ncol=2)

    # plt.savefig(path_model+saving_name)
    plt.savefig('/jet/home/ding0928/Colorado/Colorado/p-3b/fig_11.9'+saving_name, dpi=1000)


def get_vertical_profiles(model_height_coord,interp_altitudes,interp_result):
    totals_in_altitude_bins=np.zeros(len(model_height_coord.points)-6)
    numbers_in_altitude_bins=np.zeros(len(model_height_coord.points)-6)
    totals_squared_in_altitude_bins=np.zeros(len(model_height_coord.points)-6)
    model_altitude_bounds = model_height_coord.bounds
    ibin=0
    altitude_bounds_to_use = [[0,0.230]]
    i=0
    altitude_points_to_use=[0.115]
    for height_interval in model_altitude_bounds[7:]:
        altitude_bounds_to_use.append(height_interval)
        altitude_points_to_use.append(model_height_coord.points[i+7])
        i=i+1
    for height_interval in altitude_bounds_to_use:
        ialt=0
        for altitude in interp_altitudes:
            #if ialt==1:
            #    print 'altitude for vp',altitude
            if 0.001*altitude > height_interval[0] and 0.001*altitude < height_interval[1]:
                if not np.isnan(interp_result[ialt]):
                    totals_in_altitude_bins[ibin]=totals_in_altitude_bins[ibin]+interp_result[ialt]
                    numbers_in_altitude_bins[ibin]=numbers_in_altitude_bins[ibin]+1
                    totals_squared_in_altitude_bins[ibin]=totals_squared_in_altitude_bins[ibin]+interp_result[ialt]*interp_result[ialt]
            ialt=ialt+1
        ibin=ibin+1
    means_in_bins = totals_in_altitude_bins/numbers_in_altitude_bins
    stds_in_bins = np.sqrt(totals_squared_in_altitude_bins/numbers_in_altitude_bins - np.power((totals_in_altitude_bins/numbers_in_altitude_bins),2))
    #print 'means_in_bins'
    #print means_in_bins
    #print altitude_points_to_use,means_in_bins
    return means_in_bins,stds_in_bins,altitude_points_to_use


def plot_vertical(df,model_height_coord,obs,lamvar,glmvar,xmax,xlabel,saving_name, plot_obs=1, x_min=0):
    #print model_height_coord
    if plot_obs==1:
        means_obs,stds_obs,altitude_points_to_use = get_vertical_profiles(model_height_coord,df.altitude,obs)
    means_lam,stds_lam,altitude_points_to_use = get_vertical_profiles(model_height_coord,df.altitude,lamvar)
    means_glm,stds_glm,altitude_points_to_use = get_vertical_profiles(model_height_coord,df.altitude,glmvar)
    plt.figure()
    if plot_obs==1:
        plt.semilogy(means_obs,altitude_points_to_use,label='Obs', color='black',linewidth=1.5,basey=1.8 )
        plt.fill_betweenx(altitude_points_to_use,means_obs-stds_obs,means_obs+stds_obs, color='silver', alpha=0.5)
    plt.semilogy(means_lam,altitude_points_to_use,label='Regn.',basey=1.8,linewidth=1.5)
    plt.fill_betweenx(altitude_points_to_use,means_lam-stds_lam,means_lam+stds_lam, color='lightskyblue',alpha=0.5)
    plt.semilogy(means_glm,altitude_points_to_use,label='Global',basey=1.8,linewidth=1.5)
    plt.xlim(x_min,xmax)
    plt.ylim(1.5,5.5)
    plt.gca().yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks([],[])
    plt.tick_params(left=False)
    plt.yticks([1.5,2.0,3.0,4.0,5.0], [1.5,2.0,3.0,4.0,5.0])
    plt.xlabel(xlabel,fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylabel('Altitude (km)',fontsize=16)
    plt.legend()
    plt.savefig('/jet/home/ding0928/Colorado/Colorado/p-3b/fig_11.9'+saving_name, dpi=1000)

##helpers to process model data
# makes a 3D cube above sea level cube from altitude cube?
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
            cubelist.append(iris.load_cube(inputfile+time_index,cubename))
        except Exception:
            # print cubename
            cubelist.append(iris.load(inputfile+time_index,cubename)[0])
    print('cubelist[0].ndim',cubelist[0].ndim)
    if cubelist[0].ndim==4: #several time slots=4d dimension
        cube = cubelist.concatenate_cube()
        print('cube_4d',cube) #just need to connect
    elif cubelist[0].ndim==3: #some cube only has 1time slot=3d dimension,
        cube = cubelist.merge_cube() #need to make a list from 0,so called merge
        print('cube_3d', cube)
    add_orog(cube,surfcoord, (2,3,))
    print('cube_as H suggested',cube)
    if is_lam:
        add_lat_lon(cube,bbox)
    else:
        cube1 =cube.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
        cube = cube1[:,:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
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
def load_model(timeindexes,path,varlist,date,is_lam,is_dump):
    outputcubedict={}
    if is_lam:
        prefix = path+'umnsaa'
        denslabel='_pb'
        timeindexes_den=timeindexes
    else:
        prefix = path+'umglaa'
        denslabel='_pe'
        timeindexes_den = timeindexes
    if not is_dump: # pa
        orog = iris.load_cube(prefix+'_pa000',iris.AttributeConstraint(STASH='m01s00i033'))
        print('before trimming orog[:,:]',orog[:,:]) #144*192, reg:300*300
        # this orog originally is 300*300 for regional+144*192 foe global; thus need to trim down
        if is_lam: #if regional model, use box indexes to trim down
            orog = orog[50:251,50:251]
        else: # if global model, use lat/lon to constrain a region.
            orog1 = orog.extract(iris.Constraint(latitude=lambda cell: latmin - 1.0 < cell < latmax + 1))
            orog = orog1.extract(iris.Constraint(longitude=lambda cell: lonmin + 359 < cell < lonmax + 361))
            # a loop over model vertical levels,orog is 2d cubes,first_cube.coord.points[i,:,:] is a 2-d cube,
        print('after trimming orog[:,:]', orog[:, :]) #11*9;  reg:201*201
        surfcoord = iris.coords.AuxCoord(1e-3*orog.data,'surface_altitude', units='km')
        print('surfcoord with trimmed orog.data', surfcoord) #2d standard_name='surface_altitude', units=Unit('km')

    #aird = (1.0/(6371000.0*6371000))*load_um_cube(timeindexes_den,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i253'),is_lam)
    if 'T' in varlist:  #checked,
        if not is_dump:
            theta = load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i004'),is_lam) #air_potential_temperature / (K) (model_level_number: 40; latitude: 11; longitude: 9)
            air_pressure = load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i408'),is_lam) #air_pressure / (Pa) (model_level_number: 40; latitude: 11; longitude: 9)
        else:
            theta = iris.load_cube(path,iris.AttributeConstraint(STASH='m01s00i004')) #theta
            cube1 =theta.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
            theta = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
            exner = iris.load_cube(path,iris.AttributeConstraint(STASH='m01s00i255'))
            cube1 =exner.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
            exner = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
        p0 = iris.coords.AuxCoord(1000.0,long_name='reference_pressure',units='hPa')

        if not is_dump:
            p0.convert_units(air_pressure.units)
        Rd=287.05
        cp=1005.46
        Rd_cp=Rd/cp
        if not is_dump:
            temperature=theta*(air_pressure/p0)**(Rd_cp)
        else:
            temperature = theta.copy(theta.data*exner.data)
            print 'DUMP TEMP ALTITUDE COLM',temperature.coord('altitude').points[:,25,25]
            print 'MINIMUM SURFACE TEMP',np.min(temperature.data[0,:,:])
        outputcubedict["T"]  =temperature
    if 'SO2' in varlist:
        so2_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_sulfur_dioxide_in_air', is_lam)
        so2_conc = so2_mixing*0.029/0.064*1e12 #pptv
        outputcubedict["SO2"] = so2_conc
    if 'O3' in varlist:
        o3_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_ozone_in_air', is_lam)
        o3_conc = o3_mixing*0.029/0.048*1e9 #ppbv
        outputcubedict["O3"] = o3_conc
    if 'C5H8' in varlist:
        c5h8_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_isoprene_in_air',is_lam)
        c5h8_conc = c5h8_mixing*0.029/0.068*1e12 #pptv
        outputcubedict["C5H8"] = c5h8_conc
    if 'C10H16' in varlist:
        mtlabel='_pd' #I dont have mt
        mt_mixing = load_um_cube(timeindexes,surfcoord,prefix+mtlabel,iris.AttributeConstraint(STASH='m01s34i091'),is_lam)
        mt_conc = mt_mixing*0.029/0.136*1e12 #pptv
        outputcubedict["C10H16"] = mt_conc

    if 'CO' in varlist: #this is new added
        colabel='_pc' #I dont have mt
        co_mixing = load_um_cube(timeindexes,surfcoord,prefix+colabel,iris.AttributeConstraint(STASH='m01s34i010'),is_lam)
        co_conc = co_mixing*0.029/0.030*1e9    #ppbv notice CO is ppb instead of ppt
        outputcubedict["CO"] = co_conc

    if 'OH' in varlist:
        oh_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pd',iris.AttributeConstraint(STASH='m01s34i081'),is_lam)
        oh_conc = oh_mixing * 0.029 / 0.017 * 1e12  # pptv
        outputcubedict["OH"]=oh_conc
    if 'H2SO4' in varlist:
        h2so4_mixing = load_um_cube(timeindexes,surfcoord,prefix + '_pc', iris.AttributeConstraint(STASH='m01s34i073'),is_lam)
        h2so4_conc = h2so4_mixing * 0.029 / 0.098 * 1e12  # pptv
        outputcubedict["H2SO4"]=h2so4_conc
    if 'NH3' in varlist:
        nh3_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_ammonia_in_air', is_lam)
        nh3_conc = nh3_mixing*0.029/0.017*1e9 #ppbv
        outputcubedict["NH3"]=nh3_conc
    if 'HO2' in varlist:
        ho2_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pd',iris.AttributeConstraint(STASH='m01s34i082'),is_lam)
        ho2_conc = ho2_mixing*0.029/0.033*1e12 #pptv
        outputcubedict["HO2"]=ho2_conc
    #from now on, some of the composition is missing for u-cs093
    if 'SO4_Coa' in varlist:  # missing u-cs093(1211)
        so4_coa_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                      iris.AttributeConstraint(STASH='m01s34i114'), is_lam)
        outputcubedict["SO4_Coa"] = so4_coa_mixing
    if 'BC_Coa' in varlist:  # # missing u-cs093(1211)
        bc_coa_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                     iris.AttributeConstraint(STASH='m01s34i115'), is_lam)
        outputcubedict["BC_Coa"] = bc_coa_mixing
    if 'OC_Coa' in varlist:  # # missing u-cs093(1211)
        oc_coa_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                     iris.AttributeConstraint(STASH='m01s34i116'), is_lam)
        outputcubedict["OC_Coa"] = oc_coa_mixing
    if 'SS_Coa' in varlist:  # # missing u-cs093(1211)
        ss_coa_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                     iris.AttributeConstraint(STASH='m01s34i117'), is_lam)
        outputcubedict["SS_Coa"] = ss_coa_mixing
    if 'NH4_Coa' in varlist:  # yes u-cs093(1211)
        nh4_coa_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                      iris.AttributeConstraint(STASH='m01s34i135'), is_lam)
        outputcubedict["NH4_Coa"] = nh4_coa_mixing
    if 'NO3_Coa' in varlist:  # yes u-cs093(1211)
        no3_coa_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                      iris.AttributeConstraint(STASH='m01s34i139'), is_lam)
        outputcubedict["NO3_Coa"] = no3_coa_mixing

    if 'SO4_Acc' in varlist:  # yes u-cs093(1211)
        so4_acc_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                      iris.AttributeConstraint(STASH='m01s34i108'), is_lam)
        outputcubedict["SO4_Acc"] = so4_acc_mixing  # kg/kg
    if 'BC_Acc' in varlist:  # missing u-cs093(1211)
        bc_acc_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                     iris.AttributeConstraint(STASH='m01s34i109'), is_lam)
        outputcubedict["BC_Acc"] = bc_acc_mixing  # kg/kg
    if 'OC_Acc' in varlist:  # missing u-cs093(1211)
        oc_acc_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                     iris.AttributeConstraint(STASH='m01s34i110'), is_lam)
        outputcubedict["OC_Acc"] = oc_acc_mixing  # kg/kg
    if 'SS_Acc' in varlist:  # missing u-cs093(1211)
        ss_acc_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                     iris.AttributeConstraint(STASH='m01s34i111'), is_lam)
        outputcubedict["SS_Acc"] = ss_acc_mixing  # kg/kg
    if 'NH4_Acc' in varlist:  # yes u-cs093(1211)
        nh4_acc_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                      iris.AttributeConstraint(STASH='m01s34i134'), is_lam)
        outputcubedict["NH4_Acc"] = nh4_acc_mixing  # kg/kg
    if 'NO3_Acc' in varlist:  # yes u-cs093(1211)
        no3_acc_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                      iris.AttributeConstraint(STASH='m01s34i138'), is_lam)
        outputcubedict["NO3_Acc"] = no3_acc_mixing  # kg/kg

    if 'SO4_Ait' in varlist:  # yes u-cs093(1211)
        so4_ait_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                      iris.AttributeConstraint(STASH='m01s34i104'), is_lam)
        outputcubedict["SO4_Ait"] = so4_ait_mixing
    if 'BC_Ait' in varlist:  # missing u-cs093(1211)
        nolabel = '_pb'
        if is_lam:
            nolabel = '_pe'
        bc_ait_mixing = load_um_cube(timeindexes, surfcoord, prefix + nolabel,
                                     iris.AttributeConstraint(STASH='m01s34i105'), is_lam)
        outputcubedict["BC_Ait"] = bc_ait_mixing  # kg/kg
    if 'OC_Ait' in varlist:  # missing u-cs093(1211)
        nolabel = '_pb'
        if is_lam:
            nolabel = '_pe'
        oc_ait_mixing = load_um_cube(timeindexes, surfcoord, prefix + nolabel,
                                     iris.AttributeConstraint(STASH='m01s34i106'), is_lam)
        outputcubedict["OC_Ait"] = oc_ait_mixing  # unit here is kg/kg
    if 'BC_Ait_ins' in varlist:  # missing u-cs093(1211)
        bc_ait_mixing_ins = load_um_cube(timeindexes, surfcoord, prefix + '_pb',
                                         iris.AttributeConstraint(STASH='m01s34i120'), is_lam)
        outputcubedict["BC_Ait_ins"] = bc_ait_mixing_ins  # kg/kg
    if 'OC_Ait_ins' in varlist:  # missing u-cs093(1211)
        nolabel = '_pd'
        if is_lam:
            nolabel = '_pb'
        oc_ait_mixing_ins = load_um_cube(timeindexes, surfcoord, prefix + nolabel,
                                         iris.AttributeConstraint(STASH='m01s34i121'), is_lam)
        outputcubedict["OC_Ait_ins"] = oc_ait_mixing_ins  # unit here is kg/kg
    if 'NH4_Ait' in varlist:  # yes u-cs093(1211)
        nh4_ait_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                      iris.AttributeConstraint(STASH='m01s34i133'), is_lam)
        outputcubedict["NH4_Ait"] = nh4_ait_mixing
    if 'NO3_Ait' in varlist:  # yes u-cs093(1211)
        no3_ait_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                      iris.AttributeConstraint(STASH='m01s34i137'), is_lam)
        outputcubedict["NO3_Ait"] = no3_ait_mixing

    if 'NO' in varlist:
        nolabel='_pd'
        if is_lam:
            nolabel='_pc'
        no_mixing = load_um_cube(timeindexes,surfcoord,prefix+nolabel,iris.AttributeConstraint(STASH='m01s34i002'),is_lam)
        no_conc = no_mixing*0.029/0.030*1e12
        outputcubedict["NO"]=no_conc
    if 'N_All'in varlist or 'N_Nuc'in varlist or 'N_Ait_In' in varlist or 'N_Ait' in varlist or 'N_Acu' in varlist or 'N_Cor' in varlist \
            or 'D_Nuc' in varlist or 'D_Ait' in varlist or 'D_Acu' in varlist or 'D_Cor' in varlist or 'D_Ait_In' in varlist or ' N_10' in varlist:
        numlabel='_pb' #global model is in pb
        if is_lam:
            numlabel='_pe' #regional model is in pe

        nuc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i101'), is_lam)
        ait_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i103'),is_lam)
        acc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i107'),is_lam)
        cor_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i113'),is_lam)
        aitin_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i119'),is_lam)

        # n10_num_mixing = ait_num_mixing+acc_num_mixing+cor_num_mixing+aitin_num_mixing #>10nm number concentration
        # # n10_num_mixing = n10_num_mixing + lognormal_cumulative_forcubes(nuc_num_mixing, 1.0e-8, ait_diam_dry, 1.59)
        # nall_num_mixing=n10_num_mixing+nuc_num_mixing

        if is_lam:
            # ait_diam = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s38i402'),is_lam)
            nuc_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                    iris.AttributeConstraint(STASH='m01s38i401'), is_lam)
            ait_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                        iris.AttributeConstraint(STASH='m01s38i402'), is_lam)
            acc_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                        iris.AttributeConstraint(STASH='m01s38i403'), is_lam)
            coar_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                         iris.AttributeConstraint(STASH='m01s38i404'), is_lam)
            ait_diam_inso = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                         iris.AttributeConstraint(STASH='m01s38i405'), is_lam)
            # dia_list = [ait_ria_wet, acc_ria_wet, coar_ria_wet,ait_ria_inso]


        else:
            nuc_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                    iris.AttributeConstraint(STASH='m01s38i401'), is_lam)
            ait_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                        iris.AttributeConstraint(STASH='m01s38i402'), is_lam)
            acc_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                        iris.AttributeConstraint(STASH='m01s38i403'), is_lam)
            coar_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                         iris.AttributeConstraint(STASH='m01s38i404'), is_lam)
            ait_diam_inso = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                         iris.AttributeConstraint(STASH='m01s38i405'), is_lam)
        # n10_num_mixing = ait_num_mixing+acc_num_mixing+cor_num_mixing+aitin_num_mixing #>10nm number concentration
        # # n10_num_mixing = n10_num_mixing + lognormal_cumulative_forcubes(nuc_num_mixing, 1.0e-8, ait_diam_dry, 1.59)
        # nall_num_mixing=n10_num_mixing+nuc_num_mixing

        n10_num_mixing = ait_num_mixing + acc_num_mixing + cor_num_mixing + aitin_num_mixing
        nall_num_mixing = n10_num_mixing + nuc_num_mixing
        n10_num_mixing = nall_num_mixing - lognormal_cumulative_forcubes(nuc_num_mixing, 1.0e-8, nuc_diam, 1.59)

        #this n100 might be needed in the future for calculation
        n100_mixing = ait_num_mixing-lognormal_cumulative_forcubes(ait_num_mixing, 1.0e-7,ait_diam,1.59)
        n100_mixing = n100_mixing + aitin_num_mixing - lognormal_cumulative_forcubes(aitin_num_mixing, 1.0e-7,ait_diam_inso,1.59)
        n100_mixing = n100_mixing+acc_num_mixing - lognormal_cumulative_forcubes(acc_num_mixing, 1.0e-7,acc_diam,1.59)
        n100_mixing = n100_mixing+cor_num_mixing
        pref=1.013E5
        tref=293.0 #Assume STP is 20C
        zboltz=1.3807E-23
        staird=pref/(tref*zboltz*1.0E6)
        nuc_num_conc = nuc_num_mixing * staird
        ait_num_conc = ait_num_mixing * staird
        acc_num_conc = acc_num_mixing * staird
        cor_num_conc = cor_num_mixing * staird
        aitin_num_conc = aitin_num_mixing * staird

        n10_num_conc = n10_num_mixing * staird
        nall_num_conc=nall_num_mixing* staird

        # 5 for number concentration
        outputcubedict["N_Nuc"] = nuc_num_conc
        outputcubedict["N_Ait"] = ait_num_conc
        outputcubedict["N_Acu"] = acc_num_conc
        outputcubedict["N_Cor"] = cor_num_conc
        outputcubedict["N_Ait_In"] = aitin_num_conc
        outputcubedict["N_10"] = n10_num_conc
        outputcubedict["N_All"] =nall_num_conc
        outputcubedict["N_100"] = n100_mixing*staird
        # 4 diameter, nucleation mode has not been calculated yet
        outputcubedict["D_Nuc"] = nuc_diam
        outputcubedict["D_Ait"] = ait_diam
        outputcubedict["D_Acu"] = acc_diam
        outputcubedict["D_Cor"] = coar_diam
        outputcubedict["D_Ait_In"] = ait_diam_inso

    if 'CDNC' in varlist:
        if is_lam:
            cdnc_in_cloud = 1e-6*load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s38i479'),is_lam) #num/m3->num/cc
            lwc_prefix='_pb'
        else:
            cdnc_in_cloud = 1e-6*load_um_cube(timeindexes,surfcoord,prefix+'_pc',iris.AttributeConstraint(STASH='m01s34i968'),is_lam) #num/m3->num/cc
            lwc_prefix='_pe'
        outputcubedict["CDNC"]=cdnc_in_cloud

    if 'P' in varlist:
        air_pressure = load_um_cube(timeindexes, surfcoord, prefix + denslabel,
                                    iris.AttributeConstraint(STASH='m01s00i408'), is_lam)
        outputcubedict["P"] = air_pressure #00408 pressure at theta-level


    # start from here CS
    # def ukca_cond_coff_c(tsqrt, mmcg, se, dmol, cc, pmid, t, difvol):  # some of them not used?eg: airdm3/rhoa/nv??
    if 'SUMNC2' in varlist:
        numlabel = '_pb'
        if is_lam:
            numlabel = '_pe'

        ait_num_mixing = load_um_cube(timeindexes, surfcoord, prefix + numlabel, iris.AttributeConstraint(STASH='m01s34i103'), is_lam)
        acc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix + numlabel, iris.AttributeConstraint(STASH='m01s34i107'), is_lam)
        cor_num_mixing = load_um_cube(timeindexes, surfcoord, prefix + numlabel, iris.AttributeConstraint(STASH='m01s34i113'), is_lam)

        pref = 1.013E5
        tref = 293.0  # Assume STP is 20C
        zboltz = 1.3807E-23
        staird = pref / (tref * zboltz * 1.0E6)
        ait_num_conc = ait_num_mixing * staird
        acc_num_conc = acc_num_mixing * staird
        cor_num_conc = cor_num_mixing * staird
        conc_list = [ait_num_conc, acc_num_conc, cor_num_conc] #for further calculation of CS

        if is_lam:
            #ait_diam = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s38i402'),is_lam)
            ait_diam_dry = load_um_cube(timeindexes, surfcoord, prefix+'_pd',iris.AttributeConstraint(STASH='m01s38i402'),is_lam)
            ait_ria_wet = ait_diam_dry * 2 / 2  # a rough approximation from dry->wet by *2
            acc_diam_wet = load_um_cube(timeindexes, surfcoord, prefix+'_pd',iris.AttributeConstraint(STASH='m01s38i403'),is_lam)
            acc_ria_wet = acc_diam_wet / 2  #I am not sure should be 03 or 10?
            coar_diam_dry = load_um_cube(timeindexes, surfcoord, prefix + '_pd', iris.AttributeConstraint(STASH='m01s38i404'), is_lam)
            coar_ria_wet = coar_diam_dry * 2 / 2
            dia_list = [ait_ria_wet, acc_ria_wet, coar_ria_wet]
        else:
            ait_diam_dry = load_um_cube(timeindexes, surfcoord, prefix+'_pe',iris.AttributeConstraint(STASH='m01s38i402'),is_lam)
            ait_ria_wet = ait_diam_dry * 2 / 2  # a rough approximation from dry->wet by *2
            acc_diam_wet = load_um_cube(timeindexes, surfcoord, prefix+'_pe',iris.AttributeConstraint(STASH='m01s38i403'),is_lam)
            acc_ria_wet = acc_diam_wet / 2
            coar_diam_dry = load_um_cube(timeindexes, surfcoord, prefix + '_pe',iris.AttributeConstraint(STASH='m01s38i404'), is_lam)
            coar_ria_wet = coar_diam_dry * 2 / 2
            dia_list = [ait_ria_wet, acc_ria_wet, coar_ria_wet]



        se = 1.0  # Sticking efficiency for soluble modes;0.3 for insolu
        rgas = 287.05  # Dry air gas constant =(Jkg^-1 K^-1)
        rr = 8.314  # Universal gas constant(K mol^-1 K^-1)
        pi = 3.1415927
        zboltz = 1.38064852e-23  # (kg m2 s-2 K-1 molec-1)
        avc = 6.02e23  # Avogadros constant (mol-1)
        mm_da = avc * zboltz / rgas  # constant,molar mass of air (kg mol-1)
        t = 293.15  # assume to be  in model?
        pmid = 101325.0  # corresponding to STP kPa
        tsqrt = np.sqrt(293.15)  # Square-root of mid-level air temperature (K)
        mmcg = 98*1e-3  # molarmas of H2SO4(kg per mole)
        dmol = 4.5e-10  # Molecular diameter of condensable (m)
        difvol = 51.96  # Diffusion volume for H2SO4
        dair = 19.7  # diffusion air molecule

        # sinkarr = 0.0
        # s_cond_s = 0.0  # Condensation sink

        term1 = np.sqrt(8.0 * rr / (pi * mmcg))  # used in calcn of thermal velocity of condensable gas
        zz = mmcg / mm_da
        term2 = 1.0 / (
                pi * np.sqrt(1.0 + zz) * dmol * dmol)  # used in calcn of mfp of condensable gas(s & p, pg 457, eq 8.11)
        term3 = (3.0 / (8.0 * avc * dmol * dmol))  # Molecular diameter of condensable (m)
        term4 = np.sqrt((rgas * mm_da * mm_da / (2.0 * pi)) * ((mmcg + mm_da) / mmcg))
        term5 = term3 * term4  # used in calcnof diffusion coefficient of condensable gas
        term6 = 4.0e6 * pi  # used in calculation of condensation coefficient
        term7 = np.sqrt((1.0 / (mm_da * 1000.0)) + (1.0 / (mmcg * 1000.0)))
        dair = 19.7  # diffusion volume of air molecule(fuller et al, reid et al)
        term8 = (dair ** (1.0 / 3.0) + difvol ** (1.0 / 3.0)) ** 2  # used in new culation of diffusion coefficient

        cc = 0.0  # condensation coeff for cpt onto pt(m3/s)
        vel_cp = term1 * tsqrt  # Calculate diffusion coefficient of condensable gas
        dcoff_cp = (1.0e-7 * (t ** 1.75) * term7) / (
                pmid / 101325.0 * term8)  # Mann[56] when idcmfp==2,pmid=Centre level pressure (Pa)
        mfp_cp = 3.0 * dcoff_cp / vel_cp  # Mann[55] whenidcmfp==2
        kn = []
        fkn = []
        akn = []
        cc = []
        nc = []
        sumnc = 0

        for i in range(3):
            # print(i)
            # sumnc = 0
            mfp_cp_array = np.full(dia_list[i].shape, mfp_cp) #dia_list is called from model above,
            akn_array = np.full(dia_list[i].shape, 1)
            kn.append((np.divide(mfp_cp, dia_list[
                i].data)))  # rp[i]=conc_list[9/10/11] ,rp should be called from load_model line297
            fkn.append(np.divide((1.0 + kn[i]), (1.0 + np.multiply(1.71, kn[i]) + 1.33 * np.multiply(kn[i], kn[
                i]))))  # calc.corr.factor Mann[52]
            akn.append(np.divide(akn_array, (1.0 + 1.33 * np.multiply(kn[i], fkn[i]) * (1.0 / se - 1.0))))  # Mann[53]
            cc.append(term6 * dcoff_cp * np.multiply(np.multiply(dia_list[i].data, fkn[i]),
                                                     akn[i]))  # Calc condensation coefficient Mann[51]
            nc.append(np.multiply(conc_list[i].data, cc[i]))
            # nc[3] = acc_num_conc * cc[3]
            # nc[4] = cor_num_conc * cc[4]
            sumnc = sumnc + nc[i]
            # print(sumnc)
            # print(sumnc.shape)
            sumnc2 = conc_list[0].copy(sumnc)
            # print('sumnc2 value',sumnc2)
            outputcubedict["SUMNC2"]=sumnc2
            # return sumnc2
    #lwc1 = 1e3*load_um_cube(timeindexes_den,surfcoord,prefix+lwc_prefix,iris.AttributeConstraint(STASH='m01s00i254'),is_lam)
    #lwc =lwc1.copy(lwc1.data*aird.data)
    # want number concentration at STP per cc. Good, because I can't easily multiply by density anyway as pe is written every 6 hours. Change this!
    #return iris.cube.CubeList([so2_conc,oh_conc,co_conc,c5h8_conc,mt_conc,no_conc,n10_num_conc,n100_num_conc,cdnc_in_cloud])#,lwc])
    print 'calculating altitude AGL'
    first_cube =  list(outputcubedict.values())[0]
    print('first_cube',first_cube) #3*40*10*9 not correct, should be 11*9
    if first_cube.ndim==4: #if 4 dimension, chose 0th element(time),doesn't matter which time, since agl is the same
        altitude_agl_data = np.zeros(first_cube[0,:,:,:].shape)
    elif first_cube.ndim==3: #if 3 dimension, pick all elements
        altitude_agl_data = np.zeros(first_cube.shape)
    else: #in case some cases crush
        print 'altitude fail in load_model'
        sys.exit()
    # if not is_lam: #if global model, have to choose a regional domain,now(orog)is the surface altitude
    #     orog1 =orog.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1)) #288 line 00i033
    #     orog = orog1.extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
    # #a loop over model vertical levels,orog is 2d cubes,first_cube.coord.points[i,:,:] is a 2-d cube,
    # i is vertical layers.
    ##ASL-surface altutude(m-km)=AGL
    print('orog.ndim', orog) #latitude: 11; longitude: 9
    print('altitude.ndim', first_cube.coord('altitude').points[1,:,:]) #latitude: 10; longitude: 9

    for i in range(0, len(altitude_agl_data[:,0,0])):
        # if is_lam:
        #   orog[:,:] = orog[50:251,50:251]
        # else: # if global model, have to choose a regional domain,now(orog)is the surface altitude
        #   orog1 = orog.extract(iris.Constraint(latitude=lambda cell: latmin - 1.0 < cell < latmax + 1))
        #   orog = orog1.extract(iris.Constraint(longitude=lambda cell: lonmin + 359 < cell < lonmax + 361))
        # #   print('first_cube_altitude_ndim',first_cube.coord('altitude').ndim) #3d
        #   print('first_cube_i(2d)',first_cube.coord('altitude').points[i,:,:])
        # # a loop over model vertical levels,orog is 2d cubes,first_cube.coord.points[i,:,:] is a
        print('orog.ndim',orog) #latitude: 11; longitude: 9)
        altitude_agl_data[i,:,:]=first_cube.coord('altitude').points[i,:,:]-1e-3*orog.data
    if first_cube.ndim==4: # If the 1st element is 4dim, we want the altitude also 4dim, copy AGL several times to match
        altitude_agl_cube = first_cube.copy(np.broadcast_to(altitude_agl_data, first_cube.shape))
    else: #if not, dont have to do anything
        altitude_agl_cube = first_cube.copy(altitude_agl_data)
    outputcubedict['Altitude_AGL'] = 1e3*altitude_agl_cube # convert back to meters from kilometers
    print altitude_agl_cube #now you got AGL with the same coordinate with all other variables H2SO4,NH3 etc
    return outputcubedict

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
    # print 'interp_flight_data',gas_time_a.shape,gas_a.shape,track_time.shape
    if(len(gas_time_a.shape) > 1):
        # print 'strange shape of array'
        gas_time_a = gas_time_a[0]
        gas_a = gas_a[0]
        # print gas_time_a.shape,gas_a.shape
        if(len(gas_a.shape) > 1):
            gas_a = gas_a[0]
            # print gas_time_a.shape,gas_a.shape
    interp_gas = interp1d(gas_time_a,gas_a,kind='nearest', bounds_error=False,fill_value=-9999)
    gas_track = interp_gas(np.asarray(track_time))
    gas_track_a = ma.masked_where(gas_track < l_o_d, gas_track)
    return gas_track_a

def read_flight_data(date):
    flight_path='/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140722_r0_L1.ict'

    tracktime,alt,lat,lon = read_flight_track(flight_path,73, 4,2,3) # checkd-yes
    tracktime,rh,temp,pres2 = read_flight_track(flight_path, 73, 36, 18, 24)  # static pressure+static temp
    # tracktime,concp,concp_rpi,concd_lpc = read_flight_track(flight_path, 73, 33, 34, 32)  # I get the pressure wrong
    tracktime = ma.masked_where(np.asarray(lat) < -90,np.array(tracktime))
    # print 'filter',len(tracktime[~tracktime.mask]),tracktime[~tracktime.mask]
    tracktime2 = ma.masked_where(np.asarray(lon) < -180,np.array(tracktime))
    # print 'filter',len(tracktime[~tracktime.mask])
    tracktime = (tracktime2[~tracktime.mask])
    if len(tracktime.shape) > 1:
        tracktime = tracktime[0]
        # print 'tracktime.shape',tracktime.shape,np.asarray(lon).shape,np.asarray(lat).shape
    lon = np.array(lon)[~tracktime2.mask]
    lat = np.array(lat)[~tracktime2.mask]
    alt = np.array(alt)[~tracktime2.mask]
    temp=np.array(temp)[~tracktime2.mask]
    pres2=np.array(pres2)[~tracktime2.mask]
    rh=np.array(rh)[~tracktime2.mask]

    temptrack = interp_flight_data(tracktime, tracktime, temp, -99)
    alt_track = interp_flight_data(tracktime, tracktime, alt, -99)
    # rh = interp_flight_data(tracktime, tracktime, rh, -99)
    pres2track = interp_flight_data(tracktime, tracktime, pres2, -99)
    pres2track=np.multiply(pres2track,100)
    temptrack = temptrack + 273.15

    # h2so4_path='/jet/home/ding0928/Colorado/Colorado/p-3b/FRAPPE-OH-H2SO4_C130_'+date+'_R0.ict'
    # h2so4time,oh,h2so4,dummy = read_flight_track(h2so4_path,100,3,4,1)
    # ohtrack = interp_flight_data(tracktime,h2so4time,oh, -99)
    # h2so4track = interp_flight_data(tracktime, h2so4time, h2so4, -99)

    # nh3_path='/jet/home/ding0928/Colorado/Colorado/p-3b/FRAPPE-NH3_C130_'+date+'_R0.ict'
    # nh3_time,nh3,dummy,dummy = read_flight_track(nh3_path,37,3,3,3)
    # nh3track = 1000.0*interp_flight_data(tracktime,nh3_time,nh3,-99) # units should be ppt
    #
    # if date=='20140802':
    #     so2_path='/jet/home/ding0928/Colorado/Colorado/p-3b/FRAPPE-GTCIMS_C130_'+date+'_R2.ict'
    #     so2_time,so2,dummy,dummy = read_flight_track(so2_path,39,4,4,4)
    #     so2track = interp_flight_data(tracktime,so2_time,so2,-99) # units should be ppt
    # else:
    #     so2track=np.zeros(len(so2track))

    c5h8_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-PTRTOF-NMHCs_P3B_20140722_R0_L1.ict'
    c5h8time, c5h8, c10h16, nh3 = read_flight_track(c5h8_path, 67, 17, 31, 3) #.checked 0722
    c5h8track = interp_flight_data(tracktime, c5h8time, c5h8, -99) #ppbv
    c10h16track = interp_flight_data(tracktime, c5h8time, c10h16, -99) #ppbv
    nh3track = interp_flight_data(tracktime, c5h8time, nh3, -99) #ppbv

    no_o3_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-NOXYO3_P3B_20140722_R0_L1.ict'
    notime, no, no2, o3 = read_flight_track(no_o3_path, 47, 3, 5, 6) #checked 0722
    notrack = interp_flight_data(tracktime, notime, no, -99)
    no2track = interp_flight_data(tracktime, notime, no2, -99)
    o3track = interp_flight_data(tracktime, notime, o3, -99)


    n0=(pres2track*6.02e23)/(8.314*(temptrack+273.15))
    coef=n0*1e-18
    # ohtrack= ohtrack / coef
    # h2so4track=h2so4track/coef
    # c5h8track=c5h8track/coef
    # c10h16track=c10h16track/coef
    # print('pres2track:',pres2track)
    # print('temptrack:',temptrack)
    # print('pres2',pres2)

    # ho2ro2_path ='/jet/home/ding0928/Colorado/Colorado/p-3b/FRAPPE-HO2RO2_C130_'+date+'_R1.ict'
    # ho2ro2time,ho2ro2,ho2,dummy = read_flight_track(ho2ro2_path,100,5,3,1)
    # ho2ro2track = interp_flight_data(tracktime,ho2ro2time,ho2ro2,-10)
    # ho2track = interp_flight_data(tracktime,ho2ro2time,ho2,-10)

    co_p3b_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-DACOM_P3B_20140722_R1_L1.ict'
    cotime, co, ch4, dummy = read_flight_track(co_p3b_path, 37, 1, 2,
                                                 1)  # checked 0723
    cotrack = interp_flight_data(tracktime, cotime, co, -99)
    ch4track = interp_flight_data(tracktime, cotime, ch4, -99)

    n_nuc_p3b_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE-CNC_P3B_20140722_R0_L1.ict'
    n_nuctime, n3, n10, dummy = read_flight_track(n_nuc_p3b_path, 47, 1, 2,
                                               1)  # new added, 37headerline
    n3track = interp_flight_data(tracktime, n_nuctime, n3, -99)
    n10track = interp_flight_data(tracktime, n_nuctime, n10, -99)
    #nnuctrack=n3track-n10track


    obe_sumnc = merge_obs_method()
    #this .nc is no longer useful for CS calculation I think
    n100_file = Dataset('/jet/home/ding0928/Colorado/EOLdata/dataaw2NKX/RF05Z.20140731.195400_011100.PNI.nc')
    data = (n100_file.variables['CS200_RPI'][:, 0, :])
    time = (n100_file.variables['Time'][:])
    cut_off_size = [0.097, 0.105, 0.113, 0.121, 0.129, 0.145, 0.162, 0.182, 0.202, 0.222, 0.242, 0.262,
                    0.282, 0.302, 0.401, 0.57, 0.656, 0.74, 0.833, 0.917, 1.008, 1.148, 1.319, 1.479,
                    1.636, 1.796, 1.955, 2.184, 2.413, 2.661, 2.991]

    data_time = np.array(time)
    # data_time = np.asarray(data_time).reshape(19021,1)

    data_cs200 = pd.DataFrame(data)
    data_cs200 = np.asarray(data_cs200).reshape(19021, 31)
    data_cs200 = np.transpose(data_cs200)
    n100 = n100_file.variables['CONCP_RPI'][:]
    n100track = interp_flight_data(tracktime, data_time, n100, -99)
    # if subtract directly 6 hours, model would start 66-6=60h, even though observation time-6h is right,
    # but you read in the wrong model time cube.
    #what's right: hamish: don't do any time conversion here, but do it at the last step #973,
    # which gives the right value for both obs+model
    #tracktime=tracktime-21600
    date = '20140722'
    unix_epoch_of_22July2014_0001 = 1405987200
    if date == '20140722':
        trackpdtime = pd.to_datetime(np.asarray(tracktime) + unix_epoch_of_22July2014_0001, unit='s')

    if len(np.array(lat).shape) > 1:
        lat = lat[0]
        lon=lon[0]
    d= {'time':pd.Series(trackpdtime), 'latitude':pd.Series(np.asarray(lat)),'longitude':pd.Series(np.asarray(lon)),
        'altitude':pd.Series(np.asarray(alt_track)),'N_All':pd.Series(np.asarray(n3track)),'N_10':pd.Series(np.asarray(n10track)), 'N_100':pd.Series(np.asarray(n100track)),
        'C5H8':pd.Series(c5h8track),'C10H16':pd.Series(c10h16track),
        'NH3':pd.Series(nh3track),'NO':pd.Series(notrack),
        'NO2':pd.Series(no2track),'O3':pd.Series(o3track), 'T':pd.Series(temptrack),'CO':pd.Series(cotrack),
        'P':pd.Series(pres2track),'SUMNC2':pd.Series(obe_sumnc)}
    df3 = pd.DataFrame(d)
    df2 = df3.resample('3T',on='time').mean()
    df2['N_Nuc'] = df2['N_All']-df2['N_10']
    print df2.head()
    return df2

# also works for 3D cubes with no time coordinate
def do_4d_interpolation(cubelist,flight_coords, is_lam,is_dump):
    cube_keys = list(cubelist.keys())
    all_cubes = [cubelist[key] for key in cube_keys]
    cube = all_cubes[0]
    if is_lam:
        input_projection=cube.coord('grid_longitude').coord_system.as_cartopy_projection()
    else:
        input_projection=cube.coord('longitude').coord_system.as_cartopy_projection()
    #my_interpolating_function = RegularGridInterpolator((cube.coord('grid_longitude').points, cube.coord('grid_latitude').points, cube.coord('level_height').points), cube.data.T) #This is the simple version that does not account for orography
    lats_of_flight = flight_coords['latitude']
    lons_of_flight = flight_coords['longitude']
    alts_of_flight = flight_coords['altitude'] # has to be altitude ASL because model gives altitude ASL
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
            #if is_lam:
            # CONVERT FLIGHT ALTITUDES TO KM
            points.append([times_of_flight[ipoint],rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])
            #else:
            #points.append([times_of_flight[ipoint],rotated_lon,rotated_lat,1e-3*alts_of_flight[ipoint]])
        else:
            #points.append([times_of_flight[ipoint],rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])
            if is_dump:
                points.append([rotated_lon+360.0,rotated_lat,alts_of_flight[ipoint]])
            else:
                points.append([rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])
    cubedimcoords=[]
    cubedatasets=[]
    if cube.ndim==4:
        cubedimcoords.append(cube_times)
        for cube_to_interp in all_cubes:
            cubedatasets.append(np.transpose(cube_to_interp.data,(0,3,2,1)))
            print cube_to_interp
    else:
        for cube_to_interp in all_cubes:
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
    if is_lam:
        interp_method = 'nn'
    else:
        interp_method = 'nn'
    interpolator=_RegularGridInterpolator(cubedimcoords,np.asarray(points).T, hybrid_coord =cube.coord('altitude').points.T,hybrid_dims=cube.coord_dims(cube.coord('altitude')),method=interp_method)
    print 'setup interpolation'
    #interpolated_values=my_interpolating_function(points)
    interp_results={}
    i=0
    for key in cube_keys:
        interp_results[key] = np.asarray(interpolator(cubedatasets[i], fill_value=None))
        i=i+1
    #if is_lam:
    #    print np.asarray(interpolator(dataset_to_interp, fill_value=None)[0]),np.asarray(interpolator(dataset_to_interp, fill_value=None)[1]),np.asarray(interpolator(dataset_to_interp, fill_value=None)[2]),np.max(np.asarray(interpolator(dataset_to_interp, fill_value=None)[2]))
    #    print np.min(dataset_to_interp),np.max(dataset_to_interp)
        #print interpolated_values
    return interp_results

#plt.plot(np.asarray(lat),so2track)


# def colorado_mod_J():
date= '20140722'
varlist = ['C5H8','C10H16','NO','NO2','O3','CO','CH4','NH3','P','T',
            'N_Nuc','N_Ait','N_Acu','N_Ait_In',
            'D_Nuc','D_Ait','D_Acu','D_Cor','D_Ait_In',
           'SUMNC2','N_10','N_All']
# make_dataset=0
is_dump=0
if is_dump:
    varlist=['T']

path_model='/jet/home/ding0928/cylc-run/u-ct504/share/cycle/20140722T0000Z/Regn1/resn_1/RA2M/um/'
path_glm = '/jet/home/ding0928/cylc-run/u-ct504/share/cycle/20140722T0000Z/glm/um/'


#set the is_dump flag; can only use for temperature interpolation
if is_dump==1:
    path_glm = '/ocean/projects/atm200005p/shared/um-dumps/cr466a.da20140716_00'

if date=='20140722':
    timeindexes_glm=['012','015','018']  #0722+12h=0722_12pm
    timeindexes_lam=['012','018'] #0722+12h=0722_12pm
    lonmin=-112
    lonmax=-97        # now ca ntrim down to 11*9
    latmin=35
    latmax=46
elif date=='20140802':
    timeindexes_glm=['114','117','120']
    timeindexes_lam=['114','120']
    lonmin=-110
    lonmax=-100
    latmin=35
    latmax=45
bbox = [lonmin,lonmax,latmin,latmax]
crs_latlon = ccrs.PlateCarree()
#annoyingly only minor speedup
if make_dataset==1:
    df = read_flight_data(date)
    conc_dict_glm = load_model(timeindexes_glm,path_glm,varlist,date,0,is_dump)
    dict_of_interpolated_glm_data= do_4d_interpolation(conc_dict_glm,df, 0, is_dump)
    for	key in dict_of_interpolated_glm_data.keys():
        df['glm_'+str(key)] = dict_of_interpolated_glm_data[key]

    conc_dict = load_model(timeindexes_lam, path_model,varlist,date,1, 0)
    dict_of_interpolated_model_data= do_4d_interpolation(conc_dict,df, 1, 0)
    for key in dict_of_interpolated_model_data.keys():
        df['model_'+str(key)] = dict_of_interpolated_model_data[key]

    df.to_csv('interpolated_0722_lighenting_'+date+'.csv')
else:
    df = pd.read_csv('interpolated_0722_lighenting_'+date+'.csv',index_col=0, parse_dates=True)
    # print df.head()
    conc_dict = load_model(timeindexes_lam, path_model,varlist,date,1,0)
    conc_dict_glm = load_model(timeindexes_glm, path_glm,varlist,date,0,0)

#here is a good place to convert timezone:
#print(df.head())
#print 'changing timezone'
import pytz
mountain = pytz.timezone('US/Mountain') #define UTC
df.index = df.index.tz_localize(pytz.utc)
df.index = df.index.tz_convert(mountain) #from UTC to mountain time
#print df.head()
if 'SO2' in varlist:
    norm_array_so2 = [2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000, 50000,100000]
    plot_comparison(df,df['SO2'],df['model_SO2'],df['glm_SO2'],(conc_dict['SO2'])[1,3,:,:],(conc_dict_glm['SO2'])[1,1,:,:],
                    'SO$_{2}$ (pptv)',2.0,10000,2.0,10000,True,norm_array_so2,'korus_so2_'+date+'_ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['SO2'].coord('level_height'),df['SO2'],df['model_SO2'],df['glm_SO2'],2500, 'SO$_{2}$ (pptv)','korus_so2_vp_'+date+'ion_npf_cs093.png.png')

    print('time_regional_index',conc_dict['SO2'].coord('time').points)
    print('time_global_index',conc_dict_glm['SO2'].coord('time').points)

if 'OH' in varlist:
    norm_array_oh=[0,0.1,0.2,0.3,0.4,0.5]
    plot_comparison(df,df['OH'],df['model_OH'],df['glm_OH'],(conc_dict['OH'])[1,3,:,:],(conc_dict_glm['OH'])[1,1,:,:],
                    'OH (pptv)', 0,1,0,1,False,norm_array_oh,'korus_oh_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['OH'].coord('level_height'),df['OH'],df['model_OH'],df['glm_OH'],0.8, 'OH (pptv)','korus_oh_vp_'+date+'ion_npf_cs093.png')

if 'T' in varlist:
    norm_array_temp=[270,275,280,285,290,295,300,305,310]
    print 'MINIMUM INTERPOLATED TEMP:',np.min(np.asarray(df['glm_T']))
    if is_dump:
        plot_comparison(df,df['T'],df['model_T'],df['glm_T'],(conc_dict['T'])[1,3,:,:],(conc_dict_glm['T'])[1,:,:],
                        'T (K)', 260,310,280,310,False,norm_array_temp,'korus_T_'+date+'ion_npf_cs093.png', date)
    else:
        plot_comparison(df,df['T'],df['model_T'],df['glm_T'],(conc_dict['T'])[1,3,:,:],(conc_dict_glm['T'])[1,1,:,:],
                        'T (K)', 260,310,280,310,False,norm_array_temp,'korus_T_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['T'].coord('level_height'),df['T'],df['model_T'],df['glm_T'],310, 'T (K)','korus_T_vp_'+date+'ion_npf_cs093.png',1,270)

if 'H2SO4' in varlist:
    norm_array_h2so4=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0]
    plot_comparison(df,df['H2SO4'],df['model_H2SO4'],df['glm_H2SO4'],(conc_dict['H2SO4'])[1,3,:,:],(conc_dict_glm['H2SO4'])[1,1,:,:],
                    'H2SO4 (pptv)', 0,1.5,0,1.0,False,norm_array_h2so4,'korus_h2so4_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['H2SO4'].coord('level_height'),df['H2SO4'],df['model_H2SO4'],df['glm_H2SO4'],1.0, 'H2SO4 (pptv)','korus_h2so4_vp_'+date+'ion_npf_cs093.png')
if 'NH3' in varlist:
    norm_array_nh3=[0.2,0.3,0.5,0.9,1.5,2,3,5,7,10,15,20]
    plot_comparison(df,df['NH3'],df['model_NH3'],df['glm_NH3'],(conc_dict['NH3'])[1,3,:,:],(conc_dict_glm['NH3'])[1,1,:,:],
                    'NH3 (ppbv)', 0.2,20,0.2,20,True,norm_array_nh3,'korus_nh3_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['NH3'].coord('level_height'),df['NH3'],df['model_NH3'],df['glm_NH3'],20, 'NH3 (pptv)','korus_nh3_vp_'+date+'ion_npf_cs093.png')

if 'C5H8' in varlist:
    norm_array_c5h8=[1.0,3.0,10,30,100,300,1000,1500]
    plot_comparison(df,df['C5H8'],df['model_C5H8'],df['glm_C5H8'],(conc_dict['C5H8'])[1,3,:,:],(conc_dict_glm['C5H8'])[1,1,:,:],
                    'C5H8 (pptv)',1,1500,1,1500.0,False,norm_array_c5h8,'korus_c5h8_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['C5H8'].coord('level_height'),df['C5H8'],df['model_C5H8'],df['glm_C5H8'],1500, 'C5H8 (pptv)','korus_c5h8_vp_'+date+'ion_npf_cs093.png')

if 'NO' in varlist:
    norm_array_no = [5,50,100,300,700,1000,1500,2500,5500,9000]
    plot_comparison(df,df['NO'],df['model_NO'],df['glm_NO'],(conc_dict['NO'])[1,3,:,:],(conc_dict_glm['NO'])[1,1,:,:],
                    'NO (pptv)', 5,9000,20,9000,True,norm_array_no,'korus_no_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['NO'].coord('level_height'),df['NO'],df['model_NO'],df['glm_NO'],9000, 'NO (pptv)','korus_no_vp_'+date+'ion_npf_cs093.png')

if 'O3' in varlist:
    norm_array_o3=[0,0.1,0.2,0.3,0.4,0.5,1,10,20,50,70,80,100]
    plot_comparison(df,df['O3'],df['model_O3'],df['glm_O3'],(conc_dict['O3'])[1,3,:,:],(conc_dict_glm['O3'])[1,1,:,:],'O3 (ppbv)', 30,90,30,90,False,norm_array_o3,'korus_O3_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['O3'].coord('level_height'),df['O3'],df['model_O3'],df['glm_O3'],90, 'O3 (ppbv)','korus_O3_vp_'+date+'ion_npf_cs093.png')

# if 'CO' in varlist:
#     norm_array_co = [50,80,100,120,150,180]
#     plot_comparison(df,df['CO'],df['model_CO'],df['glm_CO'],(conc_dict['CO'])[1,3,:,:],(conc_dict_glm['CO'])[1,1,:,:],
#                     'CO (ppbv)', 50,180,50,180,True,norm_array_co,'korus_co_'+date+'_bt116_ligten.png', date)
#     plot_vertical(df,conc_dict['CO'].coord('level_height'),df['CO'],df['model_CO'],df['glm_CO'],200, 'CO (ppbv)','korus_co_vp_'+date+'_bt116_ligten.png')
if 'P' in varlist:
    norm_array_p=[72000,74000,76000,78000,80000,82000,84000,86000]
    plot_comparison(df,df['P'],df['model_P'],df['glm_P'],(conc_dict['P'])[1,3,:,:],(conc_dict_glm['P'])[1,1,:,:],
                    'P (pa)', 7e4,9e4,7e4,9e4,False,norm_array_p,'korus_p_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['P'].coord('level_height'),df['P'],df['model_P'],df['glm_P'],1e5, 'P (pa)','korus_p_vp_'+date+'ion_npf_cs093.png')
if 'SUMNC2' in varlist:
    norm_array_sumnc2 = [0,0.00001,0.0001,0.001,0.005,0.008,0.01,0.03,0.05]
    plot_comparison(df,df['SUMNC2'],df['model_SUMNC2'],df['glm_SUMNC2'],(conc_dict['SUMNC2'])[1,3,:,:],(conc_dict_glm['SUMNC2'])[1,1,:,:],
                    'Cond sink (s-1)', 0.001,0.05,0.001,0.05,True,norm_array_sumnc2,'korus_sumnc2_'+date+'ion_npf_cs093.png', date)

if 'N_Nuc' in varlist:
    norm_array_nuc=[100,300,500,800,1000,2000,3000,4000,5000,6000,7000,10000]
    plot_comparison(df,df['N_Nuc'],df['model_N_Nuc'],df['glm_N_Nuc'],(conc_dict['N_Nuc'])[1,3,:,:],(conc_dict_glm['N_Nuc'])[1,1,:,:],
                    'N$_{nuc(3-10nm)}$\n(#cm$^{-3}$)', 100,3e4,100,3e4,True,norm_array_nuc,'korus_nnuc_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['N_Nuc'].coord('level_height'),df['N_Nuc'],df['model_N_Nuc'],df['glm_N_Nuc'],3e3, 'N_Nuc (#cm-3)','qual_fig.pdf')


if 'N_10' in varlist:
    norm_array_n10 = [500,1000,1500,2500,3800,5000,8000,10000,12000,15000,18000,20000,25000]
    plot_comparison(df,df['N_10'],df['model_N_10'],df['glm_N_10'],(conc_dict['N_10'])[1,3,:,:],(conc_dict_glm['N_10'])[1,1,:,:],
                    'N(>10nm) (# cm-3)', 500,2.5e4,500,2.5e4,True,norm_array_n10,'korus_n10'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['N_10'].coord('level_height'),df['N_10'],df['model_N_10'],df['glm_N_10'],2.0e4, 'N(10nm)','korus_N(10nm)_vp_'+date+'ion_npf_cs093.png')

if 'N_All' in varlist:
    norm_array_n10 = [500,1000,1500,2500,3800,5000,8000,10000,12000,15000,18000,20000,25000,30000,35000]
    plot_comparison(df,df['N_All'],df['model_N_All'],df['glm_N_All'],(conc_dict['N_All'])[1,3,:,:],(conc_dict_glm['N_All'])[1,1,:,:],
                    'N$_{all(>3nm)}$\n(#cm$^{-3}$)', 500,3.5e4,500,3.5e4,True,norm_array_n10,'korus_nall'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['N_All'].coord('level_height'),df['N_All'],df['model_N_All'],df['glm_N_All'],2.5e4, 'N_All(3nm)','korus_Nall(3nm)_vp_'+date+'ion_npf_cs093.png')
if 'N_100' in varlist:                                                                                                                                               
     # x=range(1,100)                                                                                                                                                
    norm_array_num= [50,100,200,500,1000,2000,5000,10000]                                                                                                           
    plot_comparison(df,df['N_100'],df['model_N_100'],df['glm_N_100'],(conc_dict['N_100'])[1,3,:,:],(conc_dict_glm['N_100'])[1,1,:,:],                                    
                    'N >100nm (cm$^{-3}$ stp)', 10,2000,10,2000.0,True,norm_array_num,'korus_n100_'+date+'ion_npf_cs093.png', date)
    plot_vertical(df,conc_dict['N_100'].coord('level_height'),df['N_100'],df['model_N_100'],df['glm_N_100'],5000, 'N_100 (cm-3 stp)','korus_n100_vp_'+date+'ion_npf_cs093.png')
#     # norm_array_cdnc = [0,100,200,300,400,500,600,700,800,900,1000]   

plt.show()

