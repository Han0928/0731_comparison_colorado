import numpy as np
import pandas as pd
import iris,sys,glob
import imageio
import netCDF4
import matplotlib.pyplot as plt
from pandas import Series
from datetime import datetime
import matplotlib.colors as cols
import matplotlib.cm as cmx
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import LineCollection
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.img_tiles import OSM
import os, sys
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
import scipy as sp
import iris.quickplot as qplt
import iris.plot as iplt
import numpy.ma as ma
import cartopy.feature as cfeature
from iris.util import unify_time_units
import matplotlib.patches as patches
import iris.coord_systems as cs

import shapefile      # for the city boundaries
from matplotlib.offsetbox import AnchoredText
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from matplotlib.colors import LogNorm
from obser_for_Merge_0731_C130 import merge_obs_method
import matplotlib as mpl
import datetime
import time

from pytz import timezone
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RegularGridInterpolator

# Figure 1 is for the MODIS data track  
os.environ["CARTOPY_USER_BACKGROUNDS"] = "/Users/hamish/scripts/cartopy/BG/"
cnindex = 0 #column 1
plt.figure(figsize=(18, 28)) 
plt.rcParams['font.size'] = 22 

def read_flight_track(day, month, filename, file_head_number, campaign, label_on_plot,num_subplot):
    if day < 10:
        dateiso = '2014' + '-0' + str(month) + '-0' + str(day)
    else:  # day31
        dateiso = '2014' + '-0' + str(month) + '-' + str(day)
    headerlen = file_head_number
    year = 2014

    cn_timeseries = []
    alt_timeseries = []
    lat_timeseries = []
    lon_timeseries = []
    utctime = []  # seconds from midnight on July 29 2014
    utctime2 = []
    iline = 0
    with open(filename) as fp:
        line = fp.readline()
        while line:
            if iline < headerlen:  # length of header
                line = fp.readline()
                iline += 1
                continue
            line = fp.readline()
            data = line.split(',')
            if campaign == 0 and iline == 180:
                global cnindex
                iname = 0
                for name in data:
                    if name == 'CONCN':
                        cnindex = iname
                        break
                    iname += 1
            else:
                try:
                    utctime.append(float(data[0])) #column 0 is the UTC time.
                except Exception:
                    break
                if campaign == 0:
                    cn_timeseries.append(float(data[cnindex]))
                    alt_timeseries.append(float(data[4]))
                    if num_subplot==1:
                        lat_timeseries.append(float(data[2]))
                        lon_timeseries.append(float(data[3]))
                    else:    
                        lat_timeseries.append(float(data[5]))
                        lon_timeseries.append(float(data[6]))
            if iline % 500 == 0:
                print('i=1')
            iline += 1

    hours = [int((ut / 3600) - 6) for ut in utctime]
    hours2 = [int((ut / 3600) - 6) for ut in utctime2]
    days2 = [day for hour in hours2]
    months2 = [month for hour in hours2]
    days = [day for hour in hours]
    months = [month for hour in hours]
    i = 0
    # Fixing the times. Not 100% sure this is correct, we should look at it more carefully if you use it.
    if hours[0] > 20:
        print(dateiso)
        dateiso2 = list(dateiso)
        if int(dateiso2[-1]) == 9:
            dateiso2[-2] = str(int(dateiso2[-2]) + 1)
            dateiso2[-1] = str(0)
        else:
            dateiso2[-1] = str(int(dateiso2[-1]) + 1)
        dateiso = "".join(dateiso2)
        print(dateiso)
    for hour in hours:
        if hour > 23:
            hours[i] = hours[i] - 24
            if days[i] == 31:
                days[i] = 1
                months[i] = months[i] + 1
            else:
                days[i] = days[i] + 1
        i = i + 1
    minutes = [int((ut - 3600 * int(ut / 3600)) / 60) for ut in utctime]
    minutes2 = [int((ut - 3600 * int(ut / 3600)) / 60) for ut in utctime2]
    print(minutes[0:10], minutes[len(utctime) - 1])
    seconds = [int(ut - 3600 * int(ut / 3600) - 60 * int((ut - 3600 * int(ut / 3600)) / 60)) for ut in utctime]
    seconds2 = [int(ut - 3600 * int(ut / 3600) - 60 * int((ut - 3600 * int(ut / 3600)) / 60)) for ut in utctime2]
    print(utctime[0], utctime[len(utctime) - 1], len(utctime))

    # Now make a plot.
    plt.figure(figsize=(18, 28))   
    ax = plt.subplot(3, 2, num_subplot, projection=ccrs.PlateCarree())
    # plt.text(-104.99, 39.48, '*Denver', fontsize=22, color='red')
    plt.text(-105.00, 40.03, '*BAO Tower', fontsize=22, color='red')
    # plt.text(-104.82, 41.14, '*Cheyenne', fontsize=22, color='red')  
    # plt.text(-106.50, 41.58, 'Colorado Region', fontsize=24, color='Red')
    ax.coastlines(resolution='50m', linewidth=2)
    if num_subplot==1:
        ax.text(0.2, 1.10, '(a) P-3B Airplane track', transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')
    else:
        ax.text(0.2, 1.10, '(b) C-130 Airplane track', transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')

    # To get the MODIS data
    ax.add_wmts('https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
                'MODIS_Aqua_CorrectedReflectance_TrueColor', wmts_kwargs={'time': dateiso})
    cmap = plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    norm = BoundaryNorm([500, 1.5E3, 2.5e3, 3e3, 3.5e3, 4e3, 5e3], cmap.N, clip=False)
    # The flight track
    points = np.array([lon_timeseries, lat_timeseries]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles='solid')
    lc.set_array(np.asarray(alt_timeseries))
    lc.set_linewidth(4)
    ax.add_collection(lc)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)

    # Timing labels
    if label_on_plot == 1:
        for itime in range(0, len(hours)):
            if minutes[itime] == 0 and seconds[itime] == 0:
                plt.plot(lon_timeseries[itime], lat_timeseries[itime], marker='|', markersize=0, color='red')
                if hours[itime] < 10:
                    timestring = '0' + str(hours[itime]) + ':00'
                else:
                    timestring = str(hours[itime]) + ':00'
                if hours[itime] != 4:
                    plt.text(lon_timeseries[itime], lat_timeseries[itime], timestring, fontsize=19, color='yellow')
    ax.coastlines(resolution='50m', linewidth=2)
    # Grid-lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 22}
    gl.ylabel_style = {'size': 22}
    cb = plt.colorbar(lc, label='Aircraft height (m)', extend="max", shrink=0.7)
    cb.set_label(label='Aircraft height (m)', fontsize=22)
    cb.ax.tick_params(labelsize=22)

## plot for 4 other spatial maps
def horizontalplot(ax, cube,minv,maxv):
    pl =  iplt.pcolormesh(cube,vmin=minv,vmax=maxv)
    pl.cmap.set_under('k')
    plt.gca().stock_img() 
    plt.gca().coastlines(resolution='50m',linewidth=2)
    ax.add_feature(cfeature.LAND) 
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    plt.text(-104.99, 39.48, '*Denver', fontsize=22, color='red')
    plt.text(-105.00, 40.03, '*BAO Tower', fontsize=22, color='red')
    plt.text(-104.82, 41.14, '*Cheyenne', fontsize=22, color='red')  
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
#plot the comparison figure 
def plot_contour(lam2d, ylabel, map_min, map_max, date, saving_name, i):
    cmap = plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    plt.figure(figsize=(18, 28))   
    ax = plt.subplot(323, projection=ccrs.PlateCarree())
    if saving_name =='Rgn H2SO4':  
        ax.text(0.3, 1.10, '(c)'+saving_name, transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')
    elif saving_name =='Rgn NH3':
        ax.text(0.3, 1.10, '(d)'+saving_name, transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')
    elif saving_name =='Rgn N_nuc':
       ax.text(0.3, 1.10, '(e)'+saving_name, transform=ax.transAxes,fontsize=22, fontweight='bold', va='top') 
    else:       
        ax.text(0.3, 1.10, '(f)'+saving_name, transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')

    horizontalplot(ax, lam2d, map_min, map_max)
    cb = plt.colorbar(label=ylabel,extend="max", shrink=0.7)
    cb.ax.tick_params(labelsize=22)
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    cb.ax.yaxis.set_major_formatter(formatter)
    # Read in the shapefile
    shapefile_path = '/jet/home/ding0928/shapefile_denver/ne_10m_admin_1_states_provinces'
    shapes = shpreader.Reader(shapefile_path).geometries()
    for shape in shapes:
        ax.add_geometries([shape], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1)
    plt.text(-106.50, 41.58, '- -city boundary', fontsize=18, color='black')
    
    plt.savefig('/jet/home/ding0928/python_analysis/fig/'+saving_name + '_' + str(i) + '.png', dpi=100)

def add_orog(cube,surfcoord, dims):
    cube.add_aux_coord(surfcoord,dims)
    cube.coord('level_height').convert_units('km')
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
    cube.add_aux_coord(longit, (2,3))
    cube.add_aux_coord(latit, (2,3))
    return bbox_extract_2Dcoords(cube, bbox)

def load_um_cube(timeindexes,surfcoord, inputfile,cubename):
    cubelist = iris.cube.CubeList()
    for time_index in timeindexes:
        try:
            cubelist.append(iris.load_cube(inputfile+time_index,cubename))
        except Exception:
            cubelist.append(iris.load(inputfile+time_index,cubename)[0])
    print('cubelist[0].ndim',cubelist[0].ndim)
    if cubelist[0].ndim==4: #several time slots=4d dimension
        cube = cubelist.concatenate_cube()
        print('cube_4d',cube) #just need to connect
    elif cubelist[0].ndim==3: #some cube only has 1time slot=3d dimension,
        cube = cubelist.merge_cube() #need to make a list from 0,so called merge
        print('cube_3d', cube)
    add_orog(cube,surfcoord, (2,3,))
    add_lat_lon(cube,bbox)
    return cube

def load_model(timeindexes,path,varlist,date):
    outputcubedict={}
    prefix = path+'umnsaa'

    orog = iris.load_cube(prefix+'_pa000',iris.AttributeConstraint(STASH='m01s00i033'))
    print('before trimming orog[:,:]',orog[:,:])
    orog = orog[50:251,50:251]
    surfcoord = iris.coords.AuxCoord(1e-3*orog.data,'surface_altitude', units='km')
    print('surfcoord with trimmed orog.data', surfcoord) #2d standard_name='surface_altitude', units=Unit('km')

    if 'SO2' in varlist:
        so2_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_sulfur_dioxide_in_air')
        so2_conc = so2_mixing*0.029/0.064*1e12 #pptv
        outputcubedict["SO2"] = so2_conc
    if 'H2SO4' in varlist:
        h2so4_mixing = load_um_cube(timeindexes,surfcoord,prefix + '_pc', iris.AttributeConstraint(STASH='m01s34i073'))
        h2so4_conc = h2so4_mixing * 0.029 / 0.098 * 1e12  # pptv,glm in pc,
        outputcubedict["H2SO4"]=h2so4_conc
    if 'NH3' in varlist: #m01s34i076
        nh3_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_ammonia_in_air')
        nh3_conc = nh3_mixing*0.029/0.017*1e9 #ppbv
        outputcubedict["NH3"]=nh3_conc

    if 'N_All'or 'N_nuc'in varlist:
        numlabel='_pe' #regional model is in pe
        nuc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i101'))
        ait_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i103'))
        acc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i107'))
        cor_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i113'))
        aitin_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i119'))
        n10_num_mixing = ait_num_mixing + acc_num_mixing + cor_num_mixing + aitin_num_mixing
        nall_num_mixing = n10_num_mixing + nuc_num_mixing

        pref=1.013E5
        tref=293.0 #Assume STP is 20C
        zboltz=1.3807E-23
        staird=pref/(tref*zboltz*1.0E6)
        nuc_num_conc = nuc_num_mixing * staird
        nall_num_conc= nall_num_mixing* staird
        outputcubedict["N_All"] =nall_num_conc
        outputcubedict["N_nuc"] = nuc_num_conc
        
    first_cube =  list(outputcubedict.values())[0]
    if first_cube.ndim==4: #if 4 dimension, chose 0th element(time),doesn't matter which time, since agl is the same
        altitude_agl_data = np.zeros(first_cube[0,:,:,:].shape)
    elif first_cube.ndim==3: #if 3 dimension, pick all elements
        altitude_agl_data = np.zeros(first_cube.shape)
    else: #in case some cases crush
        print('altitude fail in load_model')
        sys.exit()

    ##ASL-surface altutude(m-km)=AGL
    for i in range(0, len(altitude_agl_data[:,0,0])):
        print('orog.ndim',orog) #latitude: 11; longitude: 9)
        altitude_agl_data[i,:,:]=first_cube.coord('altitude').points[i,:,:]-1e-3*orog.data
    if first_cube.ndim==4: # If the 1st element is 4dim, we want the altitude also 4dim, copy AGL several times to match
        altitude_agl_cube = first_cube.copy(np.broadcast_to(altitude_agl_data, first_cube.shape))
    else: #if not, dont have to do anything
        altitude_agl_cube = first_cube.copy(altitude_agl_data)
    outputcubedict['Altitude_AGL'] = 1e3*altitude_agl_cube # convert back to meters from kilometers
    return outputcubedict

def plot_topo(path,num_subplot):
    prefix = path+'umnsaa'
    orog = iris.load_cube(prefix+'_pa000',iris.AttributeConstraint(STASH='m01s00i033'))

    plt.figure(figsize=(18, 28)) 
    ax = plt.subplot(3, 2, num_subplot, projection=ccrs.PlateCarree())
    pl =  iplt.pcolormesh(orog,vmin=500,vmax=5500)
    pl.cmap.set_under('k')
    plt.gca().stock_img() 
    plt.gca().coastlines(resolution='50m',linewidth=2) 
    plt.text(-104.99, 39.48, '*Denver', fontsize=22, color='red')
    plt.text(-105.00, 40.03, '*BAO Tower', fontsize=22, color='red')
    plt.text(-104.82, 41.14, '*Cheyenne', fontsize=22, color='red')  
    plt.text(-106.50, 41.58, 'Colorado Region', fontsize=24, color='Red')
    ax.coastlines(resolution='50m', linewidth=2)
    ax.text(0.2, 1.10, '(g)Topography(m)', transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')
    cmap = plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    norm = BoundaryNorm([500, 1.5E3, 2.5e3, 3e3, 3.5e3, 4e3, 5e3], cmap.N, clip=False)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 22}
    gl.ylabel_style = {'size': 22}
    cb = plt.colorbar(label='Topography (m)', extend="max", shrink=0.7)
    cb.set_label(label='Topography (m)', fontsize=22)
    cb.ax.tick_params(labelsize=22)

# path_model='/jet/home/ding0928/cylc-run/u-cs093/share/cycle/20140730T0000Z/Regn1/resn_1/RA2M/um/'
path_model='/jet/home/ding0928/cylc-run/u-ct706/share/cycle/20140730T0000Z/Regn1/resn_1/RA2M/um/'
date= '20140731'
varlist = ['NH3','H2SO4','N_All','N_nuc']

timeindexes_lam=['030','036','042','048'] #0730+24h+18h=0731_18pm(5th); 
lonmin=-106.5
lonmax=-103.5   
latmin=39.0
latmax=41.8

bbox = [lonmin,lonmax,latmin,latmax]
crs_latlon = ccrs.PlateCarree()

conc_dict = load_model(timeindexes_lam, path_model,varlist,date)

read_flight_track(31, 7, '/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/discoveraq-reveal_p3b_20140731_r0.ict',73,0,1,1) #campnane0=p3b
read_flight_track(31, 7, '/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/FRAPPE-NCAR-LRT-NAV_C130_20140731_R4.ict',132,0,1,2) #campnane1=C130
for i in range(8): 
    plot_contour((conc_dict['H2SO4'])[i,1,:,:],'H2SO4 (pptv)', 0.1,2,date, 'Rgn H2SO4', i)
    plot_contour((conc_dict['NH3'])[i,1,:,:],'NH3 (ppbv)', 0.2,5,date,'Rgn NH3', i)
    plot_contour((conc_dict['N_All'])[i,1,:,:],'N_all(#/cc)', 1e3,4e4,date,'Rgn N_all', i)
    plot_contour((conc_dict['N_nuc'])[i,1,:,:],'N_nuc(#/cc)', 1e1,4e3,date,'Rgn N_nuc', i)

plot_topo(path_model,1)

# Gif
species_list = ['H2SO4', 'NH3', 'N_all','N_nuc']
file_names = []
folder_path = '/jet/home/ding0928/python_analysis/fig/'
for species in species_list:
    for i in range(8):
        # Construct the absolute path of the PNG file using os.path.join
        file_path = os.path.join(folder_path, 'Rgn '+ species + '_' + str(i) + '.png')
        file_names.append(file_path)

### try a different version
from PIL import Image, ImageDraw, ImageFont
# Create GIF for each species
for species in species_list:
    with imageio.get_writer(species + '.gif', mode='I') as writer:
        for i in range(8):
            # Use the absolute path of the PNG file
            file_path = os.path.join(folder_path, 'Rgn ' + species + '_' + str(i) + '.png')          
            # Open the image file using PIL
            image = Image.open(file_path)           
            # Create a text overlay indicating the time
            draw = ImageDraw.Draw(image)
            # font = ImageFont.truetype('arial.ttf', 30)
            time_text = 'Time: ' + str(i) + ' min'
            draw.text((10, 10), time_text, fill=(255, 255, 255))       
            # Save the modified image as a temporary file
            temp_file_path = 'temp.png'
            image.save(temp_file_path)           
            # Read the modified image using imageio
            image = imageio.imread(temp_file_path)            
            # Append the image to the GIF
            writer.append_data(image)            
            # Delete the temporary file
            os.remove(temp_file_path)

plt.show()