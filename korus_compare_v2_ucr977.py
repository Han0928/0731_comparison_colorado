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
from scipy.stats import lognorm,pearsonr
#from cis.data_io.ungridded_data import UngriddedData
#from cis.data_io.hyperpoint import HyperPoint
import pandas as pd
# from cmcrameri import cm
#flags for whether to make and write intermediate files or use them
new_diam_calc=1
make_dataset=0
sizedistrosOnly=1

loncor=0

sizebins_p3b = [10.0, 11.2, 12.6, 14.1, 15.8, 17.8, 20, 22.4, 25.1, 28.2, 31.6, 35.5, 39.8, 44.7, 50.1, 56.2, 63.1, 70.8, 79.4,
         89.1, 100, 112.2, 125.9, 141.3, 158.5, 177.8, 199.5, 223.9, 251.2, 281.8]
las_sizes_p3b = [282,316,355,398,447,501,562,631,708,794,891,1000,1259,1585,1995,2512,3162,3981,5012]
##plotting routines

def print_metrics(obs_data,model_data):
    obs_array = np.asarray(obs_data)
    model_array = np.asarray(model_data)
    nas = np.logical_or(np.isnan(obs_array), np.isnan(model_array))
    #print 'Pearsons R',pearsonr(obs_array[~nas],model_array[~nas])[0]
    bias_denom= np.sum(obs_array[~nas])
    bias_num = np.sum(model_array[~nas]-obs_array[~nas])
    #print 'nmb', bias_num/bias_denom
    return pearsonr(obs_array[~nas],model_array[~nas])[0],bias_num/bias_denom
def add_times_to_plot(df,date): #the same with other long python script
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
    lc.set_array(np.asarray(df.altitude))
    lc.set_linewidth(2)
    ax.add_collection(lc)
    #add_times_to_plot(dataframe,date)
def plot_wind(uwind,v2wind):
    vwind = v2wind[:-1,:].copy()
    vwind.coord('grid_longitude').points = uwind.coord('grid_longitude').points
    vwind.coord('grid_latitude').points = uwind.coord('grid_latitude').points
    ulon = uwind.coord('grid_longitude')
    vlon = vwind.coord('grid_longitude')
    x = vlon.points[::12]
    y = vwind.coord('grid_latitude').points[::12]
    v = vwind.data[::12, ::12]
    u = uwind.data[::12, ::12]
    transform = ulon.coord_system.as_cartopy_projection()
    plt.quiver(x, y, u[:,:], v[:,:], pivot='middle', transform=transform, width=0.01)
def horizontalplot(ax, cube,minv,maxv,norm):
    if norm !=None:
        pl =  iplt.pcolormesh(cube,vmin=minv,vmax=maxv,norm=norm)
    else:
        pl =  iplt.pcolormesh(cube,vmin=minv,vmax=maxv)
    #print('made pcolor')
    # pl.set_cmap(cm.batlow)
    pl.cmap.set_under('k')
    #plt.gca().stock_img()
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
def plot_comparison(df,col_label,lam2d,glm2d,ylabel,var_min,var_max,map_min,map_max,is_log,norm_array,saving_name,date,two_models,df2,plot_obs=1):
    plt.figure(figsize=(8,4.8))
    plt.subplot(212)
    if plot_obs==1:
        df[col_label].plot(logy=is_log,ylim=(var_min,var_max),label='Obs.', color='black',linewidth=1.5)
    if two_models==0:
        df['model_'+col_label].plot(logy=is_log,ylim=(var_min,var_max),label='Regn.',color='blue' )
    else:
        df['model_'+col_label].plot(logy=is_log,ylim=(var_min,var_max),label='KORUSv5',color='blue' )
        df2['model_'+col_label].plot(logy=is_log,ylim=(var_min,var_max),label='CMIP6',color='green' )
    ax2 = df['glm_'+col_label].plot(logy=is_log,ylim=(var_min,var_max),label='Global',color='orange')
    ax2.set_ylabel(ylabel)
    ax3 = df['altitude_agl'].plot(secondary_y=True,label='Altitude',color='lightcoral')
    ax3.set_ylabel('Altitude AGL (m)')
    ax3.set_ylim(0,10000)

    h2,l2 = ax2.get_legend_handles_labels()
    h3,l3 = ax3.get_legend_handles_labels()
    #plt.legend(h2+h3,l2+l3,bbox_to_anchor=(1.0, 1.1),ncol=2)
    plt.legend(h2+h3,l2+l3,ncol=2)
    ax = plt.subplot(222, projection=ccrs.PlateCarree())
    if is_log:
        horizontalplot(ax, glm2d, map_min, map_max, LogNorm())
    else:
        horizontalplot(ax, glm2d, map_min, map_max, None)
    plt.colorbar(label=ylabel)
    cmap=plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    norm = BoundaryNorm(norm_array, cmap.N,clip=False)
    add_flight_track_to_latlon(ax, norm,cmap,df[col_label],df,date)

    ax = plt.subplot(221, projection=ccrs.PlateCarree())
    #print('made subplots')
    if is_log:
        horizontalplot(ax, lam2d, map_min, map_max, LogNorm())
    else:
        horizontalplot(ax, lam2d, map_min, map_max,None)
    lat_constraint = iris.Constraint(latitude=lambda cell: latmin < cell < latmax)                                                            
    lon_constraint = iris.Constraint(longitude=lambda cell: lonmin < cell < lonmax)                                                           
    small_glm_cube = glm2d.extract(lat_constraint & lon_constraint)       
    reg_mean = lam2d.collapsed(['grid_latitude','grid_longitude'],iris.analysis.MEAN).data
    glob_mean=small_glm_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN).data
    print ylabel+' mean across domain,reg,glob =',reg_mean,glob_mean
    plt.colorbar(label=ylabel)
    #print('plot_comparison, adding flight track')
    cmap=plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    add_flight_track_to_latlon(ax, norm,cmap,df[col_label],df,date)
    
    plt.tight_layout()
    plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf/0731_comparison/'+saving_name)
    #print ylabel,"KORUSv5,CMIP6,glm"
    rk,nmbk = print_metrics(df.resample('10T').mean()[col_label],df.resample('10T').mean()['model_'+col_label])
    rg,nmbg = print_metrics(df.resample('10T').mean()[col_label],df.resample('10T').mean()['glm_'+col_label])
    if two_models==1:
        rc,nmbc = print_metrics(df2.resample('10T').mean()[col_label],df2.resample('10T').mean()['model_'+col_label])
        print ylabel,' & {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} & {4:.2f} & {5:.2f}  \\\\'.format(nmbg,rg,nmbc,rc,nmbk,rk)
    else:
        print ylabel,' & {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} & \\\\'.format(nmbg,rg,nmbk,rk)
def get_vertical_profiles(model_height_coord_1,interp_altitudes,interp_result):
    model_height_coord=[5.0000000e+00,2.1666664e+01,4.5000000e+01,7.5000000e+01,1.1166668e+02,
                        1.5500000e+02, 2.0500000e+02, 2.6166668e+02, 3.2500000e+02, 3.9500000e+02,
                        4.7166680e+02, 5.5500000e+02, 6.4500000e+02, 7.4166680e+02,8.4500000e+02,
                        9.5500000e+02, 1.0716668e+03, 1.1950000e+03, 1.3250000e+03, 1.4616668e+03,
                        1.6050000e+03, 1.7550000e+03, 1.9116668e+03, 2.0750000e+03, 2.2450004e+03,
                        2.4216668e+03, 2.6050000e+03, 2.7950000e+03, 2.9916668e+03, 3.1950000e+03,
                        3.4050000e+03, 3.6216668e+03, 3.8450000e+03, 4.0750000e+03, 4.3116680e+03,
                        4.5550000e+03, 4.8050000e+03, 5.0616680e+03, 5.3250000e+03, 5.5950000e+03,
                        5.8716680e+03, 6.1550080e+03, 6.4451480e+03, 6.7424920e+03,
                        7.0478160e+03, 7.3623600e+03, 7.6879200e+03, 8.0269280e+03,
                        8.3825800e+03, 8.7589160e+03]
    model_height_coord_bounds = [[   0.      ,   13.333332],
                                 [  13.333332,   33.333332],
                                 [  33.333332,   60.      ],
                                 [  60.      ,   93.33332 ],
                                 [  93.33332 ,  133.33332 ],
                                 [ 133.33332 ,  180.      ],
                                 [ 180.      ,  233.33332 ],
                                 [ 233.33332 ,  293.33332 ],
                                 [ 293.33332 ,  360.      ],
                                 [ 360.      ,  433.3332  ],
                                 [ 433.3332  ,  513.3332  ],
                                 [ 513.3332  ,  600.      ],
                                 [ 600.      ,  693.3332  ],
                                 [ 693.3332  ,  793.3332  ],
                                 [ 793.3332  ,  900.      ],
                                 [ 900.      , 1013.3332  ],
                                 [1013.3332  , 1133.3332  ],
                                 [1133.3332  , 1260.      ],
                                 [1260.      , 1393.3332  ],
                                 [1393.3332  , 1533.3332  ],
                                 [1533.3332  , 1680.      ],
                                 [1680.      , 1833.3332  ],
                                 [1833.3332  , 1993.3332  ],
                                 [1993.3332  , 2160.      ],
                                 [2160.      , 2333.3336  ],
                                 [2333.3336  , 2513.3336  ],
                                 [2513.3336  , 2700.      ],
                                 [2700.      , 2893.3336  ],
                                 [2893.3336  , 3093.3332  ],
                                 [3093.3332  , 3300.      ],
                                 [3300.      , 3513.3332  ],
                                 [3513.3332  , 3733.3332  ],
                                 [3733.3332  , 3960.      ],
                                 [3960.      , 4193.332   ],
                                 [4193.332   , 4433.332   ],
                                 [4433.332   , 4680.      ],
                                 [4680.      , 4933.332   ],
                                 [4933.332   , 5193.332   ],
                                 [5193.332   , 5460.      ],
                                 [5460.      , 5733.332   ],
                                 [5733.332  , 6013.336  ],
                                 [6013.336  , 6300.08   ],
                                 [6300.08   , 6593.82   ],
                                 [6593.82   , 6895.156  ],
                                 [6895.156  , 7205.088  ],
                                 [7205.088  , 7525.14   ],
                                 [7525.14   , 7857.424  ],
                                 [7857.424  , 8204.756  ],
                                 [8204.756  , 8570.748  ],
                                 [8570.748  , 8959.928  ]]
    model_height_coord = 1e-3*np.array(model_height_coord)
    model_height_coord_bounds = 1e-3*np.array(model_height_coord_bounds)
    totals_in_altitude_bins=np.zeros(len(model_height_coord)-6)
    numbers_in_altitude_bins=np.zeros(len(model_height_coord)-6)
    totals_squared_in_altitude_bins=np.zeros(len(model_height_coord)-6)
    model_altitude_bounds = model_height_coord_bounds
    ibin=0
    altitude_bounds_to_use = [[0,0.230]]
    i=0
    altitude_points_to_use=[0.115]
    for height_interval in model_altitude_bounds[7:]:
        altitude_bounds_to_use.append(height_interval)
        altitude_points_to_use.append(model_height_coord[i+7])
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


def plot_vertical(df,model_height_coord,var_name,xmax,xlabel,saving_name,two_models,df2, plot_obs=1, x_min=0):
    #print model_height_coord
    if plot_obs==1:
        means_obs,stds_obs,altitude_points_to_use = get_vertical_profiles(model_height_coord,df.altitude_agl,df[var_name])
    means_lam,stds_lam,altitude_points_to_use = get_vertical_profiles(model_height_coord,df.altitude_agl,df['model_'+var_name])
    if two_models==1:
        means_lam2,stds_lam2,altitude_points_to_use2 = get_vertical_profiles(model_height_coord,df2.altitude_agl,df2['model_'+var_name])
    means_glm,stds_glm,altitude_points_to_use = get_vertical_profiles(model_height_coord,df.altitude_agl,df['glm_'+var_name])
    plt.figure()
    if plot_obs==1:
        plt.semilogy(means_obs,altitude_points_to_use,label='Obs', color='black',linewidth=1.5,basey=1.8 )

    if two_models==0:
        plt.semilogy(means_lam,altitude_points_to_use,label='Regn.',basey=1.8,linewidth=1.5)
    else:
        plt.semilogy(means_lam,altitude_points_to_use,label='KORUSv5',basey=1.8,linewidth=1.5,color='blue')
        plt.semilogy(means_lam2,altitude_points_to_use,label='CMIP6',basey=1.8,linewidth=1.5,color='green')
    plt.fill_betweenx(altitude_points_to_use,means_lam-stds_lam,means_lam+stds_lam, color='lightskyblue',alpha=0.5)
    if two_models==1:
        plt.fill_betweenx(altitude_points_to_use,means_lam2-stds_lam2,means_lam2+stds_lam2, color='lawngreen',alpha=0.5)
    if plot_obs==1:
        plt.fill_betweenx(altitude_points_to_use,means_obs-stds_obs,means_obs+stds_obs, color='silver', alpha=0.3)
    plt.semilogy(means_glm,altitude_points_to_use,label='Global',basey=1.8,linewidth=1.5, color='orange')
    plt.xlim(x_min,xmax)
    plt.ylim(0.1,8.5)
    plt.gca().yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks([],[])
    plt.tick_params(left=False)
    plt.yticks([0.2,0.5,1.0,2.0,5.0,8.0], [0.2,0.5,1.0,2.0,5.0,8.0])
    plt.xlabel(xlabel,fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylabel('Altitude (km)',fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf/0731_comparison/'+saving_name)

def plot_pdfs(df,df2,varname,lower,higher, label,labels=0):
    counts, bins = np.histogram(df['glm_'+varname],bins=40,range=(lower,higher))
    if labels ==1:
        plt.hist(bins[:-1], bins, weights=0.5*counts,color='orange',histtype='step',label='Global', alpha=1.0)
        ##plt.hist(df['glm_'+varname],bins=40,range=(lower,higher),color='orange',histtype='step',label='Global')
        plt.hist(df['model_'+varname],bins=40,range=(lower,higher),color='blue',histtype='step',label='KORUSv5', alpha=1.0)
        plt.hist(df2['model_'+varname],bins=40,range=(lower,higher),color='green',histtype='step',label='CMIP6', alpha=1.0)
        plt.hist(df[varname],bins=40,range=(lower,higher),color='k',histtype='step',label='Observations')
    else:
        #plt.hist(df['glm_'+varname],bins=40,range=(lower,higher),color='orange',histtype='step', alpha=0.0)
        plt.hist(bins[:-1], bins, weights=0.5*counts,color='orange',histtype='step', alpha=1.0)
        plt.hist(df['model_'+varname],bins=40,range=(lower,higher),color='blue',histtype='step', alpha=1.0)
        plt.hist(df2['model_'+varname],bins=40,range=(lower,higher),color='green',histtype='step', alpha=1.0)
        plt.hist(df[varname],bins=40,range=(lower,higher),color='k',histtype='step')
    stats = [np.mean(df[varname]),np.mean(df['glm_'+varname]),np.mean(df2['model_'+varname]),np.mean(df['model_'+varname]),
             np.std(df[varname]),np.std(df['glm_'+varname]),np.std(df2['model_'+varname]),np.std(df['model_'+varname])]
    plt.xlabel(label)
    return stats

def plot_correlations(df,df2, varname, lower,higher, label, labels=0):
    if labels==1:
        plt.scatter(df[varname],df['model_'+varname], label='KORUSv5')
    else:
        plt.scatter(df[varname],df['model_'+varname])
    plt.xlabel('Obs. '+label)
    plt.ylabel('Model '+label)
    plt.xlim(lower,higher)
    plt.ylim(lower,higher)

def my_lognorm(xarr,n,sigma,dpg):
    pdf=[]
    for x in xarr:      #p362 eq:8.32
        arg = -1*((np.log(x)-np.log(dpg))*(np.log(x)-np.log(dpg)))/(2*np.log(sigma)*np.log(sigma))
        dndlndp = n/(np.log(sigma)*np.sqrt(2*np.pi))*np.exp(arg)
        pdf.append(dndlndp)
    return np.array(pdf)
def plot_sizedistribution(df_smps,saving_name):
    df_smps_np = df_smps.to_numpy()
    print(df_smps.to_string())
    all_sizes = 1e-3*np.array(sizebins_p3b+las_sizes_p3b) #use microns
    plt.figure()
    plt.loglog(all_sizes,df_smps[5:len(all_sizes)+5],'+',color='k', label='Obs')
    sumN = 0
    i=0
    dlogD = 0.05
    for size in all_sizes:
        if size >1.0:
            sumN = sumN+df_smps[5+i]*0.1
        else:
            sumN = sumN+df_smps[5+i]*dlogD
        i=i+1
    print 'My calculation of N_SMPS - includes LAS',sumN, 'actual calculation',df_smps['nSMPS_stdPT']
    x = np.arange(0.001,10.0,0.0003)
    #acc_lognorm_pdf = 2.303*(df_smps["model_AccN"]/(1e6*df_smps["model_AccD"]))*lognorm.pdf(x,np.log(1.4),scale = 1e6*df_smps["model_AccD"])
    #ait_lognorm_pdf = 2.303*(df_smps["model_AitN"]/(1e6*df_smps["model_AitD"]))*lognorm.pdf(x,np.log(1.59),scale = 1e6*df_smps["model_AitD"])                    
    #aitins_lognorm_pdf = 2.303*(df_smps["model_AInsN"]/(1e6*df_smps["model_AInsD"]))*lognorm.pdf(x,np.log(1.59),scale = 1e6*df_smps["model_AInsD"])
    #cor_lognorm_pdf = 2.303*(df_smps["model_CorN"]/(1e6*df_smps["model_CorD"]))*lognorm.pdf(x,np.log(2.0),scale = 1e6*df_smps["model_CorD"])
    #dN/dlnDp = (1/2.303)*dN/dlog10Dp: as dlog10Dp = 0.05, dlogeDp =0.1151 
    #acc_lognorm_pdf = 2.303*df_smps["model_AccN"]*lognorm.pdf(x,np.log(1.4),scale = 1e6*df_smps["model_AccD"]) # this is dN/dlogeDp
    #ait_lognorm_pdf =2.303*df_smps["model_AitN"]*lognorm.pdf(x,np.log(1.59),scale = 1e6*df_smps["model_AitD"])# multiplying by 0.05 or 0.1 would be needed for reasonable behaviour. But doing this would mean N100 is underestimated, and actually it isn't too bad.
    #aitins_lognorm_pdf =2.303*df_smps["model_AInsN"]*lognorm.pdf(x,np.log(1.59),scale = 1e6*df_smps["model_AInsD"])
    #cor_lognorm_pdf =2.303*df_smps["model_CorN"]*lognorm.pdf(x,np.log(2.0),scale = 1e6*df_smps["model_CorD"])
    nuc_lognorm_pdf =2.303*my_lognorm(x,df_smps["model_NucN"],1.59,1e6*df_smps["model_NucD"])
    acc_lognorm_pdf =2.303*my_lognorm(x,df_smps["model_AccN"],1.4,1e6*df_smps["model_AccD"])
    ait_lognorm_pdf = 2.303*my_lognorm(x,df_smps["model_AitN"],1.59,1e6*df_smps["model_AitD"])
    aitins_lognorm_pdf =  2.303*my_lognorm(x,df_smps["model_AInsN"],1.59,1e6*df_smps["model_AInsD"])
    cor_lognorm_pdf = 2.303*my_lognorm(x,df_smps["model_CorN"],2.0,1e6*df_smps["model_CorD"])
    total_cor_N = lognorm.cdf(10.0,np.log(2.0),scale = 1e6*df_smps["model_CorD"]) # check this is 1, yes it is.
    print 'model normalization check',df_smps["model_CorN"],total_cor_N#[len(total_cor_N)-1]

    plt.loglog(x,nuc_lognorm_pdf,linestyle=':', label='Nuc')
    plt.loglog(x,ait_lognorm_pdf,linestyle=':', label='Ait')
    plt.loglog(x,acc_lognorm_pdf,linestyle=':', label='Acc')
    plt.loglog(x,cor_lognorm_pdf,linestyle=':', label='Cor')
    plt.loglog(x,aitins_lognorm_pdf,linestyle=':', label='Ait-ins')

    # i commented out the dust related thing since Hamish cp754 does not have dust
    # dust_dlogD = 0.5
    # dust_bins = [0.0316,0.1,0.316,1.0,3.16,10.0]
    # bin_widths =[0.1-0.0316,0.316-0.1,1-0.316,3.16-1,10-3.16, 31.6-10]
    # dust_dndlogd = (1e-6/dust_dlogD)*np.array([df_smps['model_db1'],df_smps['model_db2'],df_smps['model_db3'],df_smps['model_db4'],df_smps['model_db5'],df_smps['model_db6']])
    # plt.bar(dust_bins, dust_dndlogd,width=bin_widths,align='edge', label='Dust', fill=False,lw=1.5, color='gray', linestyle='--', edgecolor='gray')
    total = nuc_lognorm_pdf+ait_lognorm_pdf+acc_lognorm_pdf+cor_lognorm_pdf+aitins_lognorm_pdf
    # i=0
    # for ix in x:
    #     if ix > 0.0316 and ix <= 0.1:
    #         total[i] = total[i]+1e-6/dust_dlogD*df_smps['model_db1']
    #     elif ix > 0.1 and ix <= 0.316:
    #         total[i] = total[i]+1e-6/dust_dlogD*df_smps['model_db2']
    #     elif ix > 0.316 and ix <= 1.0:
    #         total[i] = total[i]+1e-6/dust_dlogD*df_smps['model_db3']
    #     elif ix > 1.0 and ix <= 3.16:
    #         total[i] = total[i]+1e-6/dust_dlogD*df_smps['model_db4']
    #     elif ix > 3.16 and ix <= 10.0:
    #         total[i] = total[i]+1e-6/dust_dlogD*df_smps['model_db5']
    #     i=i+1
    plt.loglog(x,total,linestyle='-',label='Reg. total')
    plt.ylim(1e-2,1e5)
    plt.xlim(1e-3,12.0)
    plt.legend()
    plt.tight_layout()
    plt.ylabel('dN/dlogDp',fontsize=16)
    plt.xlabel('Diameter ($\mu$m)',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf/0731_comparison/'+saving_name)

    
    
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
    if cube.ndim==4:
        cube.add_aux_coord(longit, (2,3))
        cube.add_aux_coord(latit, (2,3))
    elif cube.ndim==3:
        cube.add_aux_coord(longit, (1,2))
        cube.add_aux_coord(latit, (1,2))
    elif cube.ndim==2:
        cube.add_aux_coord(longit, (0,1))
        cube.add_aux_coord(latit, (0,1))
    return bbox_extract_2Dcoords(cube, bbox)

def load_um_cube(timeindexes,surfcoord, inputfile,cubename, is_lam):
    sys.stdout.flush()
    cubelist = iris.cube.CubeList()
    for time_index in timeindexes:
        try:
            cubelist.append(iris.load_cube(inputfile+time_index,cubename))
        except Exception:
            # print cubename
            cubelist.append(iris.load(inputfile+time_index,cubename)[0])
    if cubelist[0].ndim == 4:
        cube = cubelist.concatenate_cube()
    elif cubelist[0].ndim == 3:
        # print cubelist
        cube = cubelist.merge_cube()
    add_orog(cube, surfcoord, (2, 3,))
    if is_lam:
        add_lat_lon(cube, bbox)
    else:
        cube1 = cube.extract(iris.Constraint(latitude=lambda cell: latmin - 1.0 < cell < latmax + 1))
        cube = cube1[:, :40, :, :].extract(iris.Constraint(longitude=lambda cell: lonmin + 359 < cell < lonmax + 361))
    # print(cube)
    return cube
from math import log10
def getNfromV(du_vol, dp1,dp2,dp3):
    n0V = du_vol/(log10(dp2)-log10(dp1))
    N = 2*n0V/(2.303*3.14159)*(1/pow(dp3,3)-1/pow(dp2,3))
    return N


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
def load_model(timeindexes,path,varlist,date,is_lam,is_dump,add_smps,offline1=0):
    outputcubedict={}
    smpscubedict={} #an extra dictionary for smps log normal distribution
    print varlist
    if is_lam:
        prefix = path+'umnsaa'
        denslabel='_pb'
        timeindexes_den=timeindexes
    else:
        prefix = path+'umglaa'
        denslabel='_pe'
        timeindexes_den = timeindexes

    if not is_dump:
        print('loading orog')
        orog = iris.load_cube(prefix+'_pa000',iris.AttributeConstraint(STASH='m01s00i033'))
        surfcoord = iris.coords.AuxCoord(1e-3*orog.data,'surface_altitude', units='km')
    # else:
    #     surfcoord=None
    # #
    if 'T' in varlist:
        if not is_dump:
            theta = load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i004'),is_lam)
            air_pressure = load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i408'),is_lam)
        else:
            theta = iris.load_cube(path,iris.AttributeConstraint(STASH='m01s00i004')) #theta
            cube1 =theta.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
            theta = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
            exner = iris.load_cube(path,iris.AttributeConstraint(STASH='m01s00i255'))
            # print exner
            cube1 =exner.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
            #exner = cube1[:,:40,:,:].
            exner = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
            # print exner
        p0 = iris.coords.AuxCoord(1000.0,
                                  long_name='reference_pressure',
                                  units='hPa')
        if not is_dump:
            p0.convert_units(air_pressure.units)

    # if 'T' in varlist:
    #     if not is_dump:
    #         theta = load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i004'),is_lam)
    #         # print theta.coord('level_height')
    #         air_pressure = load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i408'),is_lam)
    #     else:
    #         theta = iris.load_cube(path,iris.AttributeConstraint(STASH='m01s00i004'))
    #         cube1 =theta.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
    #         theta = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+loncor-1 < cell < lonmax+loncor+1))
    #         # theta = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
    #         exner = iris.load_cube(path,iris.AttributeConstraint(STASH='m01s00i255'))
    #         cube1 =exner.extract(iris.Constraint(latitude = lambda cell: latmin-1.0 < cell < latmax+1))
    #         #exner = cube1[:,:40,:,:].
    #         exner = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+loncor-1 < cell < lonmax+loncor+1))
    #         # exner = cube1[:40,:,:].extract(iris.Constraint(longitude = lambda cell: lonmin+359 < cell < lonmax+361))
    #     p0 = iris.coords.AuxCoord(1000.0,long_name='reference_pressure',units='hPa')
    #     if not is_dump:
    #         p0.convert_units(air_pressure.units)
        # if not is_dump:
        #     p0.convert_units(air_pressure.units)
        Rd=287.05                                                                                                               
        cp=1005.46                                                                                                                                  
        Rd_cp=Rd/cp
        if not is_dump:
            temperature=theta*(air_pressure/p0)**(Rd_cp)
            ## I followed colorado_rerun_0731_p3b to start with

            if mod_lev==None:
                print(temperature.data[0,0,0,0])
                print('MIN TEMP',np.min(temperature.data),'COL',temperature.data[0,:,3,3])
            else:
                print(temperature.data[0,0,0])
                print('MIN TEMP',np.min(temperature.data),'COL',temperature.data[0,3,3])
            #to do, since I dont have RH added yet in my u-co631 so commented out.
            # model_q = load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i010'),is_lam,mod_lev)
            # exp_term = np.exp((17.62*(temperature.data-273.12))/(temperature.data-30.0))
            # model_rh = model_q.copy(0.263*model_q.data*air_pressure.data/exp_term)
            # print(model_rh.data[0,0,0])
            # outputcubedict["RH"]= model_rh
        else:
            temperature = theta.copy(theta.data*exner.data)
            # print(temperature.data[0,0,0])
            # print('DUMP TEMP ALTITUDE COLM',temperature.coord('altitude').points[:,25,25])
            # print('MINIMUM SURFACE TEMP',np.min(temperature.data[0,:,:]),'MIN TEMP',np.min(temperature.data))
        outputcubedict["T"]  =temperature
    if 'SO2' in varlist:
        so2_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_sulfur_dioxide_in_air', is_lam)
        so2_conc = so2_mixing*0.029/0.064*1e12 #pptv   
        outputcubedict["SO2"] = so2_conc
    if 'O3' in varlist:
        if offline1==0:
            o3_string = 'mass_fraction_of_ozone_in_air'
        else:
            o3_string = iris.AttributeConstraint(STASH='m01s50i206')
        o3_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc',o3_string, is_lam)
        o3_conc = o3_mixing*0.029/0.048*1e9 #ppbv
        outputcubedict["O3"] = o3_conc
    if 'C5H8' in varlist:
        c5h8_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_isoprene_in_air',is_lam)
        c5h8_conc = c5h8_mixing*0.029/0.068*1e12 #pptv
        # c5h8_mixing = load_um_cube(timeindexes, surfcoord, prefix + '_pc', iris.AttributeConstraint(STASH='m01s34i027'),
        #                          is_lam,mod_lev)
        # c5h8_conc = c5h8_mixing * 0.029 / 0.068 * 1e12  # pptv
        outputcubedict["C5H8"] = c5h8_conc
    if 'C10H16' in varlist:
        mtlabel='_pd' #I dont have mt
        mt_mixing = load_um_cube(timeindexes,surfcoord,prefix+mtlabel,iris.AttributeConstraint(STASH='m01s34i091'),is_lam)
        mt_conc = mt_mixing*0.029/0.136*1e12 #pptv
        outputcubedict["C10H16"] = mt_conc

    if 'CO' in varlist: #this is new added
        colabel='_pc'
        co_mixing = load_um_cube(timeindexes,surfcoord,prefix+colabel,iris.AttributeConstraint(STASH='m01s34i010'),is_lam)
        co_conc = co_mixing*0.029/0.028*1e9    #ppbv notice CO is ppb instead of ppt
        outputcubedict["CO"] = co_conc

    if 'OH' in varlist: #an extra string label to distinguish offline or online
        if offline1==0:
            oh_string=iris.AttributeConstraint(STASH='m01s34i081')
            midfix='_pd'
        else:
            oh_string =iris.AttributeConstraint(STASH='m01s50i207')
            midfix='_pc'
        oh_mixing = load_um_cube(timeindexes,surfcoord,prefix+midfix,oh_string,is_lam)
        oh_conc = oh_mixing * 0.029 / 0.017 * 1e12  # pptv
        outputcubedict["OH"]=oh_conc
    if 'JO1D' in varlist:  #photolysis_rate_of_ozone_to_1D_oxygen_atom
        if is_lam:
            jo1d = load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s50i228'),is_lam)
        else:
            jo1d = load_um_cube(timeindexes,surfcoord,prefix+'_pe',iris.AttributeConstraint(STASH='m01s50i228'),is_lam)
        outputcubedict["JO1D"]=1e5*jo1d
    if 'H2SO4' in varlist:  #same with colorado_rereun_0731
        h2so4_mixing = load_um_cube(timeindexes,surfcoord,prefix + '_pc', iris.AttributeConstraint(STASH='m01s34i073'),is_lam)
        h2so4_conc = h2so4_mixing * 0.029 / 0.098 * 1e12  # pptv
        outputcubedict["H2SO4"]=h2so4_conc
    if 'NH3' in varlist: #same with colorado_rereun_0731
        nh3_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_ammonia_in_air', is_lam)
        nh3_conc = nh3_mixing*0.029/0.017*1e12 #pptv
        outputcubedict["NH3"]=nh3_conc
    if 'HO2' in varlist: #an extra string label to distinguish offline or online
        if offline1==0:
            ho2_string=iris.AttributeConstraint(STASH='m01s34i082')
            midfix='_pd'
        else:
            ho2_string=iris.AttributeConstraint(STASH='m01s50i209')
            midfix='_pc'
        ho2_mixing = load_um_cube(timeindexes,surfcoord,prefix+midfix,ho2_string,is_lam)
        ho2_conc = ho2_mixing*0.029/0.033*1e12 #pptv
        outputcubedict["HO2"]=ho2_conc
    # if 'CO' in varlist:
    #     co_mixing = load_um_cube(timeindexes,surfcoord,prefix+'_pc','mass_fraction_of_carbon_monoxide_in_air',is_lam,mod_lev)
    #     co_conc = co_mixing*0.029/0.028*1e9 #ppbv
    #     outputcubedict["CO"]=co_conc
    if 'NO' in varlist: #same with colorado_rereun_0731
        nolabel='_pd'
        if is_lam:
            nolabel='_pc'
        no_mixing = load_um_cube(timeindexes,surfcoord,prefix+nolabel,iris.AttributeConstraint(STASH='m01s34i002'),is_lam)
        no_conc = no_mixing*0.029/0.030*1e12
        outputcubedict["NO"]=no_conc
    if 'U' in varlist: #for both u/v direction, in pc file.
        u_wind = load_um_cube(timeindexes,surfcoord,prefix+'_pc',iris.AttributeConstraint(STASH='m01s00i002'),is_lam)
        outputcubedict["U"] = u_wind
    if 'V' in varlist:
        v_wind = load_um_cube(timeindexes,surfcoord,prefix+'_pc',iris.AttributeConstraint(STASH='m01s00i003'),is_lam)
        outputcubedict["V"] = v_wind
    if 'dust' in varlist:
        if is_lam:
            db1 = load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s00i431'),is_lam) #checked
            db2 = load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s00i432'),is_lam) #checked
            db3 = load_um_cube(timeindexes,surfcoord,prefix+'_pd',iris.AttributeConstraint(STASH='m01s00i433'),is_lam) #checked
            db4 = load_um_cube(timeindexes,surfcoord,prefix+'_pe',iris.AttributeConstraint(STASH='m01s00i434'),is_lam) #checked
            db5 = load_um_cube(timeindexes,surfcoord,prefix+'_pe',iris.AttributeConstraint(STASH='m01s00i435'),is_lam) #checked
            db6 = load_um_cube(timeindexes,surfcoord,prefix+'_pd',iris.AttributeConstraint(STASH='m01s00i436'),is_lam) #checked
        else:
            db1 = load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s00i431'),is_lam) #checked
            db2 = load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s00i432'),is_lam) #checked
            db3 = load_um_cube(timeindexes,surfcoord,prefix+'_pe',iris.AttributeConstraint(STASH='m01s00i433'),is_lam) #checked
            db4 = load_um_cube(timeindexes,surfcoord,prefix+'_pe',iris.AttributeConstraint(STASH='m01s00i434'),is_lam) #checked
            db5 = load_um_cube(timeindexes,surfcoord,prefix+'_pc',iris.AttributeConstraint(STASH='m01s00i435'),is_lam) #checked
            db6 = load_um_cube(timeindexes,surfcoord,prefix+'_pc',iris.AttributeConstraint(STASH='m01s00i436'),is_lam) #checked
        #dust mass mixing ratio in bin
        pref=1.013E5
        tref=293.0 #Assume STP is 20C
        dust_density = 2560
        gc=8.314
        staird=pref/(tref*gc)*0.029/dust_density # kg of air m-3 divided by dust density
        #db1*staird/dust_density gives m3 dust m-3
        #dust_bins = [0.0316,0.1,0.316,1.0,3.16,3.16,10.0,31.6]
        du_vol_1 = db1*staird
        du_vol_2 = db2*staird
        du_vol_3 = db3*staird
        du_vol_4 = db4*staird
        du_vol_5 = db5*staird
        du_vol_6 = db6*staird
        du_num_1 = getNfromV(du_vol_1, 0.632e-7,0.2e-6,0.632e-7)
        du_num_2 = getNfromV(du_vol_2, 0.2e-6,0.632e-6,0.2e-6)
        du_num_3 = getNfromV(du_vol_3,0.632e-6,2.0e-6,0.632e-6)
        du_num_4 = getNfromV(du_vol_4,2.0e-6,6.32e-6,2.0e-6)
        du_num_5 = getNfromV(du_vol_5,6.32e-6,2.0e-5,6.32e-6)
        du_num_6 = getNfromV(du_vol_6,2.0e-5,6.32e-5,2.0e-5)
        outputcubedict["db1"] =du_num_1
        outputcubedict["db2"] =du_num_2
        outputcubedict["db3"] =du_num_3
        outputcubedict["db4"] =du_num_4
        outputcubedict["db5"] =du_num_5
        outputcubedict["db6"] =du_num_6
        if add_smps:
            smpscubedict["db1"] =du_num_1
            smpscubedict["db2"] =du_num_2
            smpscubedict["db3"] =du_num_3
            smpscubedict["db4"] =du_num_4
            smpscubedict["db5"] =du_num_5
            smpscubedict["db6"] =du_num_6
    if 'N3' in varlist or 'N100' in varlist or 'N10' in varlist or add_smps:
        numlabel='_pb'
        sizelabel = '_pe' #newly added???
        if is_lam:
            sizelabel='_pd'
            numlabel='_pe'
        # load Number first
        nuc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i101'),is_lam)
        ait_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i103'),is_lam)
        acc_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i107'),is_lam)
        cor_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i113'),is_lam)
        aitins_num_mixing = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s34i119'),is_lam)

        # n10_num_mixing = ait_num_mixing+acc_num_mixing+cor_num_mixing+aitins_num_mixing
        # nall_num_mixing = n10_num_mixing + nuc_num_mixing
        #if is_lam:
        #ait_diam = load_um_cube(timeindexes, surfcoord, prefix+numlabel,iris.AttributeConstraint(STASH='m01s38i402'),is_lam,mod_lev)
        # load D second
        if is_lam:
            # for regional model, all in pd
            nuc_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                        iris.AttributeConstraint(STASH='m01s38i401'), is_lam)
            ait_diam= load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                        iris.AttributeConstraint(STASH='m01s38i402'), is_lam)
            acc_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                        iris.AttributeConstraint(STASH='m01s38i403'), is_lam)
            cor_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                         iris.AttributeConstraint(STASH='m01s38i404'), is_lam)
            aitins_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pd',
                                         iris.AttributeConstraint(STASH='m01s38i405'), is_lam)
            # dia_list = [ait_ria_wet, acc_ria_wet, coar_ria_wet,ait_ria_inso]

        else:
            # for global model, all in pe
            nuc_diam= load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                        iris.AttributeConstraint(STASH='m01s38i401'), is_lam)
            ait_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                        iris.AttributeConstraint(STASH='m01s38i402'), is_lam)
            acc_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                        iris.AttributeConstraint(STASH='m01s38i403'), is_lam)
            cor_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                         iris.AttributeConstraint(STASH='m01s38i404'), is_lam)
            aitins_diam = load_um_cube(timeindexes, surfcoord, prefix + '_pe',
                                         iris.AttributeConstraint(STASH='m01s38i405'), is_lam)

        #sys.stdout.flush()
        #else:
        #    if new_diam_calc:
        #        ait_so4 =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i104'),is_lam,mod_lev)
        #        ait_bc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i105'),is_lam,mod_lev)
        #        ait_oc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i106'),is_lam,mod_lev)
        #        acc_so4 =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i108'),is_lam,mod_lev)
        #        acc_bc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i109'),is_lam,mod_lev)
        #        acc_oc =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i110'),is_lam,mod_lev)
        #        acc_ss =load_um_cube(timeindexes, surfcoord, prefix+'_pb',iris.AttributeConstraint(STASH='m01s34i111'),is_lam,mod_lev)
        #        ait_diam = calc_diameter(ait_num_mixing,ait_so4,ait_bc,ait_oc)
        #        acc_diam = calc_diameter(acc_num_mixing,acc_so4,acc_bc,acc_oc, acc_ss)
        #        iris.save(ait_diam,path+'Aitken_diameter_'+date+'.nc')
        #        iris.save(acc_diam,path+'Accumulation_diameter_'+date+'.nc')
        #        print(ait_diam)
        #        print('saved diameters')
        #    else:
        #        if mod_lev==None:
        #            ait_diam = iris.load(path+'Aitken_diameter_'+date+'.nc')[0]
        #            acc_diam = iris.load(path+'Accumulation_diameter_'+date+'.nc')[0]
        #        else:
        #            ait_diam = iris.load(path+'Aitken_diameter_'+date+'.nc',iris.Constraint(model_level_number=mod_lev))[0]
        #            acc_diam = iris.load(path+'Accumulation_diameter_'+date+'.nc',iris.Constraint(model_level_number=mod_lev))[0]
        print(ait_diam)
        n10_num_mixing = ait_num_mixing + acc_num_mixing + cor_num_mixing + aitins_num_mixing
        nall_num_mixing = n10_num_mixing + nuc_num_mixing
        n10_num_mixing = nall_num_mixing - lognormal_cumulative_forcubes(nuc_num_mixing, 1.0e-8, nuc_diam, 1.59)
        #ait_diam = load_um_cube(timeindexes, surfcoord, prefix+'_pe',iris.AttributeConstraint(STASH='m01s38i402'),is_lam,mod_lev)
        #acc_diam = load_um_cube(timeindexes, surfcoord, prefix+'_pe',iris.AttributeConstraint(STASH='m01s38i403'),is_lam,mod_lev)
        n100_mixing = ait_num_mixing-lognormal_cumulative_forcubes(ait_num_mixing, 1.0e-7,ait_diam,1.59)
        n100_mixing = n100_mixing+aitins_num_mixing - lognormal_cumulative_forcubes(aitins_num_mixing, 1.0e-7,aitins_diam,1.59)
        n100_mixing = n100_mixing+acc_num_mixing - lognormal_cumulative_forcubes(acc_num_mixing, 1.0e-7,acc_diam,1.59)
        print('done lognormal')
        sys.stdout.flush()
        n100_mixing = n100_mixing+cor_num_mixing
        pref=1.013E5
        tref=293.0 #Assume STP is 20C                                                                                                                                           
        zboltz=1.3807E-23
        staird=pref/(tref*zboltz*1.0E6)

        n10_num_conc = n10_num_mixing * staird
        nall_num_conc = nall_num_mixing * staird
        n100_num_conc = n100_mixing*staird

        outputcubedict["N10"] = n10_num_conc
        outputcubedict["N100"] = n100_num_conc
        outputcubedict["Nall"] = nall_num_conc

        if add_smps==1:
            smpscubedict["NucN"] = nuc_num_mixing*staird
            smpscubedict["NucD"] = nuc_diam
            smpscubedict["AitN"] = ait_num_mixing*staird
            smpscubedict["AitD"] = ait_diam
            smpscubedict["AccN"] = acc_num_mixing*staird
            smpscubedict["AccD"] = acc_diam
            smpscubedict["CorN"] = cor_num_mixing*staird
            smpscubedict["CorD"] = cor_diam
            smpscubedict["AInsN"] = aitins_num_mixing*staird
            smpscubedict["AInsD"] = aitins_diam
        print('output number cubes')

    if 'CDNC' in varlist:
        aird = (1.0/(6371000.0*6371000))*load_um_cube(timeindexes,surfcoord,prefix+denslabel,iris.AttributeConstraint(STASH='m01s00i253'),is_lam)
        if is_lam:
            cdnc_in_cloud = 1e-6*load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s38i479'),is_lam) #num/m3->num/cc
#            cdnc_in_cloud = 1e-6*load_um_cube(timeindexes,surfcoord,prefix+'_pb',iris.AttributeConstraint(STASH='m01s00i075'),is_lam,mod_lev_cld) #num/m3->num/cc                                 
            lwc_prefix='_pb'
        else:
            cdnc_in_cloud = 1e-6*load_um_cube(timeindexes,surfcoord,prefix+'_pc',iris.AttributeConstraint(STASH='m01s34i968'),is_lam) #num/m3->num/cc
            lwc_prefix='_pe'
        outputcubedict["CDNC"]=cdnc_in_cloud
        lwc1 = 1e3*load_um_cube(timeindexes,surfcoord,prefix+lwc_prefix,iris.AttributeConstraint(STASH='m01s00i254'),is_lam)
        lwc =lwc1.copy(lwc1.data*aird.data)
        outputcubedict["LWC"]=lwc

    if 'P' in varlist:
        air_pressure = load_um_cube(timeindexes, surfcoord, prefix + denslabel,
                                    iris.AttributeConstraint(STASH='m01s00i408'), is_lam)
        outputcubedict["P"] = air_pressure
    # want number concentration at STP per cc. Good, because I can't easily multiply by density anyway as pe is written every 6 hours. Change this!
    #return iris.cube.CubeList([so2_conc,oh_conc,co_conc,c5h8_conc,mt_conc,no_conc,n10_num_conc,n100_num_conc,cdnc_in_cloud])#,lwc])
    if add_smps:
        return outputcubedict,smpscubedict
    else:
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
    # print('interp_flight_data',gas_time_a.shape,gas_a.shape,track_time.shape)
    if(len(gas_time_a.shape) > 1):
        print('strange shape of array')
        gas_time_a = gas_time_a[0]
        gas_a = gas_a[0]
        print(gas_time_a.shape,gas_a.shape)
        if(len(gas_a.shape) > 1):
            gas_a = gas_a[0]
            print(gas_time_a.shape,gas_a.shape)
    interp_gas = interp1d(gas_time_a,gas_a,kind='nearest', bounds_error=False,fill_value=-9999)
    gas_track = interp_gas(np.asarray(track_time))
    gas_track_a = ma.masked_where(gas_track < l_o_d, gas_track)
    return gas_track_a


def read_flight_data(date):
    flight_path='/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-reveal_p3b_20140731_r0.ict'

    if date == '20140731':
        headerlen = 132
    elif date == '20140802':
        headerlen = 133

    tracktime, alt, lat, lon = read_flight_track(flight_path, 73, 4, 2, 3)
    tracktime, rh, temp, pres2 = read_flight_track(flight_path, 73, 36, 18, 24)

    # tracktime = ma.masked_where(np.asarray(lat) < -90, np.array(tracktime))
    # tracktime2 = ma.masked_where(np.asarray(lon) < -180, np.array(tracktime))
    # tracktime = (tracktime2[~tracktime.mask])
    # if len(tracktime.shape) > 1:
    #     tracktime = tracktime[0]
    # # print 'tracktime.shape',tracktime.shape,np.asarray(lon).shape,np.asarray(lat).shape
    # lon = np.array(lon)[~tracktime2.mask]
    # lat = np.array(lat)[~tracktime2.mask]
    # alt = np.array(alt)[~tracktime2.mask]
    # temp = np.array(temp)[~tracktime2.mask]
    # pres2 = np.array(pres2)[~tracktime2.mask]
    # print('lon', lon)
    # # rh = np.array(rh)[~tracktime2.mask]
    # #
    # temp = interp_flight_data(tracktime, tracktime, temp, -99)
    # alt = interp_flight_data(tracktime, tracktime, alt, -99) #needs to double check here should be asl???
    # # # rhtrack = interp_flight_data(tracktime, tracktime, rh, -99) #we will use this as output
    # pres2 = interp_flight_data(tracktime, tracktime, pres2, -99)

    pres2track = np.multiply(pres2, 100)
    # temptrack = temptrack + 273.15 #we will use this as output
    temptrack = np.array(temp) + 273.15
    ####what commented(4lines) out is H's version
    # tracktime = np.array(tracktime)
    # alt_agl=0.3048*np.array(alt_agl)-35.0
    # temp = np.array(temp)+273.15
    # rhtrack = interp_flight_data(tracktime,tracktime,rh, 0)

    c5h8_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/discoveraq-PTRTOF-NMHCs_P3B_20140731_R0.ict'
    c5h8time, c5h8, c10h16, nh3 = read_flight_track(c5h8_path, 67, 17, 31, 3) #headerline67
    c5h8track = interp_flight_data(tracktime, c5h8time, c5h8, -99) #ppbv
    c10h16track = interp_flight_data(tracktime, c5h8time, c10h16, -99) #ppbv
    nh3track = interp_flight_data(tracktime, c5h8time, nh3, -99) #ppbv

    no_o3_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-NOXYO3_P3B_20140731_R0.ict'
    notime, no, no2, o3 = read_flight_track(no_o3_path, 47, 3, 5, 6) #headerline 47
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

    co_p3b_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-DACOM_P3B_20140731_R1_L1.ict'
    cotime, co, ch4, dummy = read_flight_track(co_p3b_path, 37, 1, 2,
                                                 1)  # new added, 37headerline
    cotrack = interp_flight_data(tracktime, cotime, co, -99)
    ch4track = interp_flight_data(tracktime, cotime, ch4, -99)

    n_nuc_p3b_path = '/jet/home/ding0928/Colorado/Colorado/p-3b/DISCOVERAQ-LARGE-CNC_P3B_20140731_R0.ict'
    n_nuctime, n3, n10, dummy = read_flight_track(n_nuc_p3b_path, 47, 1, 2,
                                               1)  # new added, 37headerline
    n3track = interp_flight_data(tracktime, n_nuctime, n3, -99)
    n10track = interp_flight_data(tracktime, n_nuctime, n10, -99)
    nnuctrack=n3track-n10track

    n100path='/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE-LAS_P3B_20140731_R0.ict'
    n100time,n100_all,dummy,dummy = read_flight_track(n100path,69,1,1,1) #header69, 1st column is N_all
    print('n100',len(n100time),len(n100_all),len(tracktime))
    n100track_all = interp_flight_data(tracktime,n100time,n100_all, -99)

    smps_path_p3b='/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/DISCOVERAQ-LARGE-SMPS_P3B_20140731_R0.ict'
    twod_smps = pd.read_csv(smps_path_p3b,delimiter=",",header=72,na_values=-9999.00)
    stringbins = ['bin'+str(sizebin) for sizebin in sizebins_p3b]
    columns_smps = ['Start_UTC','End_UTC','Mid_UTC','nSMPS_stdPT','sSMPS_stdPT','vSMPS_stdPT']
    columns_smps = columns_smps + stringbins
    twod_smps.columns=columns_smps

    # las~1s while smps~60s, thus needs to interpolate
    twod_las = pd.read_csv(n100path,delimiter=",",header=120,na_values=-9999.00)
    print twod_las.head()
    numpy_las = twod_las.to_numpy()
    las_bins = ['bin'+str(sizebin) for sizebin in las_sizes_p3b]
    for i in range(12,len(las_bins)+12): #H starts 224nm(10th column in python), here I starts from 282nm
        las_bin = interp_flight_data(twod_smps['Mid_UTC'],numpy_las[:,0],numpy_las[:,i],-99)
        twod_smps[str(las_bins[i-12])] = las_bin

    print('twod_smps',twod_smps)
    # now need to interpolate lat, lon,altitude,alttiude_agl onto smps array
    lat_smps = interp_flight_data(twod_smps['Mid_UTC'],tracktime,lat,-99)
    twod_smps['latitude'] = lat_smps
    alt_smps = interp_flight_data(twod_smps['Mid_UTC'], tracktime, alt, -99)
    twod_smps['altitude'] = alt_smps

    # altagl_smps = interp_flight_data(twod_smps['Mid_UTC'], tracktime, alt,-99)
    # twod_smps['altitude_agl'] = altagl_smps
    lon_smps = interp_flight_data(twod_smps['Mid_UTC'],tracktime,lon,-999)
    twod_smps['longitude'] = lon_smps

    # altagl_smps = interp_flight_data(twod_smps['Mid_UTC'],tracktime,alt_agl,-99)
    # twod_smps['altitude_agl'] = altagl_smps

    # new
    unix_epoch_of_31July2014_0001 = 1406764800
    # tracktime = tracktime - 21600
    if date == '20140731':
        trackpdtime = pd.to_datetime(np.asarray(tracktime) + unix_epoch_of_31July2014_0001, unit='s')
    elif date == '20140801':
        unix_epoch_of_01August2014_0001 = unix_epoch_of_31July2014_0001 + 86400
        trackpdtime = pd.to_datetime(np.asarray(tracktime) + unix_epoch_of_01August2014_0001, unit='s')
    elif date == '20140802':
        unix_epoch_of_02August2014_0001 = unix_epoch_of_31July2014_0001 + 86400 * 2
        trackpdtime = pd.to_datetime(np.asarray(tracktime) + unix_epoch_of_02August2014_0001, unit='s')

    smpspdtime = pd.to_datetime(np.asarray(twod_smps['Mid_UTC'])+unix_epoch_of_31July2014_0001,unit='s')
    twod_smps['time'] = smpspdtime
    twod_smps.set_index('time',inplace=True)
    print twod_smps.head(15)

    d = {'time': pd.Series(trackpdtime), 'latitude': pd.Series(np.asarray(lat)),
         'longitude': pd.Series(np.asarray(lon)),'T':pd.Series(temptrack),'RH':pd.Series(rh),
         'altitude': pd.Series(np.asarray(alt)), 'N_All': pd.Series(np.asarray(n3track)),
         'N_10': pd.Series(np.asarray(n10track)),'C10H16': pd.Series(c10h16track),
         'N_Nuc': pd.Series(nnuctrack), 'C5H8': pd.Series(c5h8track),
         'NH3': pd.Series(nh3track), 'NO': pd.Series(notrack),
         'NO2': pd.Series(no2track), 'O3': pd.Series(o3track),
         'P': pd.Series(pres2track), 'CO': pd.Series(cotrack)}
    df3 = pd.DataFrame(d)
    df2 = df3.resample('3T', on='time').mean()
    return df2,twod_smps

# also works for 3D cubes with no time coordinate
def do_4d_interpolation(cubelist,flight_coords, is_lam,is_dump,year,month,day):
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
    alts_of_flight = flight_coords['altitude']
    #times_of_flight = flight_coords.index
    epoch = datetime.datetime(year, month, day)
    print('DATE',year,month,day)
    sys.stdout.flush()
    times_of_flight=[(d - epoch).total_seconds() for d in flight_coords.index]
    points=[]
    if cube.ndim==4:
        cube_times1 = [cell.point for cell in cube.coord('time').cells()]
        cube_times = [(d - epoch).total_seconds() for d in cube_times1]
        print(cube_times)
    if is_lam:
        print('center of grid,lon,lat',cube.coord('grid_longitude').points[150],cube.coord('grid_latitude').points[150])
        print('offset-lon of grid,lon,lat',cube.coord('grid_longitude').points[120],cube.coord('grid_latitude').points[150])
        print('offset-lat of grid,lon,lat',cube.coord('grid_longitude').points[150],cube.coord('grid_latitude').points[120])
        sys.stdout.flush()
    for ipoint in range(0,len(lats_of_flight)):
        rotated_lon,rotated_lat = input_projection.transform_point(lons_of_flight[ipoint],lats_of_flight[ipoint],crs_latlon)
        if ipoint < 20:
            print('rotated_pole,time,lon,lat',times_of_flight[ipoint],lons_of_flight[ipoint],lats_of_flight[ipoint],rotated_lon,rotated_lat)
        if cube.ndim==4:
            if is_lam:
            # CONVERT FLIGHT ALTITUDES TO KM
                points.append([times_of_flight[ipoint],rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])
            else:
                points.append([times_of_flight[ipoint],rotated_lon+loncor,rotated_lat,1e-3*alts_of_flight[ipoint]])
        else:
            #points.append([times_of_flight[ipoint],rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])
            if is_dump:
                points.append([rotated_lon+360.0,rotated_lat,alts_of_flight[ipoint]])
            else:
                if is_lam:
                    points.append([rotated_lon+360.0,rotated_lat,1e-3*alts_of_flight[ipoint]])   
                else:
                    points.append([rotated_lon+loncor,rotated_lat,1e-3*alts_of_flight[ipoint]])
    cubedimcoords=[]
    cubedatasets=[]
    if cube.ndim==4:
        cubedimcoords.append(cube_times)
        for cube_to_interp in all_cubes:
            cubedatasets.append(np.transpose(cube_to_interp.data,(0,3,2,1)))
            print(cube_to_interp)
    else:
        for cube_to_interp in all_cubes:
            cubedatasets.append(cube_to_interp.data.T)
    sys.stdout.flush()
    if is_lam:
        cubedimcoords.append(cube.coord('grid_longitude').points)
        cubedimcoords.append(cube.coord('grid_latitude').points) 
    else:
        cubedimcoords.append(cube.coord('longitude').points)
        cubedimcoords.append(cube.coord('latitude').points)
    cubedimcoords.append(cube.coord('level_height').points)
    
    print('interpolator hybrid_dim args',cube.coord_dims(cube.coord('altitude')))
    # complicated version of interpolator that handles topography. Credit to Duncan Watson-Parris (cis) and scitools
    if is_lam:
        interp_method = 'nn'
    else:
        interp_method = 'nn'
    interpolator=_RegularGridInterpolator(cubedimcoords,np.asarray(points).T, hybrid_coord =cube.coord('altitude').points.T,hybrid_dims=cube.coord_dims(cube.coord('altitude')),method=interp_method)
    #interpolated_values=my_interpolating_function(points)
    interp_results={}
    i=0
    for key in cube_keys:
        interp_results[key] = np.asarray(interpolator(cubedatasets[i], fill_value=None))
        i=i+1
        #if is_lam:
        #    print(np.asarray(interpolator(cubedatasets[i], fill_value=None)[0]),np.asarray(interpolator(cubedatasets[i], fill_value=None)[1]),np.asarray(interpolator(cubedatasets[i], fill_value=None)[2]),np.max(np.asarray(interpolator(cubedatasets[i], fill_value=None)[2])))
        #    print(np.min(cubedatasets[i]),np.max(cubedatasets[i]))
        #print interpolated_values
    return interp_results

#plt.plot(np.asarray(lat),so2track)


# varlist = ['OH','HO2','C5H8','C10H16','NH3','NO','O3','SO2','CO','N10','dust']
varlist = ['OH','HO2','C5H8','C10H16','NH3','NO','O3','SO2','CO','N10']

if make_dataset==0:
    varlist.append('U')
    varlist.append('V')
#varlist=['N10','N100']

is_dump=0
two_models=0

# job_2='cg502_cycled'#ca706 #for 10 May
# job_glm='ch081'
# job_glm='cg502_cycled' #in principle this should be the same as cj252 for 10 May, but it isn't. This one is used in the paper, except for dust
# job='cj252'#for both 10 and 25 May KORUSv5
# job='cj252_withdust'#for 10 May KORUSv5 with dust
# job_glm='cj252_withdust'

is_cycled=1
offline1=0
#u-co631 is the new rose suite in which I requested the dust stash output on 16th June
path_model='/jet/home/ding0928/cylc-run/u-cr977/share/cycle/20140729T0000Z/Regn1/resn_1/RA2M/um/'
path_glm = '/jet/home/ding0928/cylc-run/u-cr977/share/cycle/20140729T0000Z/glm/um/'
path= '/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/'

date= '20140731'
#here we also changed to a simple version from H's orignal, since he looked at different dates.
if date=='20140731':
    timeindexes_glm=['066','069','072']
    # timeindexes_glm=['114','117','120']
    timeindexes_lam=['066','072']
    # timeindexes_lam=['114','120'] #066+48h=114 FOR 0802
    lonmin=-107
    lonmax=-103.5
    latmin=38
    latmax=42
elif date=='20140802':
    # timeindexes_glm=['042','045','048','051']
    # timeindexes_lam=['042','048']
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
    df,df_smps = read_flight_data(date)

    conc_dict,smps_dict = load_model(timeindexes_lam, path_model,varlist,date,1, 0, 1,offline1)
    dict_of_interpolated_model_data= do_4d_interpolation(conc_dict,df, 1, 0,int(date[0:4]),int(date[4:6]),int(date[6:8]))
    dict_of_interpolated_smps_data = do_4d_interpolation(smps_dict,df_smps,1,0,int(date[0:4]),int(date[4:6]),int(date[6:8]))
    for key in dict_of_interpolated_model_data.keys():
        df['model_'+str(key)] = dict_of_interpolated_model_data[key]
    for key in dict_of_interpolated_smps_data.keys():
        df_smps['model_'+str(key)] = dict_of_interpolated_smps_data[key]

    conc_dict_glm,smps_dict_glm = load_model(timeindexes_glm,path_glm,varlist,date,0,is_dump, 1,0)
    dict_of_interpolated_glm_data= do_4d_interpolation(conc_dict_glm,df, 0, is_dump,int(date[0:4]),int(date[4:6]),int(date[6:8]))
    dict_of_interpolated_smps_glm_data = do_4d_interpolation(smps_dict_glm,df_smps,0,0,int(date[0:4]),int(date[4:6]),int(date[6:8]))
    for	key in dict_of_interpolated_glm_data.keys():
        df['glm_'+str(key)] = dict_of_interpolated_glm_data[key]
    for key in dict_of_interpolated_smps_glm_data.keys():
        df_smps['glm_'+str(key)] = dict_of_interpolated_smps_glm_data[key]
    print df_smps.head()

    df.to_csv(path+'interpolated_977_dataset_'+date+'.csv')
    df_smps.to_csv(path+'interpolated_977_smps_dataset_'+date+'.csv')
    
    sys.exit()
else:
    df = pd.read_csv(path+'interpolated_977_dataset_'+date+'.csv',index_col=0, parse_dates=True)
    if two_models==1:
        df2 = pd.read_csv(path+'interpolated_977_dataset_'+date+'.csv',index_col=0, parse_dates=True)
    else:
        df2=None
    # print df.head()
    if date=='20160510':
        df_surf = df.between_time(start_time='01:45', end_time='04:30')
        if two_models==1:
            df2_surf = df2.between_time(start_time='01:45', end_time='04:30')
    elif date=='20160526':    
        df_surf = df.between_time(start_time='03:40', end_time='05:25')
        if two_models==1:
            df2_surf = df2.between_time(start_time='03:40', end_time='05:25')
    if two_models==1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        
        labels = ['CO (ppbv)','O$_{3}$ (ppbv)','NO (pptv)','OH (pptv)','HO$_{2}$ (pptv)',
                  'SO$_{2}$ (pptv)','N > 10nm (cm$^{-3}$)','N > 100nm (cm$^{-3}$)']
        if date=='20160510':
            upperlimits=[350,80,1700,0.7,30,5000,12000,2500]
        elif date=='20160526':
            upperlimits=[250,120,150,1.2,60,500,6000,2500]
        lowerlimits=[100,50,0,0,0,0,0,0]
        keys=['CO','O3','NO','OH','HO2','SO2','N10','N100']
        stats=[]
        i=0
        for label in labels:
            fig.add_subplot(331+i)
            if i < 7:
                stats.append(plot_pdfs(df_surf,df2_surf,keys[i],lowerlimits[i],upperlimits[i],label))
            else:
                stats.append(plot_pdfs(df_surf,df2_surf,keys[i],lowerlimits[i],upperlimits[i],label,1))
            i=i+1
        plt.legend(bbox_to_anchor=(1.25, 1.0))
        ax.set_ylabel('Number of samples')
        i=0
        fig2=plt.figure()
        for label in labels:
            fig2.add_subplot(331+i)
            if i < 7:
	        plot_correlations(df_surf,df2_surf,keys[i],lowerlimits[i],upperlimits[i],label)
            else:
                plot_correlations(df_surf,df2_surf,keys[i],lowerlimits[i],upperlimits[i],label,1)
            i=i+1
                
        print(' & Obs. mean & Glob. mean & CMIP6 mean & KORUSv5 mean  \\\\')
        i=0
        for label in labels:
            print label,' & {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f}  \\\\'.format((stats[i])[0],(stats[i])[1],(stats[i])[2],(stats[i])[3]) 
            i=i+1
        i=0
        print(' & Obs. std & Glob. std & CMIP6 std & KORUSv5 std  \\\\')
        for label in labels:
	    print label,' & {0:.2f} & {1:.2f} & {2:.2f} & {3:.2f}  \\\\'.format((stats[i])[4],(stats[i])[5],(stats[i])[6],(stats[i])[7])
            i=i+1

    #plt.tight_layout()

    if sizedistrosOnly==0:
        conc_dict = load_model(timeindexes_lam, path_model,varlist,date,1,0,3,20,0,offline1) # second-last argument is model level number for 2d plot at cloud
        conc_dict_glm = load_model(timeindexes_glm, path_glm,varlist,date,0,0,1,13,0,0)
if sizedistrosOnly==0:
    print 'Pollutant & glob NMB & glob R & CMIP NMB & CMIP R & KORUSv5 NMB & KORUSv5 R \\\\'
    print '\hline'


    norm_array_so2 = [2,5,10,20,50,100,200,500,1000,2000,5000,10000]
    if 'SO2' in varlist:
        if date=='20160526':
            so2minflight = 0.1
            so2maxflight = 40000
            so2minmap = 20
            so2maxmap = 10000
            so2maxvp = 12000
        else:
            so2minflight = 2.0
            so2maxflight = 40000
            so2minmap = 2.0
            so2maxmap = 10000
            so2maxvp = 25000
        plot_comparison(df,'SO2',(conc_dict['SO2'])[tr,:,:],(conc_dict_glm['SO2'])[tg,:,:],
                        'SO$_{2}$ (pptv)',so2minflight,so2maxflight,so2minmap,so2maxmap,
                        True,norm_array_so2,'korus_so2_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        print('time_regional_index',conc_dict['SO2'].coord('time').points)
        print('time_global_index',conc_dict_glm['SO2'].coord('time').points)
        if 'U' in varlist:
            plot_wind((conc_dict['U'])[tr,:,:],(conc_dict['V'])[tr,:,:])
        plot_vertical(df,conc_dict['SO2'].coord('level_height'),'SO2',so2maxvp, 'SO$_{2}$ (pptv)','korus_so2_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)
    if 'OH' in varlist:
        norm_array_oh=[0,0.1,0.2,0.3,0.4,0.5]
        plot_comparison(df,'OH',(conc_dict['OH'])[tr,:,:],(conc_dict_glm['OH'])[tg,:,:],
                        'OH (pptv)', 0,1.2,0,0.6,False,norm_array_so2,'test_korus_oh_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['OH'].coord('level_height'),'OH',0.8, 'OH (pptv)','test_korus_oh_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)

    if 'HO2' in varlist:
    # norm_array_ho2=[0,5,10,20,30,40,50,60]
        if date=='20160526':
            ho2maxflight=60
            ho2maxvp=50
        else:
            ho2maxflight=30
            ho2maxvp=50
        plot_comparison(df,'HO2',(conc_dict['HO2'])[tr,:,:],(conc_dict_glm['HO2'])[tg,:,:],
                        'HO2 (pptv)', 0,ho2maxflight,0,30,False,norm_array_so2,'korus_ho2_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['HO2'].coord('level_height'),'HO2',ho2maxvp, 'HO2 (pptv)','korus_ho2_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)

    if 'JO1D' in varlist:
        if date=='20160526':
            jo1dmaxflight=10
            jo1dmaxvp=7
            jo1dmaxh=3
        else:
            jo1dmaxflight=6
            jo1dmaxvp=6
            jo1dmaxh=8

        plot_comparison(df,'JO1D',(conc_dict['JO1D'])[tr,:,:],(conc_dict_glm['JO1D'])[tg,:,:],
                        'J O(1D) (x$10^{-5}$s$^{-1}$)', 0,jo1dmaxflight,0,jo1dmaxh,False,norm_array_so2,'korus_jo1d_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['JO1D'].coord('level_height'),'JO1D',jo1dmaxvp, 'J O(1D) (x$10^{-5}$s$^{-1}$)','korus_jo1d_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)

    # #
    # norm_array_ho2ro2=[0,5,10,40,60,80,100,150]
    # plot_comparison(df,df['HO2RO2'],df['model_HO2RO2'],df['glm_HO2RO2'],(conc_dict[2])[2,:,:],(conc_dict_glm[2])[1,:,:],
    #                 'HO2RO2 (pptv)', 0,150,0,150,False,norm_array_ho2ro2,'korus_ho2ro2_'+date+'_'+utctime+'_'+job+'.png', date,df2)
    # plot_vertical(df,conc_dict[0].coord('level_height'),df['HO2RO2'],df['model_HO2RO2'],df['glm_HO2RO2'],150, 'HO2RO2 (pptv)','korus_ho2ro2_vp_'+date+'_'+utctime+'_'+job+'.png',two_models
    #
    #
    if 'T' in varlist:
        norm_array_temp=[250,255,260,265,270,275,280,285,290,295,300]
        print('MINIMUM INTERPOLATED TEMP:',np.min(np.asarray(df['glm_T'])))
        if is_dump:
            plot_comparison(df,'T',(conc_dict['T'])[tr,:,:],(conc_dict_glm['T'])[tg,:,:],
                            'T (K)', 260,310,280,310,False,norm_array_so2,'korus_T_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        else:
            print('plotting')
            plot_comparison(df,'T',(conc_dict['T'])[tr,:,:],(conc_dict_glm['T'])[tg,:,:],
                            'T (K)', 240,310,280,300,False,norm_array_so2,'korus_T_'+date+'_'+utctime+'_'+job+'.png', date,0,df2)
            print('done plotting')
        plot_vertical(df,conc_dict['T'].coord('level_height'),'T', 295, 'T (K)','korus_T_vp_'+date+'_'+utctime+'_'+job+'.png',0,df2,1,240)
    if 'RH' in varlist:
	norm_array_h2so4=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0]
        plot_comparison(df,'RH',(conc_dict['RH'])[tr,:,:],(conc_dict_glm['RH'])[tg,:,:],
	                'RH (%)', 0,100,0,100,False,norm_array_so2,'korus_h2so4_'+date+'_'+utctime+'_'+job+'.png', date,0,df2)
        plot_vertical(df,conc_dict['RH'].coord('level_height'),'RH',100.0, 'RH (%)','korus_RH_vp_'+date+'_'+utctime+'_'+job+'.png',0,df2)

    if 'H2SO4' in varlist:
        norm_array_h2so4=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0]
        plot_comparison(df,'H2SO4',(conc_dict['H2SO4'])[tr,:,:],(conc_dict_glm['H2SO4'])[tg,:,:],
                        'H2SO4 (pptv)', 0,1.5,0,1.0,False,norm_array_so2,'korus_h2so4_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['H2SO4'].coord('level_height'),'H2SO4',1.0, 'H2SO4 (pptv)','korus_h2so4_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)
    if 'NH3' in varlist:
        norm_array_nh3=[3,10,30,100,300,1000.0,3000.0,10000.0,30000.0]
        plot_comparison(df,'NH3',(conc_dict['NH3'])[tr,:,:],(conc_dict_glm['NH3'])[tg,:,:],
                        'NH3 (pptv)', 1,30000,20,30000,True,norm_array_nh3,'korus_nh3_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['NH3'].coord('level_height'),'NH3',20000, 'NH3 (pptv)','korus_nh3_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)
    if 'CO' in varlist:
        norm_array_co = [0,50,100,150,200,250,300,350]
        if date=='20160526':
            comaxflight=550
            comaxvp=600
        else:
            comaxflight=400
            comaxvp=300
        plot_comparison(df,'CO',(conc_dict['CO'])[tr,:,:],(conc_dict_glm['CO'])[tg,:,:],
                        'CO (ppbv)', 0,comaxflight,0,350,False,norm_array_so2,'test_korus_co_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        if 'U' in varlist:
            plot_wind((conc_dict['U'])[tr,:,:],(conc_dict['V'])[tr,:,:])
            plt.savefig('/jet/home/ding0928/box_model/data/overall_code/rerun_NH3_emission/neutralization/ground_obs_npf/0731_comparison/'+'test_korus_co_'+date+'_'+utctime+'_'+job+'.png')
        plot_vertical(df,conc_dict['CO'].coord('level_height'),'CO',comaxvp, 'CO (ppbv)','test_korus_co_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)

    if 'C5H8' in varlist:
        if date=='20160526':
            ipmaxvp=500
        else:
            ipmaxvp = 350
            
        norm_array_c5h8=[0.01,0.03,1.0,3.0,10,30,100,300,1000.0]
        plot_comparison(df,'C5H8',(conc_dict['C5H8'])[tr,:,:],(conc_dict_glm['C5H8'])[tg,:,:],
                        'C5H8 (pptv)',1,600,0.01,3000.0,False,norm_array_so2,'korus_c5h8_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['C5H8'].coord('level_height'),'C5H8',ipmaxvp, 'C5H8 (pptv)','korus_c5h8_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)

        # norm_array_c10h16=[0,2,4,6,8,10,12,14,16,18,20]
        # plot_comparison(df,df['C10H16'],df['model_C10H16'],df['glm_C10H16'],(conc_dict[4])[2,:,:],(conc_dict_glm[4])[1,:,:],
        #                 'C10H16 (pptv)', 1,140,0.01,140.0,True,norm_array_c10h16,'korus_c10h16_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        # print conc_dict_glm[3].coord('level_height')[0:30]
    if 'NO' in varlist:
        norm_array_no = [5,10,20,50,100,200,300,500,700,1000,1500,2500]
        if date=='20160526':
            nomaxvp=6000
        else:
            nomaxvp = 3000
        plot_comparison(df,'NO',(conc_dict['NO'])[tr,:,:],(conc_dict_glm['NO'])[tg,:,:],
                        'NO (pptv)', 5,10000,20,5000,True,norm_array_so2,'test_korus_no_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['NO'].coord('level_height'),'NO',nomaxvp, 'NO (pptv)','test_korus_no_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)
        
    if 'O3' in varlist:
        if date=='20160526':
            o3maxvp=140
            o3maxflight=140
            o3maxmap=90
        else:
            o3maxvp = 120
            o3maxflight=120
            o3maxmap=70
        norm_array_o3=[0,0.1,0.2,0.3,0.4,0.5,1,10,20,50,70,80,100]
        plot_comparison(df,'O3',(conc_dict['O3'])[tr,:,:],(conc_dict_glm['O3'])[tg,:,:],'O3 (ppbv)',
                        30,o3maxflight,30,o3maxmap,False,norm_array_so2,'korus_O3_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        if 'U' in varlist:
            plot_wind((conc_dict['U'])[tr,:,:],(conc_dict['V'])[tr,:,:])
        plot_vertical(df,conc_dict['O3'].coord('level_height'),'O3',120, 'O3 (ppbv)','korus_O3_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)
        
    if 'N10' in varlist:
        norm_array_num= [500,1000,2000,5000,10000,20000,50000]
        plot_comparison(df,'N10',(conc_dict['N10'])[tr,:,:],(conc_dict_glm['N10'])[tg,:,:],
                        'N10 (cm-3 stp)', 500,50000,500,50000.0,True,norm_array_so2,'korus_n10_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['N10'].coord('level_height'),'N10',14000, 'N10 (cm-3 stp)','korus_n10_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)
    if 'N100' in varlist:
        if date=='20160526':
            n100maxvp=9000
        else:
            n100maxvp=4000
        norm_array_num= [50,100,200,500,1000,2000,5000,10000]
        plot_comparison(df,'N100',(conc_dict['N100'])[tr,:,:],(conc_dict_glm['N100'])[tg,:,:],
                        'N100 (cm-3 stp)', 50,20000,300,10000.0,True,norm_array_so2,'korus_n100_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_vertical(df,conc_dict['N100'].coord('level_height'),'N100',n100maxvp, 'N100 (cm-3 stp)','korus_n100_vp_'+date+'_'+utctime+'_'+job+'.png',two_models,df2)

    if 'CDNC' in varlist:
        norm_array_cdnc = [0,100,200,300,400,500,600,700,800,900,1000]

        plot_comparison(df,'CDNC',(conc_dict['CDNC'])[tr,:,:],(conc_dict_glm['CDNC'])[tg,:,:],
                        'CDNC (cm-3 amb)', 0,1200,0,1000.0,False,norm_array_so2,'korus_cdnc_'+date+'_'+utctime+'_'+job+'.png', date,two_models,df2)
        plot_comparison(df,'LWC',(conc_dict['LWC'])[tr,:,:],(conc_dict_glm['LWC'])[tg,:,:],
                        'LWC (g/kg)', 0,0.1,0,0.1,False,norm_array_so2,'korus_lwc_'+date+'_'+utctime+'_'+job+'.png',date,two_models,df2)


df_smps = pd.read_csv(path+'interpolated_977_smps_dataset_'+date+'.csv',index_col=0, parse_dates=True)
# print(df_smps)
plt.figure()
df_smps['nSMPS_stdPT'].plot(label='nSMPS_obs')
df_smps['model_N'] = df_smps['model_AitN']+df_smps['model_AInsN']+df_smps['model_AccN']
df_smps['model_N'].plot(label='n_model')
# print(df_smps)
plt.legend()
df_smps_1 = df_smps.between_time(start_time='19:00', end_time='19:10').mean() #why this time priod?
plot_sizedistribution(df_smps_1,'977_13:00-13:10.png')
df_smps_2 = df_smps.between_time(start_time='21:10', end_time='21:30').mean()
plot_sizedistribution(df_smps_2,'977_15:10-15:30.png')
df_smps_3 = df_smps.between_time(start_time='20:40', end_time='20:50').mean()
plot_sizedistribution(df_smps_3,'977_14:40-14:50.png')
df_smps_4 = df_smps.between_time(start_time='22:00', end_time='22:20').mean()
plot_sizedistribution(df_smps_4,'977_16:00-16:20.png')
df_smps_5 = df_smps.between_time(start_time='22:50', end_time='23:00').mean()
plot_sizedistribution(df_smps_5,'977_16:50-17:00.png')
df_smps_6 = df_smps.between_time(start_time='19:30', end_time='19:40').mean() #why this time priod?
plot_sizedistribution(df_smps_6,'977_13:30-13:40.png')


plt.show()
