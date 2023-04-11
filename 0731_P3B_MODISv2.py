import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from datetime import datetime
import matplotlib.colors as cols
import matplotlib.cm as cmx
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import LineCollection
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import glob
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.img_tiles import OSM
import os, sys

# day=31
os.environ["CARTOPY_USER_BACKGROUNDS"] = "/Users/hamish/scripts/cartopy/BG/"
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates

cnindex = 0

def read_nav(day, month, campaign, label_on_plot):
    # This part reads in the aircraft data
    if campaign == 0:  # campaign0
        if day < 10:
            files = ['/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/discoveraq-reveal_p3b_20140731_r0'
                     '.ict']
            dateiso = '2014' + '-0' + str(month) + '-0' + str(day)
        else:  # day31
            files = ['/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/discoveraq-reveal_p3b_20140731_r0'
                     '.ict']
            dateiso = '2014' + '-0' + str(month) + '-' + str(day)
        cn_p3 = 0
        # cn_files = ['/Users/hamish/Downloads/WWW-AIR_1527607926/DISCOVERAQ-LARGE-CAS_P3B_20140731_R0.ict']
        headerlen = 73
        headerlen2 = 100
        # month=7
        year = 2014
        print(files)
    cn_timeseries = []
    cas_n_timeseries = []
    alt_timeseries = []
    lat_timeseries = []
    lon_timeseries = []
    utctime = []  # seconds from midnight on July 29 2014
    utctime2 = []
    for runfile in files:

        iline = 0
        with open(runfile) as fp:
            line = fp.readline()
            while line:
                if iline < headerlen:  # length of header
                    line = fp.readline()
                    iline += 1
                    continue
                line = fp.readline()
                data = line.split(',')
                # print(iline)
                if campaign == 0 and iline == 180:
                    global cnindex

                    iname = 0
                    for name in data:
                        if name == 'CONCN':
                            cnindex = iname
                            break
                        iname += 1
                    # print(iname)
                else:
                    try:
                        utctime.append(float(data[0]))
                    except Exception:
                        break
                    if campaign == 0:
                        cn_timeseries.append(float(data[cnindex]))
                        alt_timeseries.append(float(data[4]))
                        lat_timeseries.append(float(data[2]))
                        lon_timeseries.append(float(data[3]))
                if iline % 500 == 0:
                    print('i=1')
                iline += 1
    print('done first file')
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
    utcdatetimes = [datetime(year, months[i], days[i], hours[i], minutes[i], seconds[i]) for i in
                    range(0, len(utctime))]
    utcdatetimes2 = [datetime(year, months2[i], days2[i], hours2[i], minutes2[i], seconds2[i]) for i in
                     range(0, len(utctime2))]
    # Now make a plot.
    plt.figure(figsize=(12, 12))

    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plt.text(-104.99, 39.48, '*Denver', fontsize=22, color='red')
    plt.text(-105.48, 39.38, '*Colorado', fontsize=22, color='red')
    plt.text(-105.00, 40.05, '*Ground Mobile', fontsize=22, color='red')
    ax.coastlines(resolution='50m', linewidth=2)
    ax.set_title("P-3B aircraft track", fontsize=18)

    # This part gets the MODIS data
    ax.add_wmts('https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
                'MODIS_Aqua_CorrectedReflectance_TrueColor', wmts_kwargs={'time': dateiso})
    cmap = plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    # Altitude color scale
    norm = BoundaryNorm([500, 1e3, 1.5E3, 2e3, 2.5e3, 3e3, 3.5e3, 4e3, 4.5e3, 5e3], cmap.N, clip=False)
    # The flight track
    points = np.array([lon_timeseries, lat_timeseries]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles='solid')
    lc.set_array(np.asarray(alt_timeseries))
    lc.set_linewidth(4)
    ax.add_collection(lc)
    # x, y axis ranges
    if campaign == 0:
        xmin1 = -106.5
        xmax1 = -103.5
        ymin = 39.0
        ymax = 41.0
    ax.set_xlim(xmin1, xmax1)
    ax.set_ylim(ymin, ymax)

    # Timing labels
    if label_on_plot == 1:
        # plt.text(123.5, 37.5,str(days[i])+'/'+str(month)+'/16', fontsize=16)
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
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    if campaign == 2:
        gl.xlocator = mticker.FixedLocator([xmin1, xmin1 + 2, xmin1 + 4, xmin1 + 6, xmin1 + 8, xmin1 + 10, xmin1 + 12])
        gl.ylocator = mticker.FixedLocator([ymin, ymin + 2, ymin + 4, ymin + 6, ymin + 8])
    elif campaign == 3:
        gl.xlocator = mticker.FixedLocator(
            [xmin1, xmin1 + 2, xmin1 + 4, xmin1 + 6, xmin1 + 8, xmin1 + 10, xmin1 + 12, xmin1 + 14, xmin1 + 16])
        gl.ylocator = mticker.FixedLocator(
            [ymin, ymin + 2, ymin + 4, ymin + 6, ymin + 8, ymin + 10, ymin + 12, ymin + 14])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    cb = plt.colorbar(lc, label='Altitude (m)', extend="max", shrink=0.7)
    cb.set_label(label='Altitude (m)', fontsize=18)
    cb.ax.tick_params(labelsize=18)
    plt.savefig("0731p3b_flighttrackd_nonBAO.jpg", dpi=1500)


def read_nav_segmentation(day, month, campaign, label_on_plot):
    # This part reads in the aircraft data
    if campaign == 0:  # campaign0
        if day < 10:
            files = ['/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/discoveraq-reveal_p3b_20140731_r0'
                     '.ict']
            dateiso = '2014' + '-0' + str(month) + '-0' + str(day)
        else:  # day31
            files = ['/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/discoveraq-reveal_p3b_20140731_r0'
                     '.ict']
            dateiso = '2014' + '-0' + str(month) + '-' + str(day)
        cn_p3 = 0
        # cn_files = ['/Users/hamish/Downloads/WWW-AIR_1527607926/DISCOVERAQ-LARGE-CAS_P3B_20140731_R0.ict']
        headerlen = 73
        headerlen2 = 100
        # month=7
        year = 2014
        print(files)
    cn_timeseries = []
    cas_n_timeseries = []
    alt_timeseries = []
    lat_timeseries = []
    lon_timeseries = []
    utctime = []  # seconds from midnight on July 29 2014
    utctime2 = []
    for runfile in files:

        iline = 0
        with open(runfile) as fp:
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
                    # print(data)
                else:
                    try:
                        utctime.append(float(data[0]))
                    except Exception:
                        break
                    if campaign == 0:
                        cn_timeseries.append(float(data[cnindex]))
                        alt_timeseries.append(float(data[4]))
                        lat_timeseries.append(float(data[2]))
                        lon_timeseries.append(float(data[3]))
                if iline % 500 == 0:
                    print('i=1')
                iline += 1
    print('done first file')
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
    utcdatetimes = [datetime(year, months[i], days[i], hours[i], minutes[i], seconds[i]) for i in
                    range(0, len(utctime))]
    utcdatetimes2 = [datetime(year, months2[i], days2[i], hours2[i], minutes2[i], seconds2[i]) for i in
                     range(0, len(utctime2))]
    # Now make a plot.
    plt.figure(figsize=(12, 12))

    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plt.text(-104.99, 39.74, '*Denver', fontsize=22, color='red')
    plt.text(-104.7, 40.2, '*Ground Mobile', fontsize=24, color='red')
    plt.text(-105.00, 40.03, '*BAO Tower', fontsize=22, color='red')

    ax.coastlines(resolution='50m', linewidth=2)

    # This part gets the MODIS data
    ax.add_wmts('https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
                'MODIS_Aqua_CorrectedReflectance_TrueColor', wmts_kwargs={'time': dateiso})
    cmap = plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    norm = BoundaryNorm([500, 1e3, 1.5E3, 2e3, 2.5e3, 3e3, 3.5e3, 4e3, 4.5e3, 5e3], cmap.N, clip=False)
    # The flight track
    points = np.array([lon_timeseries, lat_timeseries]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles='solid')
    lc.set_array(np.asarray(alt_timeseries))
    lc.set_linewidth(5)
    if campaign == 0:
        xmin1 = -106.5
        xmax1 = -103.5
        ymin = 39.0
        ymax = 41.0

    ax.set_xlim(xmin1, xmax1)
    ax.set_ylim(ymin, ymax)

    print('len(hours)', len(hours)) 

    if label_on_plot == 1:
        for itime in range(0, len(hours) - 3600):
            if minutes[itime] == 0 and seconds[itime] == 0:
                # print('minutes:', minutes)
                if ((hour >= 12) and (hour <= 20) for hour in hours):
                    j = 1

                    plt.figure(figsize=(12, 12))
                    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
                    plt.text(-104.99, 39.74, '*Denver', fontsize=22, color='red')
                    plt.text(-104.7, 40.2, '*Ground Mobile', fontsize=24, color='red')
                    plt.text(-105.00, 40.03, '*BAO Tower', fontsize=22, color='red')

                    ax.coastlines(resolution='50m', linewidth=2)
                    # This part gets the MODIS data
                    ax.add_wmts('https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
                                'MODIS_Aqua_CorrectedReflectance_TrueColor', wmts_kwargs={'time': dateiso})
                    cmap = plt.get_cmap('jet')
                    cmap.set_under('w')
                    cmap.set_over('k')
                    # Altitude color scale
                    norm = BoundaryNorm([500, 1e3, 1.5E3, 2e3, 2.5e3, 3e3, 3.5e3, 4e3, 4.5e3, 5e3],
                                        cmap.N, clip=False)

                    points = np.array(
                        [lon_timeseries[itime + (3600 * (j - 1)):itime + (3600 * j)],
                         lat_timeseries[itime + (3600 * (j - 1)):itime + (3600 * j)]]).T.reshape(
                        -1,
                        1,
                        2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles='solid')
                    lc.set_array(np.asarray(alt_timeseries))
                    lc.set_linewidth(4)
                    ax.add_collection(lc)
                    # x, y axis ranges
                    if campaign == 0:
                        xmin1 = -106.5
                        xmax1 = -103.5
                        ymin = 39.0
                        ymax = 41.0

                    ax.set_xlim(xmin1, xmax1)
                    ax.set_ylim(ymin, ymax)

                    plt.plot(lon_timeseries[minutes[itime]], lat_timeseries[minutes[itime]], marker='|', markersize=0,
                             color='red')
                    if hours[itime] < 10:
                        timestring = '0' + str(hours[itime]) + ':00'
                    else:
                        timestring = str(hours[itime]) + ':00'
                        print('hours[itime]', hours[itime])
                        timestring2 = str(hours[itime + 3600]) + ':00'
                    if hours[itime] != 4:
                        plt.text(lon_timeseries[itime], lat_timeseries[itime], timestring, fontsize=19,
                                 color='yellow')
                        plt.text(lon_timeseries[itime + 3600], lat_timeseries[itime + 3600], timestring2, fontsize=19,
                                 color='yellow')

                    ax.coastlines(resolution='50m', linewidth=2)
                    # Grid-lines
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
                    gl.xlabels_top = False
                    gl.ylabels_right = False
                    if campaign == 2:
                        gl.xlocator = mticker.FixedLocator(
                            [xmin1, xmin1 + 2, xmin1 + 4, xmin1 + 6, xmin1 + 8, xmin1 + 10, xmin1 + 12])
                        gl.ylocator = mticker.FixedLocator([ymin, ymin + 2, ymin + 4, ymin + 6, ymin + 8])
                    elif campaign == 3:
                        gl.xlocator = mticker.FixedLocator(
                            [xmin1, xmin1 + 2, xmin1 + 4, xmin1 + 6, xmin1 + 8, xmin1 + 10, xmin1 + 12, xmin1 + 14,
                             xmin1 + 16])
                        gl.ylocator = mticker.FixedLocator(
                            [ymin, ymin + 2, ymin + 4, ymin + 6, ymin + 8, ymin + 10, ymin + 12, ymin + 14])
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER
                    gl.xlabel_style = {'size': 18}
                    gl.ylabel_style = {'size': 18}
                    cb = plt.colorbar(lc, label='Altitude (m)', extend="max", shrink=0.7)
                    cb.set_label(label='Altitude (m)', fontsize=18)
                    cb.ax.tick_params(labelsize=18)
                    j = j + 1
                    plt.savefig("p3b%d.jpg" % j)


def read_nav_lasthour(day, month, campaign, label_on_plot):
    # This part reads in the aircraft data
    if campaign == 0:  # campaign0
        if day < 10:
            files = ['/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/discoveraq-reveal_p3b_20140731_r0'
                     '.ict']
            dateiso = '2014' + '-0' + str(month) + '-0' + str(day)
        else:  # day31
            files = ['/jet/home/ding0928/box_model/data/overall_code/ground_ob_data/discoveraq-reveal_p3b_20140731_r0'
                     '.ict']
            dateiso = '2014' + '-0' + str(month) + '-' + str(day)
        cn_p3 = 0
        # cn_files = ['/Users/hamish/Downloads/WWW-AIR_1527607926/DISCOVERAQ-LARGE-CAS_P3B_20140731_R0.ict']
        headerlen = 73
        headerlen2 = 100
        year = 2014
        print(files)
    cn_timeseries = []
    cas_n_timeseries = []
    alt_timeseries = []
    lat_timeseries = []
    lon_timeseries = []
    utctime = []  # seconds from midnight on July 29 2014
    utctime2 = []
    for runfile in files:

        iline = 0
        with open(runfile) as fp:
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
                    # print(data)
                else:
                    try:
                        utctime.append(float(data[0]))
                    except Exception:
                        break
                    if campaign == 0:
                        cn_timeseries.append(float(data[cnindex]))
                        alt_timeseries.append(float(data[4]))
                        lat_timeseries.append(float(data[2]))
                        lon_timeseries.append(float(data[3]))
                if iline % 500 == 0:
                    print('i=1')
                iline += 1
    print('done first file')
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
    utcdatetimes = [datetime(year, months[i], days[i], hours[i], minutes[i], seconds[i]) for i in
                    range(0, len(utctime))]
    utcdatetimes2 = [datetime(year, months2[i], days2[i], hours2[i], minutes2[i], seconds2[i]) for i in
                     range(0, len(utctime2))]
    # Now make a plot.
    plt.figure(figsize=(12, 12))

    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=2)

    # This part gets the MODIS data
    ax.add_wmts('https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
                'MODIS_Aqua_CorrectedReflectance_TrueColor', wmts_kwargs={'time': dateiso})
    cmap = plt.get_cmap('jet')
    cmap.set_under('w')
    cmap.set_over('k')
    # Altitude color scale
    norm = BoundaryNorm([500, 1e3, 1.5E3, 2e3, 2.5e3, 3e3, 3.5e3, 4e3, 4.5e3, 5e3], cmap.N, clip=False)
    # The flight track
    points = np.array([lon_timeseries, lat_timeseries]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles='solid')
    lc.set_array(np.asarray(alt_timeseries))
    lc.set_linewidth(5)

    if campaign == 0:
        xmin1 = -106.5
        xmax1 = -103.5
        ymin = 39.0
        ymax = 41.0
    ax.set_xlim(xmin1, xmax1)
    ax.set_ylim(ymin, ymax)

    print('len(hours)', len(hours))  # 19019
    # print('hours.data', hours)  # 19019

    if label_on_plot == 1:
        for itime in range(0, len(hours)):  # loop for # 19019 times.
            if minutes[itime] == 0 and seconds[itime] == 0:
                # print('minutes:', minutes)
                if ((hour >= 16) and (hour <= 17) for hour in hours):
                    k = 1
                    print('len(hour)', hour)  # totally 6 hours
                    print('len(hours)', len(hours))  # totally 6 hours
                    print('hours[itime]', hours[itime])
                    print('lon_timeseries[itime]', lon_timeseries[itime])
                    print('itime', itime)
                    print(i)

                    plt.figure(figsize=(12, 12))
                    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
                    # plt.plot(-105, 40, marker='o')
                    plt.text(-104.99, 39.74, '*Denver', fontsize=22, color='red')
                    # plt.text(-104.7, 40.42, '*G', fontsize=22, color='red')
                    # plt.text(-105.49, 39.38, '*F', fontsize=22, color='red')
                    plt.text(-104.7, 40.2, '*Ground Mobile', fontsize=24, color='red')
                    plt.text(-105.00, 40.03, '*BAO Tower', fontsize=22, color='red')
                    # plt.title('MODIS AQUA Reflectance imagery  with the flight path on 02/08/14')

                    ax.coastlines(resolution='50m', linewidth=2)
                    # This part gets the MODIS data
                    ax.add_wmts('https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi',
                                'MODIS_Aqua_CorrectedReflectance_TrueColor', wmts_kwargs={'time': dateiso})
                    cmap = plt.get_cmap('jet')
                    cmap.set_under('w')
                    cmap.set_over('k')
                    # Altitude color scale
                    norm = BoundaryNorm([500, 1e3, 1.5E3, 2e3, 2.5e3, 3e3, 3.5e3, 4e3, 4.5e3, 5e3],
                                        cmap.N, clip=False)

                    points = np.array(
                        [lon_timeseries[itime + (3600 * (k - 1)):itime + (3600 * k)],
                         lat_timeseries[itime + (3600 * (k - 1)):itime + (3600 * k)]]).T.reshape(
                        -1,
                        1,
                        2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles='solid')
                    lc.set_array(np.asarray(alt_timeseries))
                    lc.set_linewidth(4)
                    ax.add_collection(lc)
                    # x, y axis ranges
                    if campaign == 0:
                        xmin1 = -106.5
                        xmax1 = -103.5
                        ymin = 39.0
                        ymax = 41.0

                    ax.set_xlim(xmin1, xmax1)
                    ax.set_ylim(ymin, ymax)
                    # ax.set_title("MODIS AQUA Reflectance imagery  with the flight path on 02/08/14", fontsize=20)
                    # ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())

                    # Timing labels
                    # if label_on_plot == 1:
                    # # plt.text(123.5, 37.5,str(days[i])+'/'+str(month)+'/16', fontsize=16)
                    # for itime in range(0, len(hours)):
                    #     if minutes[itime] == 0 and seconds[itime] == 0:
                    plt.plot(lon_timeseries[minutes[itime]], lat_timeseries[minutes[itime]], marker='|',
                             markersize=0,
                             color='red')
                    if hours[itime] < 10:
                        timestring = '0' + str(hours[itime]) + ':00'
                    else:
                        timestring = str(hours[itime]) + ':00'
                        print('hours[itime]', hours[itime])
                        # timestring2 = str(hours[itime + 1]) + ':00'
                    if hours[itime] != 4:
                        plt.text(lon_timeseries[itime], lat_timeseries[itime], timestring, fontsize=19,
                                 color='yellow')
                        # plt.text(lon_timeseries[itime + 3600], lat_timeseries[itime + 3600], timestring2,
                        #          fontsize=19,
                        #          color='yellow')

                    ax.coastlines(resolution='50m', linewidth=2)
                    # Grid-lines
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
                    gl.xlabels_top = False
                    gl.ylabels_right = False
                    if campaign == 2:
                        gl.xlocator = mticker.FixedLocator(
                            [xmin1, xmin1 + 2, xmin1 + 4, xmin1 + 6, xmin1 + 8, xmin1 + 10, xmin1 + 12])
                        gl.ylocator = mticker.FixedLocator([ymin, ymin + 2, ymin + 4, ymin + 6, ymin + 8])
                    elif campaign == 3:
                        gl.xlocator = mticker.FixedLocator(
                            [xmin1, xmin1 + 2, xmin1 + 4, xmin1 + 6, xmin1 + 8, xmin1 + 10, xmin1 + 12, xmin1 + 14,
                             xmin1 + 16])
                        gl.ylocator = mticker.FixedLocator(
                            [ymin, ymin + 2, ymin + 4, ymin + 6, ymin + 8, ymin + 10, ymin + 12, ymin + 14])
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER
                    gl.xlabel_style = {'size': 18}
                    gl.ylabel_style = {'size': 18}
                    cb = plt.colorbar(lc, label='Altitude (m)', extend="max", shrink=0.7)
                    cb.set_label(label='Altitude (m)', fontsize=18)
                    cb.ax.tick_params(labelsize=18)
                    k = k + 1
                    # plt.show()

    plt.show()

read_nav(31, 7, 0, 1)
read_nav_segmentation(31, 7, 0, 1)
# read_nav_lasthour(31, 7, 0, '0731_C130_lasthour.png', '0731_lasthour.png', 1)
read_nav_lasthour(31, 7, 0, 1)
