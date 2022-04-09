#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 1/20/2017
# Last edit 1/20/2017

# Purpose: To convert a goven AOD and ANG gloabl image into a RGB decompositon image. This technique follows from Fujikawa and Takeuchi. It is implemented as foloows
# (1) : Call the monthly image generator for AOD and ANG
# (2) : Create a function for entering (AOD, ANG) and receiving back (R, G, B)
# (3) : Find the vertices of the RGB triangle


# Location of output: E:\Acads\Research\AQM\Data process\CSVOut

# terminology used: T -  target resolution to be achieved. usually MODIS, OMI image; S - source of the image to be resampled
'''# output filenames produced

'''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os.path
from glob import glob
from datetime import timedelta, date
from dateutil import rrule
from PIL import Image
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
import matplotlib.patches as patches
from itertools import product
from adjustText import adjust_text

# pvt imports

import infoFinder as info
import shp_rstr_stat as srs
from classRaster import Raster_file
import im_mean_temporal as im_mean
import coord_translate as ct



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
#Input

gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path from my folders
gb_path_wt = '/home/wataru/research/MODIS/MOD04L2//'   # global path to be appended for all files to be sourced from wt

aod = Raster_file()
aod.path = gb_path_wt + r'/L3//'
aod.sat = 'MODIS'
aod.prod = 'AOD'
aod.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'MOD04L2.A201511.AOD.Global'
# > georef needs to be updated as per the dataset used
aod.georef = gb_path + r'/Data/Data_process/Georef_img//MODIS_global_georef.tif'

ang = Raster_file()
ang.path = gb_path_wt + r'/L3//'
ang.sat = 'MODIS'
ang.prod = 'ANG'
ang.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'MOD04L2.A201511.AOD.Global'
# > georef needs to be updated as per the dataset used
ang.georef = gb_path + r'/Data/Data_process/Georef_img//MODIS_global_georef.tif'

no2 = Raster_file()
no2.path = '/home/wataru/research/OMI/L2G//'
no2.sat = 'OMI'
no2.prod = 'NO2'
no2.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'OMI.NO2.201412.min.Global'
no2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_global_georef.tif'

so2 = Raster_file()
so2.path = '/home/wataru/research/OMI/L2G//'
so2.sat = 'OMI'
so2.prod = 'SO2'
so2.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'OMI.SO2.201412.min.Global'
so2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_global_georef.tif'


gis = Raster_file()
gis.path = gb_path + '/Data/Data_raw/GIS data/' + 'GlobalLandCover_tif/'
gis.sat = 'GIS'
gis.prod = 'LCType'
gis.sample = gb_path + '/Data/Data_raw/GIS data/GlobalLandCover_tif/LCType.tif' + 'LCType.tif'
gis.georef = gb_path + '/Data/Data_raw/GIS data/GlobalLandCover_tif/LCType.tif' + 'LCType.tif'

#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step 0: Display all cities       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# (a) Display map
def display_map():

    global fig
    fig = plt.figure()
    map = Basemap(projection='cyl',lat_0=0, lon_0=0,
    resolution = 'l', area_thresh = 1000.0
        # llcrnrlon=60, llcrnrlat=5,
        # urcrnrlon=100, urcrnrlat=40
                  ) # using Cylindrical Equidistant projection.; map coordinates are shown in lat and long

    # Fill the globe with a blue color
    # map.drawmapboundary(fill_color='aqua')

    # Fill the continents with the land color
    # map.fillcontinents(color='coral',lake_color='aqua')
    # map.drawcountries()

    # map.drawcoastlines()
    # map.bluemarble()
    # map.drawlsmask(land_color = 'coral', ocean_color='white' ,lakes = False)
    map.shadedrelief()

    # city locs
    city = pd.read_csv('RGB/global_megacity2.csv', header=0)
    x,y = map(city.lon, city.lat)

    # map.plot(x,y,'ro', markersize=5)
    for label, size, xpt, ypt in zip(city.city, city.vehicle, x, y):
        map.plot(xpt, ypt, 'ro', markersize=int(size/1000000)*3, alpha=0.8)
        # plt.text(xpt, ypt, label, bbox = {'boxstyle' : 'round','facecolor':'red', 'alpha':0.3, 'pad':0.3})
        # plt.annotate(label, xy=(xpt, ypt), xytext = (0,0), bbox = dict(boxstyle = 'round', color ='k' ))

    ls =[]
    for xt, yt, s in zip (x,y, city.city):
        ls.append(plt.text (xt, yt, s))

    adjust_text(ls, arrowprops = dict(arrowstyle = "->", color = 'r', lw = 0.5), autoalign = 'y',
                only_move={'points':'y', 'text':'y'},
                expand_points=(1.2, 1.75),
                force_points=0.1)
    plt.show()
    return fig
# function end

#Function run ----




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step I: Generate Monthly image       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

start_date = date(2001, 1, 1)
end_date = date(2017, 1, 1)

a = 0
for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
    start_date = date_m

    # finding mean clena images between the range of dates for AOD and ANG and saving them
    arr_aod = im_mean.mean_month(aod, start_date, start_date + relativedelta(months=1), minmax=False)
    # print "done aod"
    arr_ang = im_mean.mean_month(ang, start_date, start_date + relativedelta(months=1), minmax=False)
    # arr_no2, arr_no2min, arr_no2max = im_mean.mean_month(no2, start_date, start_date + relativedelta(months=1), minmax=True)
    # arr_so2, arr_so2min, arr_so2max = im_mean.mean_month(so2, start_date, start_date + relativedelta(months=1), minmax=True)

    # save array as raster image
    srs.arr_to_raster(arr_aod, aod.georef, img_save_path+ '/GlobalMean/MOD04L2.A'+ str(start_date.year) + str('%02d'%start_date.month)+'.AOD.Global.tif')
    srs.arr_to_raster(arr_ang, aod.georef, img_save_path + '/GlobalMean/MOD04L2.A' + str(start_date.year) + str(
        '%02d' % start_date.month) + '.ANG.Global.tif')
    # srs.arr_to_raster(arr_no2, no2.georef, img_save_path+ '/GlobalMean/OMI.NO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.ave.Global.tif')
    # srs.arr_to_raster(arr_no2min, no2.georef, img_save_path+ '/GlobalMean/OMI.NO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.min.Global.tif')
    # srs.arr_to_raster(arr_no2max, no2.georef, img_save_path+ '/GlobalMean/OMI.NO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.max.Global.tif')
    # srs.arr_to_raster(arr_so2, so2.georef, img_save_path+ '/GlobalMean/OMI.SO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.ave.Global.tif')
    # srs.arr_to_raster(arr_so2min, so2.georef, img_save_path+ '/GlobalMean/OMI.SO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.min.Global.tif')
    # srs.arr_to_raster(arr_so2max, so2.georef, img_save_path+ '/GlobalMean/OMI.SO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.max.Global.tif')
    #

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step II: RGB triangle parameters       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Open each mean AOD ANG image and save values for each city from each year/month

aod.path = gb_path + r'/Data/Data_process/GlobalMean//'
ang.path = gb_path + r'/Data/Data_process/GlobalMean//'
no2.path = gb_path + r'/Data/Data_process/GlobalMean//'

aod.path = '/home/wataru/research/MODIS/MOD04L2/L4/'
ang.path = '/home/wataru/research/MODIS/MOD04L2/L4/'


def pixval(arr_prod):

    ls = []
    for pix in city.pixloc:
        ls.append(arr_prod[pix[1], pix[0]])
    return ls

def allpixval(prodT):
    start_date = date(2001, 1, 1)
    end_date = date(2017, 1, 10)

    # Getting pixel values from lat/long values
    # input>>lat, long   output >>pix_y, pix_x
    city['pixloc'] = ct.latLonToPixel(prodT.georef,
                                      city.apply(lambda x: [x['lat'], x['lon']],axis=1).tolist())

    for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
        for prodT.file in glob(os.path.join(prodT.path, '*.' + prodT.prod + '*' + '.Global.tif')):

            # dthis gives ate as a list with integer year and month as return
            date_ym = int(prodT.file[-21:-15]) # MODIS, OMI both
            # print date_ym

            # if (date_y_d == [date_d.year,date_d.timetuple().tm_yday ]) & os.path.isfile(prodT.file):
            #  [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
            if ((str(date_ym) == (str(date_m.year) + str('%02d'%date_m.month))) & os.path.isfile(prodT.file)):

                print 'file found ', date_ym
                try:
                    prod_arr = prodT.raster_as_array()
                    city[prodT.prod + str(date_m.year) + str('%02d'%date_m.month)] = pixval(prod_arr)


                except AttributeError:
                    print ' seems like missing file  in ', date_m

    city.to_csv(csv_save_path + 'df_GlobalcityWT_'+prodT.prod+'.csv', index=False, header=True)

# function end

# Function run ---- find aod, ang alues for global cities
city = pd.read_csv('RGB/global_megacity2.csv', header=0)
allpixval(aod)
allpixval(ang)
allpixval(no2)



# Find parameters by plotting the mean annual for aod and ang
def rgb_param():
    year = 20
    city = pd.read_csv('RGB/global_megacity2.csv', header=0)
    city['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
                                                             axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    aodval = pd.read_csv(csv_save_path + 'df_Globalcity_AOD.csv', header=0)
    angval = pd.read_csv(csv_save_path + 'df_Globalcity_ANG.csv', header=0)

    # get rid of zero as they skew the mean
    aodval = aodval.replace(0, np.NaN)
    angval = angval.replace(0, np.NaN)

    # [col for col in aodval.columns if str(year) in col]
    # Mean
    year = 20
    city['AOD' + str(year)] = aodval.filter(regex=str(year)).mean(axis=1)
    city['ANG' + str(year)] = angval.filter(regex=str(year)).mean(axis=1)



    # mean at annual level and finding the max of it
    city['AODmaxmean'] = np.max(map(lambda x: aodval.filter(regex=str(x)).mean(axis=1), range(2001, 2017)), axis=0)
    city['ANGmaxmean'] = np.max(map(lambda x: angval.filter(regex=str(x)).mean(axis=1), range(2001, 2017)), axis=0)

    # mean of all years
    year = 20
    city['AODmean'] = aodval.filter(regex=str(year)).mean(axis=1)
    city['ANGmean'] = angval.filter(regex=str(year)).mean(axis=1)


    # Max
    year = 20
    city['AODmax'] = aodval.filter(regex=str(year)).max(axis=1)
    city['ANGmax'] = angval.filter(regex=str(year)).max(axis=1)


    # Std dev
    yeard = 20
    city['AODstd' + str(yeard)] = aodval.filter(regex=str(yeard)).std(axis=1)
    city['ANGstd' + str(yeard)] = angval.filter(regex=str(yeard)).std(axis=1)

    # PLot
    fig, ax = plt.subplots()
    ax.set_title('ANG and AOD scatter plot', fontsize = 20)
    ax.set_xlabel('AOD', fontsize = 30)
    ax.set_ylabel('ANG', fontsize = 30)
    ax.tick_params(labelsize=20)

    # maxAODmaxANG
    # x = city['AODmax']
    # y = city['ANGmax']

    # # max amongst all annual mean
    # x= city['aod_maxmean']
    # y = city['ang_maxmean']

    #mean all months
    x = city['AODmaxmean']
    y = city['ANGmaxmean']

    # plot scatter
    plt.scatter (x, y , marker = 'o', s= 30, alpha = 0.5 )

    # plot error bar
    # ax.errorbar(x, y, xerr = city['AODstd' + str(yeard)], yerr = city['ANGstd' + str(yeard)],  alpha = 0.5, elinewidth= 1, fmt = ' ')

    # plot single city
    # (aodval[aodval.city=='Tokyo'].filter(regex='20')).transpose().plot()

    # Plot triangle coordinates and shade patch
    coordx = [25., 530., 45.]
    coordy = [30., 830., 1100.]

    verts = [(35., 50.),
             (465., 620.),
             (40., 610.),
             (35., 50.)]

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY]

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='orange', lw=1, alpha= 0.1)
    ax.add_patch(patch)

    # annotate triangle
    ax.annotate('B', xy=verts[0], xytext=(-2, 2),
            textcoords='offset points', ha='right', va='bottom', color = 'blue',
            bbox = dict(facecolor ='none', edgecolor = 'blue', boxstyle = 'round'))
    ax.annotate('R', xy=verts[1], xytext=(-2, 2),
                textcoords='offset points', ha='right', va='bottom', color = 'red',
                bbox=dict(facecolor='none', edgecolor='red', boxstyle='round'))
    ax.annotate('G', xy=verts[2], xytext=(-2, 2),
                textcoords='offset points', ha='right', va='bottom', color = 'green',
                bbox=dict(facecolor='none', edgecolor='green', boxstyle='round'))

    # annotate city names
    # for label, xa, ya in zip(city.city,x,y):
    #     plt.annotate(
    #         label,
    #         xy = (xa,ya), xytext = (-5, -5),
    #         textcoords = 'offset points', ha='right', va='bottom', alpha = 0.8, fontsize = 10)

    ls =[]
    for xt, yt, s in zip (x,y, city.city):
        ls.append(plt.text (xt, yt, s))

    adjust_text(ls, arrowprops = dict(arrowstyle = "-", color = 'b', lw = 0.3), autoalign = 'y',
                only_move={'points':'y', 'text':'y'},
                expand_points=(1.2, 1.75),
                force_points=0.1)


    plt.show()


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step III: RGB image       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Define the vertices
# x - AOD, y - ANG || w - B, s-R,  v-G
# xv =	150
# yv =	1500
# xs =	400
# ys =	1350
# xw =	90
# yw =	50
# xc =	1080
# yc =	1500
xv =	40
yv =	610
xs =	465
ys =	620
xw =	35
yw =	50
xc =	1080
yc =	1500

lvmax = abs((ys - yw) * (xv - xs) + (ys - yv) * (xs - xw)) / np.sqrt((ys - yw) * (ys - yw) + (xs - xw) * (xs - xw))
lsmax = abs((yw - yv) * (xs - xw) + (yw - ys) * (xw - xv)) / np.sqrt((yw - yv) * (yw - yv) + (xw - xv) * (xw - xv))
lwmax = abs((yv - ys) * (xw - xv) + (yv - yw) * (xv - xs)) / np.sqrt((yv - ys) * (yv - ys) + (xv - xs) * (xv - xs))


def distRGB(xx,yy):
    # Distance for perpendicular line * /
    # lwmin = abs((yv-ys) * (xc-xv)+(yv-yc) * (xv-xs)) / np.sqrt((yv-ys) * (yv-ys)+(xv-xs) * (xv-xs))
    lv = abs((ys - yw) * (xx - xs) + (ys - yy) * (xs - xw)) / np.sqrt((ys - yw) * (ys - yw) + (xs - xw) * (xs - xw))
    ls = abs((yw - yv) * (xx - xw) + (yw - yy) * (xw - xv)) / np.sqrt((yw - yv) * (yw - yv) + (xw - xv) * (xw - xv))
    lw = abs((yv - ys) * (xx - xv) + (yv - yy) * (xv - xs)) / np.sqrt((yv - ys) * (yv - ys) + (xv - xs) * (xv - xs))

    # scaling from 0 to 100
    lv = lv / lvmax * 100.0
    ls = ls / lsmax * 100.0
    lw = lw / lwmax * 100.0

    if (lw < 0.0):
        lw = 0.0
    ll = lv + ls + lw
    lv = lv / ll * 100.0
    ls = ls / ll * 100.0
    lw = lw / ll * 100.0

    v = lv
    s = ls
    w = lw

    # Cloud mask
    if (xx > xs) & (yy > ys):
        v = 100.
        s = 100.
        w = 100.

    # Cloud mask
    if (xx < xw) & (yy < yw):
        v = 0.
        s = 0.
        w = 0.

    return s,v,w

def toRGB(arr_aod, arr_ang):

    size = np.shape(arr_ang)
    band1 = np.zeros(size)
    band2 = np.zeros(size)
    band3 = np.zeros(size)
    rgb = np.zeros([size[0], size[1], 3])

    for i in range(0,size[0]):      # Y axis
        for j in range(0, size[1]):     # X axis

            # Input from mean images
            xx = arr_aod[i][j]
            yy = arr_ang[i][j]

            # Distance for perpendicular line * /
            s,v,w = distRGB(xx,yy)

            rgb[i,j, 0]=s
            rgb[i,j, 1]=v
            rgb[i,j, 2]=w

    print "complete"

    rgb255 = np.array(rgb*2.55, dtype=np.uint8)
    return rgb255



# III.a) Generating RGB scale triangle

# Sending all AOD,AND coordinates
arr_aod = [[i for i in range(0,1001)]]*1001
# arr_ang = np.transpose([[i for i in range(2000,-1, -1)]]*2001)
arr_ang = np.transpose([[i for i in range(0,1000)]]*1001)

rgbarr = toRGB(arr_aod, arr_ang)
rgbmask = rgbarr

# Plot
fig, ax = plt.subplots()
ax.set_xlabel('AOD', fontsize=20)
ax.set_ylabel('ANG', fontsize=20)
ax.tick_params(labelsize =15)
ax.grid(b=False)
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1000])
ax.imshow(rgbarr, origin='lower')

verts = [(35., 50.),
         (465., 620.),
         (40., 610.),
         (35., 50.)]

codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY]

path = Path(verts, codes)
# patch = patches.PathPatch(path, facecolor='white', lw=3, alpha=0.1)
# ax.add_patch(patch)
xs, ys = zip(*verts)
ax.plot(xs, ys, 'x--', lw=2, color='white', ms=10)

# annotate triangle
ax.annotate('B', xy=verts[0], xytext=(-2, 2),
            textcoords='offset points', ha='right', va='bottom', color='blue', fontsize = 20,
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
ax.annotate('R', xy=verts[1], xytext=(-2, 2),
            textcoords='offset points', ha='right', va='bottom', color='red', fontsize = 20,
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
ax.annotate('G', xy=verts[2], xytext=(-2, 2),
            textcoords='offset points', ha='right', va='bottom', color='green', fontsize = 20,
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))




# ----------- alt app startrt
rgbarr2 = rgbarr2 = [[[0]*3]*1001]*1001
for i in range(0,1000):
    for j in range(0, 1000):
        rgbarr2[i][j][0], rgbarr2[i][j][1], rgbarr2[i][j][2] = distRGB(i,j)
rgbarr2=np.array(rgbarr2 * 2.55, dtype=np.uint8)


#Constructing path around the triangle and removing outliers
verts = [(35., 50.),
         (465., 620.),
         (40., 610.),
         (35., 50.)]
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY]
path = Path(verts, codes)

for i in range(0,1000):
    for j in range(0, 1000):
        rgbmask[i,j] = rgbmask[i,j]*int(path.contains_point([i,j]))

plt.imshow(rgbmask)


# New approach beign -- not verified
li = [i for i in range(0,2000)]
lj = [i for i in range(0,2000)]
rgbtr = pd.DataFrame(list(product(li,lj)), columns=['ang','aod'])

rgbtr['r'], rgbtr['g'], rgbtr['b']  = zip(*rgbtr.apply(lambda x: distRGB(x.aod, x.ang), axis=1))
rgbtr['trg'] = rgbtr.apply(lambda x: path.contains_point([x.aod,x.ang]), axis =1)

fig, ax = plt.subplots()
ax.scatter(rgbtr.aod, rgbtr.ang,s=20, facecolors = [rgbtr.r/100., rgbtr.g/100., rgbtr.b/100.])


for x,y,r,g,b in zip(rgbtr.aod, rgbtr.ang, rgbtr.r/100., rgbtr.g/100., rgbtr.b/100.):
    plt.scatter (x,y,s=2,color=(r,g,b))
    plt.show()

# ----------- alt app ned

# Generate Air RGB final image

def AirRGB(year):
    start_date = date(year, 1, 1)
    end_date = date(year, 1, 12)

    for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
        # for aod.file, ang.file in zip(glob(os.path.join(aod.path, '*.' + aod.prod + '*' + '.Global.tif')),
        #                               glob(os.path.join(ang.path, '*.' + ang.prod + '*' + '.Global.tif'))):
        #
        #     # dthis gives ate as a list with integer year and month as return
        #     date_ym = int(aod.file[-21:-15]) # MODIS, OMI both
        #     # print date_ym

            # if ((str(date_ym) == (str(date_m.year) + str('%02d'%date_m.month))) & os.path.isfile(prodT.file)):

                file_date =  (str(date_m.year) + str('%02d'%date_m.month))
                aod.file = '/home/prakhar/Research/AQM_research///Data/Data_process/GlobalMean/MOD04L2.A'+ file_date +'.AOD.Global.tif'
                ang.file = '/home/prakhar/Research/AQM_research///Data/Data_process/GlobalMean/MOD04L2.A' + file_date +'.ANG.Global.tif'

                # print 'file found ', date_ym
                try:
                    arr_aod = aod.raster_as_array()
                    arr_ang = ang.raster_as_array()
                    rgbarr = toRGB(arr_aod, arr_ang)

                    plt.show(rgbarr)

                except AttributeError:
                    print ' seems like missing file  in ', date_m

# function end

# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step IV.a:continent wise ANG/AOD       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

def continentAQ(year, continent):

    # Finding mean AQ first

    # reading all cities
    city = pd.read_csv('RGB/global_megacity2.csv', header=0)
    city['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
                                                             axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    aodval = pd.read_csv(csv_save_path + 'df_Globalcity_AOD.csv', header=0)
    angval = pd.read_csv(csv_save_path + 'df_Globalcity_ANG.csv', header=0)

    # Mean
    city['AOD' + str(year)] = aodval.filter(regex=str(year)).mean(axis=1)
    city['ANG' + str(year)] = angval.filter(regex=str(year)).mean(axis=1)
    # Std dev
    city['AODstd' + str(year)] = aodval.filter(regex=str(year)).std(axis=1)
    city['ANGstd' + str(year)] = angval.filter(regex=str(year)).std(axis=1)

    # Plot
    x = city[city.continent==continent]['AOD' + str(year)]
    y = city[city.continent == continent]['ANG' + str(year)]

    fig, ax = plt.subplots()
    ax.set_title('ANG-AOD characteristics for ' + continent, fontsize = 20)
    ax.set_xlabel('AOD', fontsize = 20)
    ax.set_ylabel('ANG', fontsize = 20)

    # plot scatter
    plt.scatter (x,y, marker = 'o', )

    # plot error bar
    ax.errorbar(x,y, xerr = city[city.continent==continent]['AODstd' + str(year)], yerr = city[city.continent==continent]['ANGstd' + str(year)],  alpha = 0.5, elinewidth= 1, fmt = ' ')

    # annotate city names
    for label, xa, ya in zip(city[city.continent==continent]['city'],x,y):
        plt.annotate(
            label,
            xy = (xa,ya), xytext = (-2, 2),
            textcoords = 'offset points', ha='right', va='bottom')

    plt.show()
# Function end

# Function run ----
year = 2010
continentAQ(year, 'Asia')
continentAQ(year, 'Europe')


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step IV.b:city wise ANG/AOD seasonal      * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

def cityAQ(year, cityname):

    # Finding mean AQ first

    # reading all cities
    # city = pd.read_csv('RGB/global_megacity2.csv', header=0)
    # city['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
    #                                                          axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    aodval = pd.read_csv(csv_save_path + 'df_Globalcity_AOD.csv', header=0 )
    angval = pd.read_csv(csv_save_path + 'df_Globalcity_ANG.csv', header=0)
    aodval.set_index(['city'], inplace=True)
    angval.set_index(['city'], inplace=True)

    # Plot
    x = aodval[aodval.index == cityname].filter(regex=str(year)).transpose()
    y = angval[angval.index == cityname].filter(regex=str(year)).transpose()

    fig, ax = plt.subplots()
    ax.set_title('ANG-AOD characteristics for ' + cityname, fontsize = 20)
    ax.set_xlabel('AOD', fontsize = 20)
    ax.set_ylabel('ANG', fontsize = 20)
    ax.set_xlim([0,2000])
    ax.set_ylim([0, 2000])

    # plot scatter
    ax.plot (x,y, '>--', label = cityname)

    # annotate city names
    for label, xa, ya in zip(x.index.tolist(),x[cityname], y[cityname]):
        ax.annotate(
            label[3:],
            xy = (xa,ya), xytext = (-2, 2),
            textcoords = 'offset points', ha='right', va='bottom')
        print 'done ', label

    plt.show()
# Function end

# Function run ----
city = pd.read_csv('RGB/global_megacity2.csv', header=0)
year = 2015

for cityname in city.city:
    cityAQ(year, cityname)





# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step V: GDP percap and AQ       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

GDPpc = pd.read_csv('/home/prakhar/Research/AQM_research/Data/Data_process/Economics/Global/GDPpercap.csv', header=0)
GDPg = pd.read_csv('/home/prakhar/Research/AQM_research/Data/Data_process/Economics/Global/GDPgrowth.csv', header=0)

def GDPAQplot(prodT, year):
    prodval = pd.read_csv(csv_save_path + 'df_Globalcity_'+prodT.prod+'max.csv', header=0)

    # Mean AQ
    prodval[prodT.prod + str(year)] = prodval.filter(regex=str(year)).mean(axis=1)

    # Read GDP from WB and merge common
    prodGDPpc = pd.merge(prodval,GDPpc, on =['CountryCode'])

    # Plot
    x = prodGDPpc[str(year-1)]
    y = prodGDPpc[prodT.prod + str(year)]
    fig, ax = plt.subplots()
    ax.set_title(str(year-1) + ' GDPper capita & ' + str(year) + ' '+prodT.prod , fontsize = 20)
    ax.set_xlabel('GDP (USD)', fontsize = 20)
    ax.set_ylabel('NO$_2$(mg/$m^2$)', fontsize = 20)

    # plot scatter
    plt.scatter (x,y, marker = 'o', )

    # annotate city names
    for label, xa, ya in zip(prodGDPpc['city'],x,y):
        plt.annotate(
            label,
            xy = (xa,ya), xytext = (-2, 2),
            textcoords = 'offset points', ha='right', va='bottom')

    plt.show
# fucntion end

# Function run ----
year = 2005
GDPAQplot(no2, year)




def fx(x,y):
    # return pd.Series({'c1':x, 'c2':y})
    return x+2, y+2

df = pd.DataFrame({'A':[1,2,3], 'B':[11,21,31]})

df['c'], df['d'] = zip(*df.apply(lambda x: fx(x.A, x.B), axis=1))


