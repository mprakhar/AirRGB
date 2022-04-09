#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 1/20/2017
# Last edit 7/11/2017

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
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from itertools import product
from adjustText import adjust_text
from sklearn import datasets, linear_model
# import os
# os.chdir('/home/prakhar/Research/AQM_research/Codes/')
import sys

sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
# pvt imports
from spatialop import *
# pvt imports
from spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file
from spatialop import im_mean_temporal as im_mean
from spatialop import coord_translate as ct




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
#Input

gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path from my folders
gb_path_wt = '/home/wataru/research/MODIS/MOD04L2//'   # global path to be appended for all files to be sourced from wt

aod = Raster_file()
aod.path = gb_path_wt + r'/L4//'
aod.sat = 'MODIS'
aod.prod = 'AOD'
aod.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'MOD04L2.A201511.AOD.Global'
# > georef needs to be updated as per the dataset used
aod.georef = gb_path + r'/Data/Data_process/Georef_img//MODIS_global_georef.tif'
#over india
aod.georef = gb_path + r'/Data/Data_process/Meanmonth_AQ_20171111/MODIS/cleanAOD200401.tif'

ang = Raster_file()
ang.path = gb_path_wt + r'/L4//'
ang.sat = 'MODIS'
ang.prod = 'ANG'
ang.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'MOD04L2.A201511.AOD.Global'
# > georef needs to be updated as per the dataset used
ang.georef = gb_path + r'/Data/Data_process/Georef_img//MODIS_global_georef.tif'
#oevr idia
ang.georef = gb_path + r'/Data/Data_process/Meanmonth_AQ_20171111/MODIS/cleanAOD200401.tif'

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

#city locations to be cinsidered
#city0 = pd.read_csv('RGB/global_megacity2.csv', header=0)
citypath =  gb_path + r'/Codes/AirRGB//lib/india_megacity3.csv'

ind_city = ['Agra', 'Ahmedabad', 'Allahabad', 'Amritsar', 'Chennai', 'Firozabad', 'Gwalior', 'Jodhpur',
            'Kanpur', 'Lucknow', 'Ludhiana', 'Patna', 'Raipur', 'Hyderabad', 'Jaipur', 'Dehradun',
            'Bangalore', 'Kolkata', 'NewDelhi', 'Mumbai', 'Kanpur1_7','Kanpur1_5','Kanpur2_12', 'Lucknow1_4', 'Lucknow2_12', 'Lucknow2_15']

#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut/Air2RGB//'  # cas output path
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
    map.shadedrelief(alpha=0.5)

    # city locations to be considered
    city = pd.read_csv('RGB/global_megacity3.csv', header=0)
    x,y = map(city.lon, city.lat)

    # map.plot(x,y,'ro', markersize=5)
    for GDPcolor, label, size, xpt, ypt in zip(city.GDP, city.city, city.vehicle, x, y):
        if GDPcolor<=12000:
            Gcolor = 'red'
        if GDPcolor>12000:
            Gcolor = 'green'
        msize = int(size/1000000)*3
        if msize <=1:
            msize =2

        map.plot(xpt, ypt, 'o', markerfacecolor=Gcolor, markersize=msize, alpha=0.5)
        # plt.text(xpt, ypt, label, bbox = {'boxstyle' : 'round','facecolor':'red', 'alpha':0.3, 'pad':0.3})
        # plt.annotate(label, xy=(xpt, ypt), xytext = (0,0), bbox = dict(boxstyle = 'round', color ='k' ))

    ls =[]
    for xt, yt, s in zip (x,y, city.city):
        ls.append(plt.text (xt, yt, s))

    adjust_text(ls, arrowprops = dict(arrowstyle = "->", color = 'r', lw = 0.5), autoalign = 'n',
                only_move={'points':'y', 'text':'y'},
                expand_points=(1.2, 1.75),
                force_points=0.1)
    plt.show()
    return fig
# function end

#Function run ----




# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step I: Generate Monthly image       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

a = 0
def mean_gen(start_date0, end_date0, type = 'month'):

    if type =='month':
        interval = rrule.MONTHLY
    if type =='year':
        interval = rrule.YEARLY

    for date_m in rrule.rrule(interval, dtstart=start_date0, until=end_date0):
        start_date = date_m
        print start_date

        if type =='month':

            # finding mean clena images between the range of dates for AOD and ANG and saving them
            # arr_aod = im_mean.sampledmean_month(aod, start_date, start_date + relativedelta(months=1), minmax=False)

            # print "done aod"

            # arr_aq = im_mean.mean_month(ang, start_date, start_date + relativedelta(months=1), minmax=False)

            #NO2
            # arr_no2, arr_no2min, arr_no2max = im_mean.mean_month(no2, start_date, start_date + relativedelta(months=1), minmax=True)

            #SO2
            # arr_so2, arr_so2min, arr_so2max = im_mean.mean_month(so2, start_date, start_date + relativedelta(months=1), minmax=True)

            # Save array as raster image
            # srs.arr_to_raster(arr_aod, aod.georef, img_save_path+ '/GlobalMeanv2/MOD04L2.A'+ str(start_date.year) + str('%02d'%start_date.month)+'.AOD.Global.tif')

            # srs.arr_to_raster(arr_aq, aod.georef, img_save_path + '/GlobalMeanwt/MOD04L2.A' + str(start_date.year) + str('%02d' % start_date.month) + '.ANG.Global')

            #NO2
            # srs.arr_to_raster(arr_no2, no2.georef, img_save_path+ '/GlobalMean/OMI.NO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.ave.Global.tif')
            # srs.arr_to_raster(arr_no2min, no2.georef, img_save_path+ '/GlobalMean/OMI.NO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.min.Global.tif')
            # srs.arr_to_raster(arr_no2max, no2.georef, img_save_path+ '/GlobalMean/OMI.NO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.max.Global.tif')

            #SO2
            # srs.arr_to_raster(arr_so2, so2.georef, img_save_path+ '/GlobalMean/OMI.SO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.ave.Global.tif')
            # srs.arr_to_raster(arr_so2min, so2.georef, img_save_path+ '/GlobalMean/OMI.SO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.min.Global.tif')
            # srs.arr_to_raster(arr_so2max, so2.georef, img_save_path+ '/GlobalMean/OMI.SO2.'+ str(start_date.year) + str('%02d'%start_date.month)+'.max.Global.tif')

        # modifications for annual image
        if type == 'year':
            arr_aq = im_mean.mean_year(aod, start_date, start_date + relativedelta(years=1), minmax=False)
            srs.arr_to_raster(arr_aq, aod.georef,
                          img_save_path + '/GlobalMeanwt/Annual/MOD04L2.A' + str(start_date.year) + '00.AOD.Global')

# function end



# Since my mean image and sense's mean image is different . need to verify by taking daily values for a month and try to figure out what averaging scheme he sued

def output_compare(start_date0, end_date0, citychk):

    city = pd.read_csv('RGB/global_megacity2.csv', header=0)
    city['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
                                                             axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    city2 = pd.read_csv('RGB/global_megacity2.csv', header=0)
    city2['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
                                                             axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    # daily level values
    for date_m in rrule.rrule(rrule.DAILY, dtstart=start_date0, until=end_date0):
        file_date = str(date_m.year) + str('%03d'%date_m.timetuple().tm_yday)
        print file_date
        #aod
        file =  '/home/wataru/research/MODIS/MOD04L2/L3/' + 'MOD04L2.A'+ file_date +'.AOD.Global'
        aodarr = srs.raster_as_array(file)
        city['AOD'+str(date_m.day)] = pixval(aodarr, city)

        #ang
        file2 =  '/home/wataru/research/MODIS/MOD04L2/L3/' + 'MOD04L2.A'+ file_date +'.ANG.Global'
        angarr = srs.raster_as_array(file2)
        city2['ANG'+str(date_m.day)] = pixval(angarr, city2)


    # city.transpose().to_csv(csv_save_path + 'df_dailyAQcityTestAOD.csv', index=False, header=True)
    city2.transpose().to_csv(csv_save_path + 'df_dailyAQcityTestANG.csv', index=False, header=True)

    #monthly mean
    dt = (str(start_date0.year) + str('%02d'%start_date0.month))
    senarr = srs.raster_as_array('/home/wataru/research/MODIS/MOD04L2/L4/' + 'MOD04L2.A'+dt+'.ANG.Global')
    myarr = srs.raster_as_array(gb_path + r'/Data/Data_process/GlobalMeanv2//' + 'MOD04L2.A'+dt+'.ANG.Global.tif')

    print senarr[[city[city.city == citychk]['pixloc']][0].tolist()[0][1], [city[city.city == citychk]['pixloc']][0].tolist()[0][0]]
    print myarr[[city[city.city == citychk]['pixloc']][0].tolist()[0][1], [city[city.city == citychk]['pixloc']][0].tolist()[0][0]]
# function end

# functionrun



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step II: RGB triangle parameters       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# value of pixel for each city in a image
def pixval(arr_prod, city):

    ls = []
    for pix in city.pixloc:
        ls.append(arr_prod[pix[1], pix[0]])
        # print pix, arr_prod[pix[1], pix[0]]
    return ls

# open all images and gather pix valuez
def allpixval(prodT, start_date, end_date, save_path):


    # Getting pixel values from lat/long values
    # input>>lat, long   output >>pix_y, pix_x
    city = pd.read_csv(citypath, header=0)
    city['pixloc'] = ct.latLonToPixel(prodT.georef,
                                      city.apply(lambda x: [x['lat'], x['lon']],axis=1).tolist())

    # run for the dates
    for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
        for prodT.file in glob(os.path.join(prodT.path, '*' + prodT.prod + '*' + '.tif')): # no .tif extension provided in WT files and .tif extension in OM files

            # this gives date as a list with integer year and month as return
            date_ym = int(prodT.file[-10:-4]) # MODIS, OMI both WT[-17:-11], PM[-21,-15] // for clean files creaetd by me [-10:-4
            print date_ym

            if ((str(date_ym) == (str(date_m.year) + str('%02d'%date_m.month))) & os.path.isfile(prodT.file)):

                print 'file found ', date_ym
                try:
                    prod_arr = prodT.raster_as_array()
                    city[prodT.prod + str(date_ym)] = pixval(prod_arr, city)

                except AttributeError:
                    print ' seems like missing file  in ', date_ym

    # save to csv
    city.to_csv(save_path, index=False, header=True)
# function end

# Function run ---- find aod, ang alues for global cities

# Find parameters by plotting the mean annual for aod and ang
def rgb_param(aodval, angval, verts, annot = False, type='max', std = 'False'):


    # read city pixel locations
    city = pd.read_csv(citypath, header=0)
    city['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
                                                             axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    # get rid of zero as they skew the mean
    aodval = aodval.replace(0, np.NaN)
    angval = angval.replace(0, np.NaN)

    # [col for col in aodval.columns if str(year) in col]
    # prefix of all years
    # Mean
    year = 20
    city['AOD' + str(year)] = aodval.filter(regex=str(year)).mean(axis=1)
    city['ANG' + str(year)] = angval.filter(regex=str(year)).mean(axis=1)

    # for data clarity.. count the null values. These are calculated in my dataset
    #city['aodcount'] = pd.read_csv(csv_save_path + 'df_20cityWT_AODpmv2.csv', header=0).isnull().sum(axis=1)
    #city['angcount'] = pd.read_csv(csv_save_path + 'df_20cityWT_ANGpmv2.csv', header=0).isnull().sum(axis=1)

    # mean at annual level and finding the max of it
    city['AODmaxmean'] = np.nanmax(map(lambda x: aodval.filter(regex=str(x)).mean(axis=1), range(2001, 2017)), axis=0)
    city['ANGmaxmean'] = np.nanmax(map(lambda x: angval.filter(regex=str(x)).mean(axis=1), range(2001, 2017)), axis=0)

    # mean of all years
    year = 20
    city['AODmean'] = aodval.filter(regex=str(year)).mean(axis=1)
    city['ANGmean'] = angval.filter(regex=str(year)).mean(axis=1)

    # mean of all years
    year = 20
    city['AODmed'] = aodval.filter(regex=str(year)).median(axis=1)
    city['ANGmed'] = angval.filter(regex=str(year)).median(axis=1)


    # Max
    year = 20
    city['AODmax'] = aodval.filter(regex=str(year)).max(axis=1)
    city['ANGmax'] = angval.filter(regex=str(year)).max(axis=1)


    # Std dev
    year = 20
    city['AODstd' + str(year)] = aodval.filter(regex=str(year)).std(axis=1)
    city['ANGstd' + str(year)] = angval.filter(regex=str(year)).std(axis=1)

    # PLot
    fig, ax = plt.subplots()
    # ax.set_title('ANG and AOD scatter plot', fontsize = 20)
    ax.set_xlabel('AOD (no unit)', fontsize = 20)
    ax.set_ylabel('ANG (no unit)', fontsize = 20)
    ax.tick_params(labelsize=20)
    ax.set_xlim([0,1200])
    ax.set_ylim([0, 1200])

    # Turns off grid on the left Axis.
    ax.grid(False)

    # maxAODmaxANG
    # x = city['AODmax']
    # y = city['ANGmax']

    # # max amongst all annual mean
    # x= city['aod_maxmean']
    # y = city['ang_maxmean']

    #mean all months
    x = city['AOD'+type]  # Consider AOD mean and while plotting triangle consider vertices such that they include standard deviations
    y = city['ANG'+type]

    # plot scatter
    plt.scatter (x, y ,
                 #c= (city['aodcount']/50).astype(int) ,
                 marker = 'o', s= 30, edgecolors='none' )
    plt.gray()

    # plot error bar
    if std:
        ax.errorbar(x, y, xerr = city['AODstd' + str(year)], yerr = city['ANGstd' + str(year)],  alpha = 0.3, elinewidth= 1, fmt = 'o', ecolor='gray', label = 'standard deviation')

    # plot single city
    # (aodval[aodval.city=='Tokyo'].filter(regex='20')).transpose().plot()

    # plot and shade triangle if required
    if annot:
        # Plot triangle coordinates and shade patch
        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='orange', lw=1, alpha= 0.1)
        ax.add_patch(patch)

        # annotate triangle
        ax.annotate('B', xy=verts[0], xytext=(-2, 2),
                textcoords='offset points', ha='right', va='bottom', color = 'blue', fontsize = 20,
                bbox = dict(facecolor ='none', edgecolor = 'blue', boxstyle = 'round'))
        ax.annotate('R', xy=verts[1], xytext=(-2, 2),
                    textcoords='offset points', ha='right', va='bottom', color = 'red', fontsize = 20,
                    bbox=dict(facecolor='none', edgecolor='red', boxstyle='round'))
        ax.annotate('G', xy=verts[2], xytext=(-2, 2),
                    textcoords='offset points', ha='right', va='bottom', color = 'green', fontsize = 20,
                    bbox=dict(facecolor='none', edgecolor='green', boxstyle='round'))

    # annotate city names
    ls =[]
    for xt, yt, s in zip (x,y, city.city):
        ls.append(plt.text (xt, yt, s))

    adjust_text(ls, arrowprops = dict(arrowstyle = "-", color = 'r', lw = 0.3), autoalign = 'y',
                only_move={'points':'y', 'text':'y'},
                expand_points=(1.2, 1.75),
                force_points=0.1)

    plt.show()
# function end






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


# Distance for perpendicular line * /
def distRGB(xx,yy, vertsT):
    # xx = AOD
    #yy = ANG


    xv, yv = vertsT[2]
    xs, ys = vertsT[1]
    xw, yw = vertsT[0]

    xx = int(xx)
    yy = int(yy)

    lvmax = abs((ys - yw) * (xv - xs) + (ys - yv) * (xs - xw)) / np.sqrt((ys - yw) * (ys - yw) + (xs - xw) * (xs - xw))
    lsmax = abs((yw - yv) * (xs - xw) + (yw - ys) * (xw - xv)) / np.sqrt((yw - yv) * (yw - yv) + (xw - xv) * (xw - xv))
    lwmax = abs((yv - ys) * (xw - xv) + (yv - yw) * (xv - xs)) / np.sqrt((yv - ys) * (yv - ys) + (xv - xs) * (xv - xs))

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

    # Mask if values excedd the maxim
    if (xx > xs) & (yy > ys):
        v = 0.
        s = 100.
        w = 0.

    # Mask if values less than minimum
    if (xx <= xw) & (yy <= yw):
        v = 0.
        s = 0.
        w = 0.

    # Mask if values equal to zero (no observed data)
    if (xx <= 50) & (yy <= 50):
        v = 0.
        s = 0.
        w = 0.

    return [s,v,w]  #RGB
# funciton end

# given an array of AOD and ANG, this will convert it RGB
def toRGB(arr_aod, arr_ang, vertsT):

    size = np.shape(arr_ang)

    # defining a blank 3 band rgb 2D array
    rgb = np.zeros([size[0], size[1], 3])

    for i in range(0,size[0]):      # Y axis
        for j in range(0, size[1]):     # X axis

            # Input from mean images
            xx = arr_aod[i][j]
            yy = arr_ang[i][j]

            # Distance for perpendicular line * /
            [s,v,w] = distRGB(xx,yy, vertsT)

            rgb[i,j, 0]=s
            rgb[i,j, 1]=v
            rgb[i,j, 2]=w

    print "complete"

    rgb255 = np.array(rgb, dtype=np.uint8)
    return rgb255
# function end


# III. Genrating RGB triangle and RGB image


# III.a) Generating RGB scale triangle

def triangle_gen(vertsT, verts, size):

    # computing all possible AOD,ANG coordinates
    arr_aod = [[i for i in range(0,size+1)]]*(size+1)
    arr_ang = np.transpose([[i for i in range(0,size)]]*(size+1))

    # computing corrresponding rgb values
    rgbarr = toRGB(arr_aod, arr_ang, vertsT)
    rgbmask = rgbarr

    # Plot the RGB triangle
    fig, ax = plt.subplots()
    ax.set_xlabel('$ \\tau $ (no unit) ', fontsize=20)
    ax.set_ylabel('$\\alpha$ (no unit)', fontsize=20)
    ax.tick_params(labelsize =15)
    ax.grid(b=False)
    ax.set_xlim([0, size])
    ax.set_ylim([0, size])
    ax.imshow(rgbarr, origin='lower')


    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY]
    path = Path(verts, codes)
    xs, ys = zip(*verts)
    ax.plot(xs, ys, 'x--', lw=2, color='white', ms=10)

    # annotate triangle
    ax.annotate('B('+str(verts[0][0])+','+str(verts[0][1])+')', xy=verts[0], xytext=(+70, 0),
                textcoords='offset points', ha='right', va='bottom', color='white', fontsize = 15,
                bbox=dict(facecolor='none', edgecolor='white', boxstyle='round'))
    ax.annotate('R('+str(verts[1][0])+','+str(verts[1][1])+')', xy=verts[1], xytext=(+70,2),
                textcoords='offset points', ha='right', va='bottom', color='white', fontsize = 15,
                bbox=dict(facecolor='none', edgecolor='white', boxstyle='round'))
    ax.annotate('G('+str(verts[2][0])+','+str(verts[2][1])+')', xy=verts[2], xytext=(+70, +0),
                textcoords='offset points', ha='right', va='bottom', color='white', fontsize = 15,
                bbox=dict(facecolor='none', edgecolor='white', boxstyle='round'))
# fucntion end



# III. c) Generate Air RGB final image       -----------------------------
def AirRGB(year, vertsT, timetype = 'month' ):

    if timetype == 'year':
        aod.file = img_save_path + '/GlobalMeanwt/Annual/MOD04L2.A' + str(year) + '00.AOD.Global'
        ang.file = img_save_path + '/GlobalMeanwt/Annual/MOD04L2.A' + str(year) + '00.ANG.Global'

        arr_aod = aod.raster_as_array()
        arr_ang = ang.raster_as_array()
        rgbarr = toRGB(arr_aod, arr_ang, vertsT)
        np.save(('AnnualRGB' + str(year)), rgbarr)
        srs.ndarr_to_raster(rgbarr, aod.file, 'AnnualRGB'+str(year) + '.tif')
        print 'saved ', year

    if timetype == 'month':

        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)

        for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):

            # date of file
            file_date =  (str(date_m.year) + str('%02d'%date_m.month))

            # opening correspondig files
            # aod.file = '/home/prakhar/Research/AQM_research///Data/Data_process/GlobalMeanv2/MOD04L2.A'+ file_date +'.AOD.Global.tif'
            # ang.file = '/home/prakhar/Research/AQM_research///Data/Data_process/GlobalMeanv2/MOD04L2.A' + file_date +'.ANG.Global.tif'

            # files from sensei
            #aod.file = '/home/wataru/research/MODIS/MOD04L2/L4/MOD04L2.A'+ file_date +'.AOD.Global'
            #ang.file = '/home/wataru/research/MODIS/MOD04L2/L4/MOD04L2.A'+ file_date +'.ANG.Global'

            # INDIA files created on 201711. with complete India. cropped from Sensei's
            aod.file = '/home/prakhar/Research/AQM_research/Data/Data_process/Meanmonth_AQ_20171111/MODIS/cleanAOD'+ file_date +'.tif'
            ang.file = '/home/prakhar/Research/AQM_research/Data/Data_process/Meanmonth_AQ_20171111/MODIS/cleanANG'+ file_date +'.tif'

            print 'file found ', file_date

            # running AirRGB conversion
            try:
                arr_aod = aod.raster_as_array()
                arr_ang = ang.raster_as_array()

                # replace na nby xzero
                arr_aodz = np.nan_to_num(arr_aod)
                arr_angz = np.nan_to_num(arr_ang)

                rgbarr = toRGB(arr_aodz, arr_angz, vertsT)
                np.save( ('RGB'+file_date) , rgbarr)
                srs.ndarr_to_raster(rgbarr, aod.file, 'MonthRGB' + str(file_date) + '.tif')
                # plt.figure()
                # plt.imshow(rgbarr)

            except AttributeError:
                print ' seems like missing file  in ', date_m

    if timetype == 'daily':

            # opening correspondig files
            # aod.file = '/home/prakhar/Research/AQM_research///Data/Data_process/GlobalMeanv2/MOD04L2.A'+ file_date +'.AOD.Global.tif'
            # ang.file = '/home/prakhar/Research/AQM_research///Data/Data_process/GlobalMeanv2/MOD04L2.A' + file_date +'.ANG.Global.tif'
            aod.file = '/home/wataru/research/MODIS/MOD04L2/L3/MOD04L2.A'+ year +'.AOD.Global'
            ang.file = '/home/wataru/research/MODIS/MOD04L2/L3/MOD04L2.A'+ year +'.ANG.Global'

            print 'file found ', year

            # running AirRGB conversion
            try:
                arr_aod = aod.raster_as_array()
                arr_ang = ang.raster_as_array()
                rgbarr = toRGB(arr_aod, arr_ang, vertsT)
                np.save( ('RGB'+year) , rgbarr)
                srs.ndarr_to_raster(rgbarr, aod.file, 'DailyRGB' + str(year) + '.tif')
                # plt.figure()
                # plt.imshow(rgbarr)

            except AttributeError:
                print ' seems like missing file  in ', year

# function end



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step IV.b:city wise ANG/AOD seasonal      * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

def cityAQ(year, cityname):

    # Finding mean AQ first

    # reading all cities
    # city = pd.read_csv('RGB/global_megacity2.csv', header=0)
    # city['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
    #                                                          axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    aodval = pd.read_csv(csv_save_path + 'df_20city_AOD.csv', header=0 )
    angval = pd.read_csv(csv_save_path + 'df_20city_ANG.csv', header=0)
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
year = 2015

#for cityname in city.city:
#    cityAQ(year, cityname)





# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step V: GDP percap and AQ       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#


def GDPAQplot(prodT, year):
    prodval = pd.read_csv(csv_save_path + 'df_20city_'+prodT.prod+'max.csv', header=0)

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
#GDPpc = pd.read_csv('/home/prakhar/Research/AQM_research/Data/Data_process/Economics/Global/GDPpercap.csv', header=0)
#GDPg = pd.read_csv('/home/prakhar/Research/AQM_research/Data/Data_process/Economics/Global/GDPgrowth.csv', header=0)
#year = 2005
#GDPAQplot(no2, year)


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step VI: AirRGB trend       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# value of pixel for each city in a image
def pixval3D(arr_prod, city):

    ls = []
    for pix in city.pixloc:
        ls.append(arr_prod[pix[1], pix[0]])
    return ls

# store RGB values or required cities from >>> AirRGB "images"
def AirRGBimage_trend(citypath):

    #open the dataframe containg all lat lon
    city = pd.read_csv(citypath, header=0)

    # covnert lat longs to pixel locations
    city['pixloc'] = ct.latLonToPixel(aod.georef, city.apply(lambda x: [x['lat'], x['lon']],
                                                             axis=1).tolist())  # input>>lat, long   output >>pix_y, pix_x

    # datet limits
    start_date = date(2001, 1, 1)
    end_date = date(2016, 12, 1)
    ls = []
    ls1 = []
    for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):

        # date of file
        file_date =  (str(date_m.year) + str('%02d'%date_m.month))
        print file_date

        #read RGB file as numpy array
        arr_RGB = np.load('RGB'+file_date+'.npy')

        # checking RGB for all cities for a date
        city['date'] = file_date
        city['date_format'] = date_m
        city['R'] = city.apply(lambda x: arr_RGB[x.pixloc[1], x.pixloc[0], 0], axis=1)
        city['G'] = city.apply(lambda x: arr_RGB[x.pixloc[1], x.pixloc[0], 1], axis=1)
        city['B'] = city.apply(lambda x: arr_RGB[x.pixloc[1], x.pixloc[0], 2], axis=1)

        # converting all RGB to a list which will be converted to master list
        ls.append(city[['city', 'date', 'date_format', 'R', 'G', 'B']].values.tolist())


    df_RGB = pd.DataFrame(zip(*sum(ls, [])),['city', 'date', 'date_f' , 'R', 'G', 'B']).transpose()

    return df_RGB
# function end




# ---------------- Testing and analyzing time * fine tune traingle vaues --------------------
# III. b) Generate Air RGB parameters for a city and plot   -----------------------------
# find linregression paramter
def fnregress (x,y):
    [m, c] = np.polyfit(x, y, 1)
    regress = [(i*m+c) for i in x]
    return [m,c,regress]
# function end

# plot trendf rom df
def plotRGBtrend(citychk, df_RGB ):

    # set values
    x = pd.to_datetime(df_RGB[df_RGB.city == citychk]['date'], format='%Y%m')
    y1 = df_RGB[df_RGB.city == citychk]['R']
    y2 = df_RGB[df_RGB.city == citychk]['G']
    y3 = df_RGB[df_RGB.city == citychk]['B']

    # movaing average
    y1ma = df_RGB[df_RGB.city == citychk]['R'].rolling(window=4).mean()
    y2ma = df_RGB[df_RGB.city == citychk]['G'].rolling(window=4).mean()
    y3ma = df_RGB[df_RGB.city == citychk]['B'].rolling(window=4).mean()

    fig, ax = plt.subplots(figsize=(16, 3))

    # plot points
    ax.plot(x, y1ma, '--',  c = 'Red', label='R', alpha = 0.4, linewidth = 1, markersize=4)
    ax.plot(x, y2ma, ':',  c = 'Green',label='G', alpha=0.8, linewidth = 1, markersize=4)
    ax.plot(x, y3ma, '.-',  c = 'Blue', label='B', alpha=0.4, linewidth = 1, markersize=4)

    # plot time axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m\n%y'))

    ax.set_xlim(x.min(),x.max() )
    ax.set_ylim(0,100)

    # trendline regression
    x0 = range(0, np.size(x))

    # red
    [m, c, regress] = fnregress(x0, y1)
    ax.plot(x,regress, c = 'Red', label='R trend: y='+'%0.2f'%(m)+'x+'+'%0.2f'%(c), alpha = 0.7, linewidth = 1.5)

    # green
    [m, c, regress] = fnregress(x0, y2)
    ax.plot(x,regress, c = 'Green', label='G trend: y='+'%0.2f'%(m)+'x+'+'%0.2f'%(c), alpha = 0.7, linewidth = 1.5)

    # blue
    [m, c, regress] = fnregress(x0,y3)
    ax.plot(x,regress, c = 'Blue', label='B trend: y='+'%0.2f'%(m)+'x+'+'%0.2f'%(c), alpha = 0.7, linewidth = 1.5)

    yn = filter(lambda a: a != 0, y1)

    # regression = pd.ols(y=y1, x=x0)
    # regression.summary
    # trend = regression.predict(beta=regression.beta, x=x)
    # data = pd.DataFrame(index=x, data={'y': y1, 'trend': trend})
    # data.plot()
    #
    # tick_spacing = 10

    #plt.title(' AirRGB trend - '+citychk, fontsize=20)
    plt.legend( loc='upper right',
               ncol=6, borderaxespad=0., fontsize = 12)

    plt.xlabel('Month/Year', fontsize=15)
    plt.ylabel('AirRGB value', fontsize=15)

    plt.show()
# fucntion end

# function to plot trednof AOD and ANG from df
def plotAODANG(citychk, df_AQ):

    # plot AOD and ANG to compare
    fig, ax = plt.subplots(figsize=(16, 3))
    range_r = range(0,len(df_AQ.ANG))

    # assigning dates
    x_datelabel = pd.date_range('2001-1', periods=len(df_AQ.ANG), freq='M')

    # Train the model using the training sets    AOD
    [m, c] = np.polyfit(range_r, df_AQ.AOD, 1)
    regress = [(i * m + c) for i in range_r]

    #plot regresion lilne
    ax.plot(x_datelabel, regress, color='blue',linewidth=1.5, alpha=0.7, label = 'AOD trend: y='+'%0.2f'%(m)+'x+'+'%0.2f'%(c))

    # plot line
    ax.plot(x_datelabel, df_AQ.AOD, color='blue',linewidth=1, alpha=0.4, label = 'AOD')

    # Train the model using the training sets    ANG
    [m, c] = np.polyfit(range_r, df_AQ.ANG, 1)
    regress = [(i * m + c) for i in range_r]

    #plot regresion lilne
    ax.plot(x_datelabel, regress, color='red',linewidth=1.5, alpha =0.7,  label = 'ANG trend: y='+'%0.2f'%(m)+'x+'+'%0.2f'%(c))

    # plot line
    ax.plot(x_datelabel,df_AQ.ANG, color='red',linewidth=1, alpha=0.4, label = 'ANG')

    # plot time axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m\n%y'))
    ax.set_xlim(x_datelabel[0],x_datelabel[-1] )
    ax.set_ylim(0,)

    #final plot elements
    plt.legend(fontsize=15)
    plt.ylabel('MODIS measurement(x1000)', fontsize=18)
    plt.xlabel('Month/year', fontsize=18)
    plt.title(' AOD, ANG trend - ' + citychk, fontsize=20)

    plt.show()
#function end

# Function to generate R,G,B dataframe for a city if AOD ANG dataframe provided
def df_AQgen(aodval, angval, citychk, vertsT):

    year = 20

    # combining into a single frame
    frame_AQ = [angval[angval.city == citychk].filter(regex=str(year)).transpose().reset_index(),
                aodval[aodval.city == citychk].filter(regex=str(year)).transpose().reset_index()]
    df_AQ = pd.concat(frame_AQ, axis=1)
    df_AQ.columns = ['ANGdate', 'ANG', 'AODdate', 'AOD']
    df_AQ['date'] = df_AQ.AODdate.str[-6:]
    df_AQ['city'] = citychk
    df_AQ.fillna(0, inplace = True)

    # computing RGB values from AODANG values by calling distRGBG function
    df_AQ['RGB'] = df_AQ.apply(lambda row: distRGB(xx=row['AOD'], yy=row['ANG'], vertsT=vertsT), axis=1)
    df_AQ[['R', 'G', 'B']] = df_AQ['RGB'].apply(pd.Series)

    return df_AQ
#function end

# compute RGB values of city if AOD ANG already exist in a daatframe
def RGBcity(aodval, angval, vertsT, citychk = 'default', plot = True):

    # generate dataframe of R,G,B and AOD,ANG a particualr city
    df_AQ = df_AQgen(aodval, angval, citychk, vertsT)

    # save the info so that i t can be plotted as excel later on
    df_AQ.to_csv(csv_save_path+'df_AQRGB_'+citychk+'.csv', index=False, header=True)

    if plot:

        #view its R,G,B plot
        plotRGBtrend(citychk, df_AQ)

        # view its AOD, ANG plot
        plotAODANG(citychk, df_AQ)

    #return
    return df_AQ
# function end


#function for caclulating ma nd c of RGB trend for all cities from a dataframe that already stores their AOd and ANG values
def AirRGBdf_trend(aodval, angval, vertsT):

    #lsit to store scale and offset
    ls = []

    # find scale adn ofset for R,G,B for each city
    for citychk in aodval.city:

        print citychk
        # generate dataframe of R,G,B and AOD,ANG a particualr city
        df_AQ = df_AQgen(aodval, angval, citychk, vertsT)


        # # set values
        x = df_AQ.date
        y1 = df_AQ.R
        y2 = df_AQ.G
        y3 = df_AQ.B


        # define parameter to run regression on
        x0 = range(0, np.size(y1))

        # append cityname, and m,c parameters for red, blue, green
        ls.append([citychk, fnregress(x0, y1)[0:2],fnregress(x0, y2)[0:2], fnregress(x0, y3)[0:2]])

    #save scale and offset as dataframe
    df_mc = pd.DataFrame(ls, columns = ['city', 'Rmc', 'Gmc', 'Bmc'])

    df_mc[['Rm', 'Rc']] = df_mc['Rmc'].apply(pd.Series)
    df_mc[['Gm', 'Gc']] = df_mc['Gmc'].apply(pd.Series)
    df_mc[['Bm', 'Bc']] = df_mc['Bmc'].apply(pd.Series)

    #  save m and c as csv
    df_mc.to_csv(csv_save_path+'df_20cityWT_RGBmc.csv', index=False, header=True)

    return df_mc
# fucntion end

def plotRGBmc(df_mc, labelc):

    df_mc['markerstyle'] = 'x'
    df_mc.loc[df_mc.city.isin(ind_city), 'markerstyle'] = 'o'

    fig,ax = plt.subplots(figsize=(6, 6))

    if labelc == 'R':
        x = df_mc.Rm
        y = df_mc.Rc
        x1 = df_mc[df_mc.city.isin(ind_city)]['Rm']
        y1 = df_mc[df_mc.city.isin(ind_city)]['Rc']
    if labelc == 'G':
        x = df_mc.Gm
        y = df_mc.Gc
        x1 = df_mc[df_mc.city.isin(ind_city)]['Gm']
        y1 = df_mc[df_mc.city.isin(ind_city)]['Gc']
    if labelc == 'B':
        x = df_mc.Bm
        y = df_mc.Bc
        x1 = df_mc[df_mc.city.isin(ind_city)]['Bm']
        y1 = df_mc[df_mc.city.isin(ind_city)]['Bc']

    ax.plot(x, y, 'o' , color = labelc, label= labelc + ': for other non-India', alpha = 0.4, markersize=4)
    ax.plot(x1, y1, 'D', color=labelc, label=labelc + ': for India', alpha=0.4, markersize=6)

    sns.despine(ax=ax, offset=0)  # the important part here
    plt.title(labelc, fontsize=20)
    plt.legend(fontsize=15)
    plt.xlabel('Slope', fontsize=18)
    plt.ylabel('Offset', fontsize=18)

    # annotate city names
    ls =[]
    for xt, yt, s in zip (x,y, df_mc.city):
        ls.append(plt.text (xt, yt, s))

    adjust_text(ls, arrowprops = dict(arrowstyle = "-", color = 'b', lw = 0.3), autoalign = 'y',
                only_move={'points':'y', 'text':'y'},
                expand_points=(1.2, 1.75),
                force_points=0.1)

    plt.show()
# function end




####------------- -------- ------ SET and RUN ---------- ------- -------- ------- -----
#read city info data
city0 = pd.read_csv(citypath, header=0)
# ----------------------------------------------------------------
#   1. generate moena monthly files
start_date0 = date(2001, 1, 1)
end_date0 = date(2017, 1, 1)
mean_gen(start_date0, end_date0)

#       comapre results with sensei
start_date0 = date(2015, 4, 1)
end_date0 = date(2015, 5,1 )
citychk = 'Tokyo'
output_compare(start_date0, end_date0, citychk)


# ----------------------------------------------------------------
# 2 . Mean Monthly files files generated by me to fill missing files

start_date = date(2001, 1, 1)
end_date = date(2017, 1, 1)
#aod.path = gb_path + r'/Data/Data_process/GlobalMeanwt//'
#ang.path = gb_path + r'/Data/Data_process/GlobalMeanwt//'

#       Open each mean AOD ANG image and save values for each city from each year/month
#20 city version

# version for simle mean
# files generated by sensei
#aod.path = '/home/wataru/research/MODIS/MOD04L2/L4/'
#ang.path = '/home/wataru/research/MODIS/MOD04L2/L4/'
#save_path_o = csv_save_path + 'df_20cityWT_'+aod.prod+'wtv2.csv'
#save_path_n = csv_save_path + 'df_20cityWT_'+ang.prod+'wtv2.csv'

#version for mean with zero values neglected. makes use fo clean monthly AOD, ANG generated on 20171111
aod.path = gb_path + r'/Data/Data_process/Meanmonth_AQ_20171111/MODIS/'
ang.path = gb_path + r'/Data/Data_process/Meanmonth_AQ_20171111/MODIS/'
save_path_o = csv_save_path + 'df_20cityWT_'+aod.prod+'wtv3_nashi_0.csv'
save_path_n = csv_save_path + 'df_20cityWT_'+ang.prod+'wtv3_nashi_0.csv'


allpixval(aod, start_date, end_date, save_path_o)
allpixval(ang, start_date, end_date, save_path_n)

#       open soteres AOD and ANG values open monthly mean monthly estimated AOD and ANG valeus # used 'df_GlobalcityWT_AODwtv2bkp.csv'
# aodval = pd.read_csv(csv_save_path + 'df_20cityWT_AODwtv2.csv', header=0)
# angval = pd.read_csv(csv_save_path + 'df_20cityWT_ANGwtv2.csv', header=0)
aodval = pd.read_csv(save_path_o, header=0)
angval = pd.read_csv(save_path_n, header=0)

# ----------------------------------------------------------------
#   3. vertices as found by wt data max. B,R,G - nashi
#verts_wtmx = [(1., 1.),
#          (1300., 1650.),
#          (200., 1900.),
#          (1., 1.)]

#       vertices as found by my data sampled mean - nashi
#verts_pm = [(400., 950.),
#          (1100., 1550.),
#          (10., 1750.),
#          (400., 950.)]

#       vertices as found by wt data simple mean - haai
# this is used in the journal
#verts_wtmn = [(50, 50),
#          (600, 800),
#          (20, 1000),
#          (50, 50)]

# modification of the journal values as recommded by Imasu sensei
#Code name - 20171202 Imasu
verts_wtmn = [(50, 50),
          (800, 1000),
          (20, 1000),
          (50, 50)]

# modification of the Imasue sensei values to prevent saturation
#Code name - 20171215Imasumod
verts_wtmn = [(50, 50),
          (1500, 1500),
          (20, 1500),
          (50, 50)]

# modification of the Imasue sensei values to prevent saturation
#Code name - 20180108ImasumodII
verts_wtmn = [(50, 50),
          (1000, 1500),
          (20, 1500),
          (50, 50)]


#       Deciding vertices of the RGB gradient triangle . order: B,R,G
vertsT = [[50,50],[1000,1000],[20,1000]]

#       plot all AOD,ANG for cities and to find the triangle
rgb_param(aodval, angval, verts_wtmn, annot = True, type = 'mean', std =True)

#       run function to generate gradient triangle
triangle_gen(vertsT, verts_wtmn, 1200)
# try np.meshgrid here instead


# ----------------------------------------------------------------
#   4. DATA GENERATE generate RGB image Run Function : Genearte AirRGB image for the year
for year in range(2001,2017):
    AirRGB(year, verts_wtmn, 'month')

#      funciton run to save RGB trend form RGB images
df_RGB = AirRGBimage_trend(citypath)
df_RGB.to_csv(csv_save_path + '/df_20cityWT_trend.csv', index=False, header=True)

#       finding trend from the csv saved in previous step
df_RGB = pd.read_csv(csv_save_path + '/df_20cityWT_trend.csv', header=0)
plotRGBtrend('Tokyo', df_RGB )


# ----------------------------------------------------------------
#   5. function run to find RGB trend for a city from AOD,ANG datatframes and plot. Useful for experimenting vertsS

#generate RGB values from all aod ang values for the city
df_AQ = RGBcity(aodval, angval,  verts_wtmn, citychk = 'Lucknow2_3', plot = 'True')

#run all indian cities
for city in city0.city.values:
    #   for city in ['Jaipur', 'Jaipur1_5', 'Jaipur2_15', 'Jaipur2_7', 'Dehradun', 'Dehradun1_1', 'Dehradun1_8', 'Dehradun2_16']:
    df_AQ = RGBcity(aodval, angval, verts_wtmn, citychk=city, plot = False)


#       get slope and offset for RGB of all cities
df_mc = AirRGBdf_trend(aodval, angval, verts_wtmn)

#       plot all the slopes and offset for RGB separately
plotRGBmc(df_mc, 'R')
plotRGBmc(df_mc, 'G')
plotRGBmc(df_mc, 'B')


#Asia,India,Kanpur8,26.2965,80.3659,0,0
#Asia,India,Kanpur6F,26.129,80.034,0,0