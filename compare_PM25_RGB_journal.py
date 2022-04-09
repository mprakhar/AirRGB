#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 7/11/2017
# Last edit 7/11/2017

# Purpose: Read PM from US Embassy at monrhly or daily level and compare with AirRGB parameters fro accuracy
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
from pandas.tseries.offsets import MonthBegin
from pandas.tseries.offsets import MonthEnd
import seaborn as sns
import os.path
from glob import glob
from datetime import timedelta, date
from dateutil import rrule
from dateutil.relativedelta import relativedelta
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from itertools import product
from sklearn import datasets, linear_model
# import os
# os.chdir('/home/prakhar/Research/AQM_research/Codes/')
import sys

sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
# pvt imports
from spatialop import *
from spatialop import infoFinder as info
from spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file
from spatialop import im_mean_temporal as im_mean
from spatialop import coord_translate as ct





# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
#Input

gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path from my folders
gb_path_wt = '/home/wataru/research/MODIS/MOD04L2//'   # global path to be appended for all files to be sourced from wt

pm = gb_path + r'/Data/Data_process/PM2.5/'

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



gis = Raster_file()
gis.path = gb_path + '/Data/Data_raw/GIS data/' + 'GlobalLandCover_tif/'
gis.sat = 'GIS'
gis.prod = 'LCType'
gis.sample = gb_path + '/Data/Data_raw/GIS data/GlobalLandCover_tif/LCType.tif' + 'LCType.tif'
gis.georef = gb_path + '/Data/Data_raw/GIS data/GlobalLandCover_tif/LCType.tif' + 'LCType.tif'

city = pd.read_csv('RGB/global_megacity2.csv', header=0)


#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut/Air2RGB//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step1: Read/Summarize PM     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#


def get_PM(citychk):

    #openthe PM csvfile
    df_pm = pd.read_csv(pm + citychk +'_PM2.5.csv', header=0)


    #assign datetime
    df_pm['date'] = pd.to_datetime(df_pm['Date'], format = '%m/%d/%Y %H:%M')

    # remove Nan dataset
    df_pm.Value.replace(-999, np.NaN, inplace = True)

    # group at monthly level all the 10 am values
    #df_pm.resample('M', on= 'Date').mean()
    #ndf = df_pm[df_pm.Hour == 10].groupby(['Year', 'Month']).mean()
    ndf = df_pm[df_pm.Hour == 10].resample('D', on= 'date').mean()

    return ndf


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step2: Merge with RGB     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# FUNCTION COPIED FROM air2rgbWT
# create RGB by first finding AOD ANG for that city


def pixval(prod_arr, city, citychk):

    pix = city[city.city==citychk]['pixloc']
    return (prod_arr[list(pix)[0][1], list(pix)[0][0]])


# open all images and gather pix valuez
def allpixval(prodT, start_date, end_date, save_path, citychk):

    # Getting pixel values from lat/long values
    # input>>lat, long   output >>pix_y, pix_x
    city['pixloc'] = ct.latLonToPixel(prodT.georef,
                                      city.apply(lambda x: [x['lat'], x['lon']],axis=1).tolist())
    ls = []
    # run for the dates
    for date_m in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        for prodT.file in glob(os.path.join(prodT.path, '*.' + prodT.prod + '*' + '.Global')): # no .tif extension provided in WT files and .tif extension in OM files

            # this gives date as a list with integer year and month as return
            date_ym = int(prodT.file[-18:-11]) # MODIS, OMI both WT[-17:-11], PM[-21,-15]
            #print date_ym

            if ((str(date_ym) == (str(date_m.year) + str(date_m.timetuple().tm_yday))) & os.path.isfile(prodT.file)):

                print 'file found ', date_ym
                try:
                    prod_arr = prodT.raster_as_array()
                    city[prodT.prod + str(date_ym)] = pixval(prod_arr, city, citychk)
                    ls.append([date_ym, pixval(prod_arr, city, citychk)])

                except AttributeError:
                    print ' seems like missing file  in ', date_ym

    # save to csv
    print 'done 1'
    #city.to_csv(save_path, index=False, header=True)
    # save scale and offset as dataframe
    df_ls = pd.DataFrame(ls, columns=['date', prodT.prod])
    df_ls.to_csv(csv_save_path+'df_'+prodT.prod+citychk+'.csv')
# function end


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
        w = 100.

    # Mask if values equal to zero (no observed data)
    if (xx <= 50) & (yy <= 50):
        v = 0.
        s = 0.
        w = 100.

    return [s,v,w]  #RGB
# funciton end



# Function to generate R,G,B dataframe for a city if AOD ANG dataframe provided
def df_AQgen(df_AQ, vertsT):

    # computing RGB values from AODANG values by calling distRGBG function
    df_AQ['RGB'] = df_AQ.apply(lambda row: distRGB(xx=row['AOD'], yy=row['ANG'], vertsT=vertsT), axis=1)
    df_AQ[['R', 'G', 'B']] = df_AQ['RGB'].apply(pd.Series)

    return df_AQ
#function end

# compute RGB values of city if AOD ANG already exist in a daatframe
def RGBcity(citychk):
    #       vertices as found by wt data simple mean

    #       Deciding vertices of the RGB gradient triangle . order: B,R,G
    vertsT = [[50, 50], [600, 800], [20, 1000]]

    # read df that has AOD and ANG fro city chek concerned
    df_AQin = pd.read_csv(csv_save_path+'df_AQ'+citychk+'.csv', header=0)

    # generate dataframe of R,G,B and AOD,ANG a particualr city
    df_AQ = df_AQgen(df_AQin.dropna(axis=0, how='any'), vertsT)

    # save the info so that i t can be plotted as excel later on
    df_AQ.to_csv(csv_save_path+'df_AQRGB_daily'+citychk+'.csv', index=False, header=True)

    return df_AQ

# run the functions
start_date = date(2015, 1, 1)
end_date = date(2017, 1, 1)
aod.path = '/home/wataru/research/MODIS/MOD04L2/L3/'
ang.path = '/home/wataru/research/MODIS/MOD04L2/L3/'

citychk ='Beijing'
save_path =  csv_save_path+'df_'+prodT.prod+citychk+'0.csv'
df_AQ = RGBcity(citychk)



# ---------    O P E R A T I N G   O N   C S V

#open the RGB csvfile
df_rgb = pd.read_csv(csv_save_path+'df_AQRGB_daily'+citychk+'.csv', header=0)

# assign dattime
#df_rgb['date'] = pd.to_datetime(df_rgb['date'], format = '%Y%m')
#df_rgb['date'] = pd.to_datetime(df_rgb['date'], format = "%Y%m") + MonthEnd(1)
df_rgb['date'] = pd.to_datetime(df_rgb['date'], format = "%Y%j")

#make date as index
df_rgb.set_index('date', inplace=True)

# also read PM
ndf = get_PM(citychk)

# merge PM and RGB
df_mer = ndf.join(df_rgb[[ 'AOD', 'ANG', 'R', 'G', 'B']], how = 'inner')

# save befor plotting
df_mer.to_csv(csv_save_path + 'df_RGBPM_'+citychk+'2.csv', header=True, index=True)


# Commands to generate dataset used in the paper
# 1. df_mer[(df_mer.AOD>0)&(df_mer.AOD<800)&(df_mer.ANG>0)&(df_mer.ANG<1000)].to_csv(csv_save_path+'df_RGBPM_NewDelhi2.csv', header=True, index=True)
# 2. df_mer[(df_mer.AOD>0)&(df_mer.ANG>0)&(df_mer.Value>0)].to_csv(csv_save_path + 'df_RGBPM_'+citychk+'3v3.csv', header=True, index=True)