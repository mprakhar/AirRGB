#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 11/15/2017
# Last edit 11/15/2017

# Purpose: Torun the analysis for  air2rgbWT_20city and compare_PM25_RGB


import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import sys
from matplotlib import pyplot as plt
import numpy as np

sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
# pvt imports
from spatialop import *
# pvt imports
from spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file

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

ang = Raster_file()
ang.path = gb_path_wt + r'/L4//'
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

#city locations to be cinsidered
#city = pd.read_csv('RGB/global_megacity2.csv', header=0)
citypath =  gb_path + r'/Codes/AirRGB//lib/globalindia_megacity3.csv'

#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut/Air2RGB//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'



def get_AERONET():
    # opening the AERONET for Kanpur
    df_stn = pd.read_csv(r'/home/prakhar/Research/AQM_research//Data/Data_process/AQ/AERONET/20010101_20160530_Kanpur/20010101_20160530_Kanpur.lev15', header=6)
    df_stn = df_stn.rename(columns={'440-870_Angstrom_Exponent': 'ANG440_870', 'Day_of_Year(Fraction)': 'doyf', 'AOD_500nm':'AOD500', 'AOD_440nm':'AOD440','NO2(Dobson)': 'NO2', 'Date(dd-mm-yyyy)':'dmy' })
    df_stn2 = df_stn[['dmy', 'doyf', 'AOD500', 'AOD440', 'ANG440_870', 'NO2']]

    #making the datetiem as appropriate
    df_stn2['date'] = pd.to_datetime(df_stn2['dmy'], format = '%d:%m:%Y')
    df_stn2['date1'] = pd.to_datetime(df_stn2['dmy'], format = '%d:%m:%Y')
    df_stn2['year'] = pd.DatetimeIndex(df_stn2.date).year
    #make date as index
    df_stn2.set_index('date1', inplace=True)

    #doyf is the day of yea rfraction
    df_stn2 = df_stn2[(df_stn2['doyf'] % 1.0 > 4.3 / 24.0) & (df_stn2['doyf'] % 1.0 < 5.7 / 24.0)]

    #et rid of funny alues
    mask = df_stn2.AOD500<0
    df_stn2.loc[mask, 'AOD500']=np.nan

    #get daily mean aeronet aod
    df_stn2 = df_stn2.resample('D').mean()

    return df_stn2




# ---------    O P E R A T I N G   O N   C S V
def get_aqrgb():
    # name of the city
    citychk = 'Kanpur'

    #open the RGB csvfile. This csv file contais daily level AOD, ANG and RGB values.
    df_rgb = pd.read_csv(csv_save_path+'df_AQRGB_daily'+citychk+'.csv', header=0)

    # assign dattime
    #df_rgb['date'] = pd.to_datetime(df_rgb['date'], format = '%Y%m')
    #df_rgb['date'] = pd.to_datetime(df_rgb['date'], format = "%Y%m") + MonthEnd(1)
    df_rgb['date'] = pd.to_datetime(df_rgb['date'], format = "%Y%j")
    df_rgb['date1'] = pd.to_datetime(df_rgb['date'], format = "%Y%j")
    df_rgb['year'] = pd.DatetimeIndex(df_rgb.date).year
    df_rgb['month'] = pd.DatetimeIndex(df_rgb.date).month
    #make date as index
    df_rgb.set_index('date1', inplace=True)

    return df_rgb



# polot
def plotMODIS_AERONET(year, df_t, df_tA, rem_zero=False):
    # start plotting
    plt.figure(figsize=(16, 3))

    # MODIS

    #how to treat zeros remove zero
    if rem_zero:
        # daily dataplot
        ax = plt.subplot(111)
        x = df_t.date
        y_daily = df_t.AOD
        ax.plot(x, y_daily, label='daily (MODIS)')

        # monthly dataplot
        y_daily = y_daily.replace(0,np.NaN)
        y_monthly = y_daily.groupby(pd.TimeGrouper(freq='M')).mean()
        ax.plot(y_monthly, label='monthly mean (MODIS)')

    else:

        # daily dataplot
        ax = plt.subplot(111)
        x = df_t.date
        y_daily = df_t.AOD
        ax.plot(x,y_daily, label = 'daily (MODIS)')

        #monthly dataplot
        y_monthly = y_daily.groupby(pd.TimeGrouper(freq='M')).mean()
        ax.plot(y_monthly, label = 'monthly mean (MODIS)')


    # AERONET

    # daily dataplot
    ax = plt.subplot(111)
    #x = df_t.date
    if rem_zero:
    y_daily = df_tA.AOD500*1000
    if rem_zero:
        y_daily = y_daily.replace(0, np.NaN)
    ax.plot(y_daily, linestyle = '-.',label = 'daily (AERONET)')

    #monthly dataplot
    y_monthlyAER = y_daily.groupby(pd.TimeGrouper(freq='M')).mean()
    ax.plot(y_monthlyAER,  linestyle = '-.', label = 'monthly mean (AERONET)')

    plt.legend()
    plt.xlabel('month')
    plt.ylabel('AOD')
    plt.title(str(year))
    plt.show()

    return y_monthlyAER, y_monthly
#function end

def get_dfsubset(year, df_aer, df_rgb, yearcond = True):

    # make test df
    df_t = df_rgb.replace(0, np.nan)
    if yearcond:
        df_t = df_t[df_rgb.year == year]


    # let plot the same info for AERONET
    # make test df
    df_tA = df_aer.replace(0, np.nan)
    if yearcond:
        df_tA = df_tA[df_aer.year == year]


    return df_tA, df_t
#funciton end



# get df of AERONET
df_aer = get_AERONET()

# get the df of aq and rgb
df_rgb = get_aqrgb()





#lets ouick a year for analysis
year = 2001

#for year in range(2001, 2017):
year = 2005
df_tA, df_t = get_dfsubset(year, df_aer, df_rgb, yearcond = True)
y_monthlyAER, y_monthly = plotMODIS_AERONET(year, df_t, df_tA, rem_zero=True )


#overall
y_monthlyAER, y_monthly = plotMODIS_AERONET(year, df_rgb, df_aer, rem_zero=True )

#getting RMSE monthwise
y_diff = np.abs(y_monthlyAER - y_monthly)**2 #np.abs

#get monthly mean
y_diffm = y_diff.resample('M').mean()#**(0.5)

#data at month level
y_month = y_diff.groupby(y_diff.index.month).mean()**(0.5)

plt.figure()
#plt.plot(y_diffm)
plt.plot(y_month)


#data after iterpolating, bicubic interpolation



#stl of the monthly data
a =sm.tsa.seasonal_decompose(df_rgb.R, model = 'additive', freq=365)



