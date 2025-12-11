from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr

#for dealing with files:
import os
import re
from scipy.io import readsav
import h5py
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import time

#for plotting (the rcParams updates are my personal perference to change font and increase fontsize)
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24,\
                     'xtick.labelsize' : 24,\
                     'ytick.labelsize' : 24,\
                     'axes.titlesize' : 24,\
                     'axes.labelsize' : 24,\
                     'date.autoformatter.minute': '%H:%M' })

# all helper functions for downloading and parsing data
import skymap_data_helper

# for contrast adjustment
import cv2

# for resolution increase
from PIL import Image

import importlib
importlib.reload(skymap_data_helper)

import math

import importlib
importlib.reload(skymap_data_helper)


def project_lat_lon(az_arr, el_arr, lat_camera, lon_camera, h, skymap_110_mask):
    '''
    Projected the azimuth and elevation arrays to the specified height.
    
    Parameters: 
        az_arr = 2D azimuth array for each pixel (NaNs ok, degrees, xarray)
        el_arr = 2D elevation array for each pixel (NaNs ok, degrees, xarray, no need to be filtered)
        lat_camera = latitude of camera (degrees) --> shape [480, 553]
        lon_camera = longitude of camera (degrees) --> shape [480, 553]
        h = height you want to project azimuth and elevation to to get latitude and longitude for each pixel
        skymap_110_mask = mask for where the latitude and longitude are NaNs in the ground truth lat110 and lon110 projections

    Outputs:
        lat_aurora_arr = latitudes of the aurora projected to given height h (2D array same dimensions as original image, give projected latitude of each pixel)
        lon_aurora_arr = longitudes of the aurora project to give height h (2D array same dimensions as original image, give projected longitude of each pixel)
    '''

    # convert to radians + applying mask 
    el_arr = np.radians(np.array(el_arr))
    az_arr = np.radians(np.array(az_arr))
    
    # create elevation mask (True when valid)
    el_mask = (el_arr > np.radians(5)) & (el_arr < np.radians(90))
    
    # combine with skymap mask: True = valid pixel
    valid_mask = el_mask & (~skymap_110_mask)
    
    # set invalid pixels to NaN
    el_arr[~valid_mask] = np.nan
        
    # horizontal distance between camera and aurora along camera's tangent plane
    d1_arr = h / np.tan(el_arr)

    # decompose horizontal distance into east and north components relative to camera tan plane
    dx_arr = d1_arr * np.sin(az_arr)
    dy_arr = d1_arr * np.cos(az_arr)

    # convert N/E offset components to (lat, lon) --> comes out in decimal degrees 
    lat_delta_arr = dy_arr / 111045 #degrees
    lon_delta_arr = dx_arr / (np.cos(np.radians(lat_camera + lat_delta_arr)) * 111321) 

    # add lat/long offset to camera's og lat/lon to get the lat/lon of the aurora at the chosen height!
    lat_aurora_arr = lat_camera + lat_delta_arr
    lon_aurora_arr = lon_camera + lon_delta_arr

    # apply the same mask again just in case 
    lat_aurora_arr[skymap_110_mask] = np.nan
    lon_aurora_arr[skymap_110_mask] = np.nan

    return lat_aurora_arr, lon_aurora_arr



def plot_lat_lon(yknf_rgb_asi_ds, fsmi_rgb_asi_ds, time_index, site_name_yknf, site_name_fsmi, yknf_lat, yknf_lon, fsmi_lat, fsmi_lon, h_target):
    ''' 
    Plots the projected latitude and longitude of the YKNF and FSMI, and overlays them. 
    Produces 3 plots total: YKNF, FSMI, overlaid, and saves each of the images. 

    Parameters:
        yknf_rgb_asi_ds: xarray of the yknf skymap <xarray>
        fsmi_rgb_asi_ds: xarray of the fsmi skymap <xarray>
        time_index: specific frame we are looking at from the asi ds <int>
        site_name_yknf: string to label the plots
        site_name_fsmi: string to label the plots
        yknf_lat: projected latitude array of one yknf frame <2d arr>
        yknf_lon: projected longitude array of one yknf frame <2d arr>
        fsmi_lat: projected latitude array of one fsmi frame <2d arr>
        fsmi_lon: projected longitude array of one fsmi frame <2d arr>
        h_target: height that yknf & fsmi frames were projected to <int>
        lat_lon_plots: folder to save the 3 images to 

    Outputs:
        3 plots of the YKNF, FSMI, overlaid latitude and longitude
    '''    
    
    R_yknf = yknf_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_yknf = yknf_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_yknf = yknf_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    R_fsmi = fsmi_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_fsmi = fsmi_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_fsmi = fsmi_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    
    # Extract time and format it
    raw_time = yknf_rgb_asi_ds.times.values[time_index]
    time_obj = pd.to_datetime(raw_time.decode("utf-8").replace(" UTC", ""))
    time_str = time_obj.strftime("%b. %d, %Y %H:%M:%S UT")
        
    # contrast adjustment: alpha=contrast, beta=brightness
    alpha = 5
    beta = 5
    rgb_yknf = np.stack([R_yknf, G_yknf, B_yknf], axis=-1)  # shape: (x, y, 3)
    rgb_fsmi = np.stack([R_fsmi, G_fsmi, B_fsmi], axis=-1)  # shape: (x, y, 3)
    
    rgb_yknf_adjusted = cv2.convertScaleAbs(rgb_yknf, alpha=alpha, beta=beta)
    rgb_fsmi_adjusted = cv2.convertScaleAbs(rgb_fsmi, alpha=alpha, beta=beta)
    
    # yknf projected
    fig1, ax1 = plt.subplots(figsize=(8,8))
    scat1 = ax1.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf_adjusted.reshape(-1, 3)/256,s=1)
    #plt.xlim(ax_skymap.get_xlim())
    #plt.ylim(ax_skymap.get_ylim())
    ax1.set_ylabel("Latitude (deg)")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_title(f"{h_target/1000}km Projection: {site_name_yknf} – {time_str}", pad=30);
    plt.show()
    
    # fsmi projected
    fig2, ax2 = plt.subplots(figsize=(8,8))
    scat2 = ax2.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi_adjusted.reshape(-1, 3)/256,s=1)
    #plt.xlim(ax_skymap.get_xlim())
    #plt.ylim(ax_skymap.get_ylim())
    ax2.set_ylabel("Latitude (deg)")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_title(f"{h_target/1000}km Projection: {site_name_fsmi} – {time_str}", pad=30);
    plt.show()
    
    # 110km  overlaid --> 
    plt.figure(figsize=(8,8))
    plt.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf_adjusted.reshape(-1, 3)/256,s=1, alpha=0.5)
    plt.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi_adjusted.reshape(-1, 3)/256,s=1, alpha=0.02)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"Overlaid {h_target/1000}km Projection - {time_str}", pad=30)
    #plt.xlim(ax_skymap.get_xlim())
    #plt.ylim(ax_skymap.get_ylim())
    plt.show()

    return rgb_yknf_adjusted, rgb_fsmi_adjusted


def mod_plot_lat_lon(yknf_rgb_asi_ds, fsmi_rgb_asi_ds, time_index, site_name_yknf, site_name_fsmi, yknf_lat, yknf_lon, fsmi_lat, fsmi_lon, h_target):
    ''' 
    Plots the projected latitude and longitude of the YKNF and FSMI, and overlays them. 
    Produces 3 plots total: YKNF, FSMI, overlaid, and saves each of the images. 

    Parameters:
        yknf_rgb_asi_ds: xarray of the yknf skymap <xarray>
        fsmi_rgb_asi_ds: xarray of the fsmi skymap <xarray>
        time_index: specific frame we are looking at from the asi ds <int>
        site_name_yknf: string to label the plots
        site_name_fsmi: string to label the plots
        yknf_lat: projected latitude array of one yknf frame <2d arr>
        yknf_lon: projected longitude array of one yknf frame <2d arr>
        fsmi_lat: projected latitude array of one fsmi frame <2d arr>
        fsmi_lon: projected longitude array of one fsmi frame <2d arr>
        h_target: height that yknf & fsmi frames were projected to <int>
        lat_lon_plots: folder to save the 3 images to 

    Outputs:
        3 plots of the YKNF, FSMI, overlaid latitude and longitude
    '''    
    
    R_yknf = yknf_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_yknf = yknf_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_yknf = yknf_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    R_fsmi = fsmi_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_fsmi = fsmi_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_fsmi = fsmi_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    
    # Extract time and format it
    raw_time = yknf_rgb_asi_ds.times.values[time_index]
    time_obj = pd.to_datetime(raw_time.decode("utf-8").replace(" UTC", ""))
    time_str = time_obj.strftime("%b. %d, %Y %H:%M:%S UT")
        
    # contrast adjustment: alpha=contrast, beta=brightness
    alpha = 5
    beta = 5
    rgb_yknf = np.stack([R_yknf, G_yknf, B_yknf], axis=-1)  # shape: (x, y, 3)
    rgb_fsmi = np.stack([R_fsmi, G_fsmi, B_fsmi], axis=-1)  # shape: (x, y, 3)
    
    rgb_yknf_adjusted = cv2.convertScaleAbs(rgb_yknf, alpha=alpha, beta=beta)
    rgb_fsmi_adjusted = cv2.convertScaleAbs(rgb_fsmi, alpha=alpha, beta=beta)
    
    # # yknf projected
    # fig1, ax1 = plt.subplots(figsize=(8,8))
    # scat1 = ax1.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf_adjusted.reshape(-1, 3)/256,s=1)
    # #plt.xlim(ax_skymap.get_xlim())
    # #plt.ylim(ax_skymap.get_ylim())
    # ax1.set_ylabel("Latitude (deg)")
    # ax1.set_xlabel("Longitude (deg)")
    # ax1.set_title(f"{h_target/1000}km Projection: {site_name_yknf} – {time_str}", pad=30);
    # plt.show()
    
    # # fsmi projected
    # fig2, ax2 = plt.subplots(figsize=(8,8))
    # scat2 = ax2.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi_adjusted.reshape(-1, 3)/256,s=1)
    # #plt.xlim(ax_skymap.get_xlim())
    # #plt.ylim(ax_skymap.get_ylim())
    # ax2.set_ylabel("Latitude (deg)")
    # ax2.set_xlabel("Longitude (deg)")
    # ax2.set_title(f"{h_target/1000}km Projection: {site_name_fsmi} – {time_str}", pad=30);
    # plt.show()
    
    # # 110km  overlaid --> 
    # plt.figure(figsize=(8,8))
    # plt.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf_adjusted.reshape(-1, 3)/256,s=1, alpha=0.5)
    # plt.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi_adjusted.reshape(-1, 3)/256,s=1, alpha=0.02)
    # plt.xlabel("Longitude (deg)")
    # plt.ylabel("Latitude (deg)")
    # plt.title(f"Overlaid {h_target/1000}km Projection - {time_str}", pad=30)
    # #plt.xlim(ax_skymap.get_xlim())
    # #plt.ylim(ax_skymap.get_ylim())
    # plt.show()

    return rgb_yknf_adjusted, rgb_fsmi_adjusted
