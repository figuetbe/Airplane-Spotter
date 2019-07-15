# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:10:04 2019

@author: figu
"""


import numpy as np
import pandas as pd
import math
import itertools
from scipy.spatial import distance
from opensky_api import OpenSkyApi
from pyproj import Proj, transform
from bokeh.models import BoxZoomTool
from bokeh.plotting import figure


#bbox = (47, 48, 8, 9)  # Zurich area

os_columns = ['icao24',
              'callsign',
              'origin_country',
              'time_position',
              'longitude',
              'latitude',
              'geo_altitude',
              'on_ground',
              'velocity',
              'heading',
              'sensors',
              'baro_altitude',
              'squawk',
              'spi',
              'position_source']

def ll2wm(lat, lon):
    outProj = Proj(init='epsg:3857')
    inProj = Proj(init='epsg:4326')
    return transform(inProj, outProj, lat, lon)   # y, x


def bbox2range(bbox):
    y0, x0 = ll2wm(bbox[0], bbox[2])
    yf, xf = ll2wm(bbox[1], bbox[3])
    return (x0, xf), (y0, yf)


def get_plane_data(bbox):
    api = OpenSkyApi()
    df_planes = pd.DataFrame(columns=['X','Y'])
    # bbox = (min latitude, max latitude, min longitude, max longitude)
    states = api.get_states(bbox = bbox)
    planes_coordinates = []
    for s in states.states:
        x, y = ll2wm(s.longitude, s.latitude)
        df_planes = df_planes.append(pd.DataFrame([[x,y]], columns = df_planes.columns))
    return df_planes


def get_plane_full_data(bbox):
    api = OpenSkyApi()
    df_planes = pd.DataFrame(columns=os_columns)
    # bbox = (min latitude, max latitude, min longitude, max longitude)
    try:
        states = api.get_states(bbox=bbox)
    except:
        print('Open Sky Network API is not responding')
    for s in states.states:
        df_planes = df_planes.append(pd.DataFrame
                                     ([[s.icao24,
                                        s.callsign,
                                        s.origin_country,
                                        s.time_position,
                                        s.longitude,
                                        s.latitude,
                                        s.geo_altitude,
                                        s.on_ground,
                                        s.velocity,
                                        s.heading,
                                        s.sensors,
                                        s.baro_altitude,
                                        s.squawk,
                                        s.spi,
                                        s.position_source]],
                                      columns=df_planes.columns))
    return df_planes


def base_plot(tools, plot_width, plot_height, x_range, y_range, **plot_args):
    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range, outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0, **plot_args)
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.add_tools(BoxZoomTool(match_aspect=True))
    return p


def gps_to_ecef_custom(row):
    lat = row['latitude']
    lon = row['longitude']
    alt = row['geo_altitude']
    rad_lat = lat * (math.pi / 180.0)
    rad_lon = lon * (math.pi / 180.0)

    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / math.sqrt(1 - e2 * math.sin(rad_lat) * math.sin(rad_lat))

    x = (v + alt) * math.cos(rad_lat) * math.cos(rad_lon)
    y = (v + alt) * math.cos(rad_lat) * math.sin(rad_lon)
    z = (v * (1 - e2) + alt) * math.sin(rad_lat)

    return (x, y, z)


def get_close_encounters(bbox, dist_close):
    df = get_plane_full_data(bbox)
    df.dropna(axis=0, subset=['latitude', 'longitude', 'geo_altitude'],
              inplace=True)
    df_flying = df.loc[df.on_ground == 0]
    df_flying['XYZ'] = df_flying.apply(lambda row: gps_to_ecef_custom(row),
                                       axis=1)
    df_flying = df_flying.reset_index(drop='True')
    coords = list(df_flying['XYZ'].values)
    distances = distance.cdist(coords, coords, 'euclidean')
    distances[distances == 0] = np.nan
    close_encounters = np.where(distances < dist_close)
    # zip the 2 arrays to get the exact coordinates
    close_encounters = [[close_encounters[0][i],
                         close_encounters[1][i]]
                        for i in range(len(close_encounters[0]))]

    # Then we need to remove the duplicates:
    close_encounters.sort()
    close_encounters = list(close_encounters
                            for close_encounters,_
                            in itertools.groupby(close_encounters))
    df_flying['minDist'] = np.nanmin(distances, axis=1)
    list_close_encounters = [encounter for
                             sublist in close_encounters
                             for encounter in sublist]
    return df_flying.loc[list_close_encounters]
