import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import csv
import h5py
from geopy import distance
import helper_functions
import globals


bounds = globals.bounds(2)

def plot_rain_distribution():
    """Plots rain distribution over distance around chosen center, inside
    Ganghes-Brahmaputra river basin.
    """
    if not os.path.exists('data_results/rain'):
        os.makedirs('data_results/rain')

    dates = ['2013-06-16', '2013-06-17']
    filenames = []

    for date in dates:
        for (dirpath, dirnames, filenames_) in os.walk('data_pmm/raw/{}'.format(date)):
            if not os.path.exists('data_results/rain/{}'.format(date)):
                os.makedirs('data_results/rain/{}'.format(date))

            for filename in filenames_:
                file_specs = re.search('3B-HHR.MS.MRG.3IMERG.(.*?).V06B.HDF5', filename).group(1)

                gpm_data = helper_functions.GPM('data_pmm/raw/{}/'.format(date)+filename, bounds)

                # rain_grid: grid mapping precipitation in mm/h
                # lonmin = 71.95, lonmax = 99.75, latmin = 20.55, latmax = 33.35 (all in degrees)
                # rain_grid indexes [i][j] can be translated to coordinates in the following way:
                # [0][0] --> (lat,lon) = (33.35,71.95)
                # [0][1] --> (lat,lon) = (33.35,72.05)
                # [1][0] --> (lat,lon) = (33.25, 71.95)
                rain_grid = gpm_data.get_crop()
                grid_bounds = gpm_data.get_bounds()

                # center is determined to be at coordinates with higher precipitation
                max_precip = 0
                center_lat_ind, center_lon_ind = 0, 0
                for i in range(len(rain_grid)):
                    for j in range(len(rain_grid[i])):
                        precip = rain_grid[i][j]
                        if precip > max_precip:
                            center_lat_ind, center_lon_ind = i, j

                def index_to_coords(i, j):
                    """Converts latitude index and longitude index to latitude and longitude"""
                    return grid_bounds[3]-0.10*i, grid_bounds[0]+0.10*j

                center_lat, center_lon = index_to_coords(center_lat_ind, center_lon_ind)

                # rain_to_distance = [...,[distance_to_center (km), precipitation (mm/h)],...]
                rain_to_distance = []

                for i in range(len(rain_grid)):
                    for j in range(len(rain_grid[i])):
                        precip = rain_grid[i][j]
                        lat, lon = index_to_coords(i, j)
                        distance_to_center = distance.distance((lat,lon),(center_lat,center_lon)).km
                        rain_to_distance.append([distance_to_center, precip])

                rain_log_binned_to_distance = [0 for i in range(29)]
                distances = [row[0] for row in rain_to_distance]
                max_distance = max(distances)
                bins = np.linspace(0, max_distance+1, 30)
                for i in range(len(bins)-1):
                    bin_min = bins[i]
                    bin_max = bins[i+1]
                    for j in range(len(distances)):
                        if bin_min <= distances[j] < bin_max:
                            rain_log_binned_to_distance[i] += rain_to_distance[j][1]
                log_binned_distance = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]

                plt.plot(log_binned_distance, rain_log_binned_to_distance, 'ko', markerfacecolor='grey')
                plt.ylabel('Precipitation (mm/h)')
                plt.xlabel('Distance(km)')
                plot_name = 'data_results/rain/{}/'.format(date) + file_specs + '.png'
                plt.savefig(plot_name)
                plt.close()

def hoshen_kopelman(rain_grid):
    """Hoshen-Kopelman algorithm for precipitation clusters labeling on rain grid

    Params:
        - rain_grid: rain grid with precipitation at each point.
    """

    # rain grid with 1 where there is precipitation, 0 if there is not
    binary_rain_grid = [[1 if rain_grid[i][j]>0 else 0 for j in range(len(rain_grid[i]))]
                        for i in range(len(rain_grid))]
    # weight dictionary
    M = dict()
    k = 2

    for i in range(len(binary_rain_grid)-1,-1,-1):
        for j in range(len(binary_rain_grid[i])):
            if binary_rain_grid[i][j] == 1:
                try:
                    neighbour_left = binary_rain_grid[i][j-1]
                    neighbour_down = binary_rain_grid[i+1][j]
                    try:
                        if M[neighbour_left] < 0:
                            neighbour_left = (-1)*M[neighbour_left]
                    except KeyError:
                        pass
                    try:
                        if M[neighbour_down] < 0:
                            neighbour_down = (-1)*M[neighbour_down]
                    except KeyError:
                        pass
                    if ( neighbour_left != 0 ) and ( neighbour_down != 0 ):
                        # two neighbour k0
                        if neighbour_left == neighbour_down:
                            M[neighbour_left] += 1
                            binary_rain_grid[i][j] = neighbour_left
                        # one neighbour k0 and one neighbour k1
                        else:
                            M[neighbour_down] += M[neighbour_left] + 1
                            M[neighbour_left] = (-1)*neighbour_down
                            binary_rain_grid[i][j] = neighbour_down
                    # one neighbour k0 (on the left)
                    elif neighbour_left != 0:
                        M[neighbour_left] += 1
                        binary_rain_grid[i][j] = neighbour_left
                    # one neighbour k0 (down)
                    elif neighbour_down != 0:
                        M[neighbour_down] += 1
                        binary_rain_grid[i][j] = neighbour_down
                    # isolated
                    elif ( neighbour_left == 0 ) and ( neighbour_down == 0 ):
                        k += 1
                        M[k] = 1
                        binary_rain_grid[i][j] = k
                except IndexError:
                    # no down neighbour
                    if j-1 in range(len(binary_rain_grid[i])):
                        neighbour_left = binary_rain_grid[i][j-1]
                        if neighbour_left != 0:
                            M[neighbour_left] += 1
                            binary_rain_grid[i][j] = neighbour_left
                        else:
                            k += 1
                            M[k] = 1
                            binary_rain_grid[i][j] = k
                    # no left neighbour
                    elif i+1 in range(len(binary_rain_grid)-1,-1,-1):
                        neighbour_down = binary_rain_grid[i+1][j]
                        if neighbour_down != 0:
                            M[neighbour_down] += 1
                            binary_rain_grid[i][j] = neighbour_down
                        else:
                            k += 1
                            M[k] = 1
                            binary_rain_grid[i][j] = k
                    # no left nor down neighbour (initial position)
                    else:
                        M[k] = 1
                        binary_rain_grid[i][j] = k

    clustered_rain_grid = []
    for i in range(len(binary_rain_grid)):
        row = []
        for j in range(len(binary_rain_grid[i])):
            if binary_rain_grid[i][j] == 0:
                row.append(0)
            else:
                if M[binary_rain_grid[i][j]] < 0:
                    row.append(-1*M[binary_rain_grid[i][j]])
                else:
                    row.append(binary_rain_grid[i][j])
        clustered_rain_grid.append(row)

    return M, clustered_rain_grid


# gpm_data = helper_functions.GPM('data_pmm/raw/2013-06-16/3B-HHR.MS.MRG.3IMERG.20130616-S010000-E012959.0060.V06B.HDF5', bounds)
#
# rain_grid = gpm_data.get_crop()

rain_grid = [[0,0,0,1,0,0,1],
             [1,1,0,1,1,0,0],
             [0,0,0,0,1,0,1],
             [0,0,1,1,1,0,1],
             [0,0,1,0,0,0,0],
             [1,0,1,1,0,1,0],
             [1,1,0,1,0,0,0]]

if __name__ == '__main__':
    # plot_rain_distribution()
    hoshen_kopelman(rain_grid)
