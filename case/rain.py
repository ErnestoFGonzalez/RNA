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

def plot_rain_distribution(rain_grid, grid_bounds, file_name):
    """Plots rain distribution over distance around chosen center, inside
    Ganghes-Brahmaputra river basin.
    """
    print("\tRain distribution")
    # center is determined to be at coordinates with higher precipitation
    max_precip = 0
    center_lat_ind, center_lon_ind = 0, 0
    for i in range(len(rain_grid)):
        for j in range(len(rain_grid[i])):
            precip = rain_grid[i][j]
            if precip > max_precip:
                center_lat_ind, center_lon_ind = i, j

    center_lat, center_lon = index_to_coords(ij=(center_lat_ind,center_lon_ind), grid_bounds=grid_bounds)

    # rain_to_distance = [...,[distance_to_center (km), precipitation (mm/h)],...]
    rain_to_distance = []

    for i in range(len(rain_grid)):
        for j in range(len(rain_grid[i])):
            precip = rain_grid[i][j]
            lat, lon = index_to_coords(ij=(i,j), grid_bounds=grid_bounds)
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
    plot_name = 'data_results/rain/{}/'.format(date) + file_name + '-rain-distribution' + '.png'
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

    clustered_positions = []
    for i in range(len(binary_rain_grid)-1,-1,-1):
        for j in range(len(binary_rain_grid[i])):
            if binary_rain_grid[i][j] == 1:
                # not in left limit neither inferior limit
                if ( j != 0 ) and ( i != len(binary_rain_grid)-1 ):
                    neighbour_left = binary_rain_grid[i][j-1]
                    neighbour_down = binary_rain_grid[i+1][j]
                    # two neighbours
                    if ( neighbour_left != 0 ) and ( neighbour_down != 0 ):
                        # two neighbours k0
                        if neighbour_left == neighbour_down:
                            M[neighbour_left] += 1
                            binary_rain_grid[i][j] = neighbour_left
                        # one neighbour k0 and one neighbour k1
                        else:
                            M[neighbour_down] += M[neighbour_left] + 1
                            # M[neighbour_left] = (-1)*neighbour_down
                            binary_rain_grid[i][j] = neighbour_down
                            for m, n in clustered_positions:
                                if binary_rain_grid[m][n] == neighbour_left:
                                    binary_rain_grid[m][n] = neighbour_down
                            del M[neighbour_left]
                    # one neighbour k0 (down)
                    elif ( neighbour_left == 0 ) and ( neighbour_down != 0 ):
                        M[neighbour_down] += 1
                        binary_rain_grid[i][j] = neighbour_down
                    # one neighbour k0 (left)
                    elif ( neighbour_left != 0 ) and ( neighbour_down == 0 ):
                        M[neighbour_left] += 1
                        binary_rain_grid[i][j] = neighbour_left
                    # isolated
                    else:
                        k += 1
                        M[k] = 1
                        binary_rain_grid[i][j] = k
                # in left limit but not inferior limit
                elif ( j == 0 ) and ( i != len(binary_rain_grid)-1 ):
                    neighbour_down = binary_rain_grid[i+1][j]
                    # one neighbour k0 (down)
                    if neighbour_down != 0:
                        M[neighbour_down] += 1
                        binary_rain_grid[i][j] = neighbour_down
                    # isolated
                    else:
                        k += 1
                        M[k] = 1
                        binary_rain_grid[i][j] = k
                # not in left limit but in inferior limit
                elif ( j != 0 ) and ( i == len(binary_rain_grid)-1 ):
                    neighbour_left = binary_rain_grid[i][j-1]
                    # one neighbour k0 (left)
                    if neighbour_left != 0:
                        M[neighbour_left] += 1
                        binary_rain_grid[i][j] = neighbour_left
                    # isolated
                    else:
                        k += 1
                        M[k] = 1
                        binary_rain_grid[i][j] = k
                # low-left corner (initial position)
                else:
                    M[k] = 1
                    binary_rain_grid[i][j] = k
                clustered_positions.append([i,j])

    def get_cluster_center_of_mass(cluster):
        x_CM = 0
        y_CM = 0
        total_M = 0

        for i in range(len(binary_rain_grid)):
            for j in range(len(binary_rain_grid[i])):
                # if point belongs to cluster
                if binary_rain_grid[i][j] == cluster:
                    x_CM += i*rain_grid[i][j]
                    y_CM += j*rain_grid[i][j]
                    total_M += rain_grid[i][j]
        x_CM = x_CM / total_M
        y_CM = y_CM / total_M
        return x_CM, y_CM

    new_M = dict()
    for key in M:
        if M[key] > 0:
            new_val = {"size": M[key], 'CM': get_cluster_center_of_mass(key)}
            new_M[key] = new_val
    M = new_M
    return M, binary_rain_grid


def get_rain_clusters_coords(rain_grid, grid_bounds):
    M, clustered_rain_grid = hoshen_kopelman(rain_grid)
    df = pd.DataFrame.from_dict(M, orient='index')
    df.index.name = 'Cluster'
    df['Lat_Lon_CM'] = [index_to_coords(ij=ij, grid_bounds=grid_bounds) for ij in df.loc[:, 'CM']]
    df = df.drop(columns=['CM'])

    return df


def plot_rain_clusters_size_distribution(rain_grid, file_name):
    M, clustered_rain_grid = hoshen_kopelman(rain_grid)
    df = pd.DataFrame.from_dict(M, orient='index')
    df.index.name = 'Cluster'
    size_max = df['size'].max()
    bins = np.logspace(0, np.log10(size_max+1), 15)
    size_distribution = np.histogram(df['size'], bins=bins)
    bins_mean_size = []
    for i in range(len(size_distribution[1])-1):
        bin_min = size_distribution[1][i]
        bin_max = size_distribution[1][i+1]
        bins_mean_size.append(( bin_max+bin_min ) / 2)
    plt.plot(bins_mean_size[:-5], size_distribution[0][:-5], 'ko', markerfacecolor='grey')
    plt.yscale('log')
    plt.ylabel('Rain clusters frequency')
    plt.xlabel('Cluster Size')
    plot_name = 'data_results/rain/{}/'.format(date) + file_name + '-rain-clusters-size-distribution' + '.png'
    plt.savefig(plot_name)
    plt.close()


def plot_rain_clusters_distribution(rain_grid, grid_bounds, file_name):
    """Plot rain clusters distribution over space"""
    print("\tRain clusters distribution")
    M, clustered_rain_grid = hoshen_kopelman(rain_grid)

    df = pd.DataFrame.from_dict(M, orient='index',)
    df.index.name = 'Cluster'

    center_of_grid = (len(rain_grid[0])/2,len(rain_grid)/2)
    center_of_grid = index_to_coords(ij=center_of_grid, grid_bounds=grid_bounds)
    df['Lat_Lon_CM'] = [index_to_coords(ij=ij, grid_bounds=grid_bounds) for ij in df.loc[:, 'CM']]
    df = df.drop(columns=['CM'])
    df['Dist_To_Center'] = df['Lat_Lon_CM'].apply(distance.distance, args=center_of_grid)
    df['Dist_To_Center'] = [x.km for x in df.loc[:, 'Dist_To_Center']]

    spatial_distribution = np.histogram(df['Dist_To_Center'], bins=20)

    bins_mean_distance = []
    for i in range(len(spatial_distribution[1])-1):
        bin_min = spatial_distribution[1][i]
        bin_max = spatial_distribution[1][i+1]
        bins_mean_distance.append(( bin_max+bin_min ) / 2)

    plt.plot(bins_mean_distance, spatial_distribution[0], '-ko', markerfacecolor='grey')
    plt.ylabel('Rain cluster frequency')
    plt.xlabel('Distance to center of grid (km)')
    plot_name = 'data_results/rain/{}/'.format(date) + file_name + '-rain-clusters-distribution' + '.png'
    plt.savefig(plot_name)
    plt.close()


def index_to_coords(ij, grid_bounds):
    """Converts latitude index and longitude index to latitude and longitude"""
    # rain_grid: grid mapping precipitation in mm/h
    # lonmin = 71.95, lonmax = 99.75, latmin = 20.55, latmax = 33.35 (all in degrees)
    # rain_grid indexes [i][j] can be translated to coordinates in the following way:
    # [0][0] --> (lat,lon) = (33.35,71.95)
    # [0][1] --> (lat,lon) = (33.35,72.05)
    # [1][0] --> (lat,lon) = (33.25, 71.95)
    i, j = ij[0], ij[1]
    return grid_bounds[3]-0.10*i, grid_bounds[0]+0.10*j




def rain_pair_distribution(rain_grid, grid_bounds, file_name):
    """Plots spatial distribution of rain pairs. A rain pair exists when two
    random points have precipitation."""
    print("\tRain pair distribution")

    rain_points = []
    for i in range(len(rain_grid)):
        for j in range(len(rain_grid[i])):
            if rain_grid[i][j] != 0:
                latlon = index_to_coords(ij=[i,j], grid_bounds=grid_bounds)
                rain_points.append(latlon)

    rain_pair_dists = []
    for latlon0 in rain_points:
        for latlon1 in rain_points:
            if latlon0 != latlon1:
                dist = distance.distance(latlon0, latlon1).km
                rain_pair_dists.append(dist)

    csv_name = 'data_results/rain/{}/'.format(date) + file_name + '-rain-pair-distances' + '.csv'
    with open(csv_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(map(lambda row: [row], rain_pair_dists))

    spatial_distribution = np.histogram(rain_pair_dists, bins=20)

    bins_mean_distance = []
    for i in range(len(spatial_distribution[1])-1):
        bin_min = spatial_distribution[1][i]
        bin_max = spatial_distribution[1][i+1]
        bins_mean_distance.append(( bin_max+bin_min ) / 2)

    plt.plot(bins_mean_distance, spatial_distribution[0], 'ko', markerfacecolor='grey')
    plt.ylabel('Rain pair frequency')
    plt.xlabel('Distance between pair (km)')
    plot_name = 'data_results/rain/{}/'.format(date) + file_name + '-rain-pair-distribution' + '.png'
    plt.savefig(plot_name)
    plt.close()


def rain_clusters_pair_distribution(rain_grid, grid_bounds, file_name):
    """Plots spatial distribution of rain cluster pair."""
    print("\tRain clusters pair distribution")

    clusters = get_rain_clusters_coords(rain_grid=rain_grid, grid_bounds=grid_bounds)

    rain_clusters_pair_dists = []
    for latlon0 in clusters['Lat_Lon_CM']:
        for latlon1 in clusters['Lat_Lon_CM']:
            if latlon0 != latlon1:
                dist = distance.distance(latlon0, latlon1).km
                rain_clusters_pair_dists.append(dist)

    csv_name = 'data_results/rain/{}/'.format(date) + file_name + '-rain-clusters-pair-distances' + '.csv'
    with open(csv_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(map(lambda row: [row], rain_clusters_pair_dists))

    spatial_distribution = np.histogram(rain_clusters_pair_dists, bins=20)

    bins_mean_distance = []
    for i in range(len(spatial_distribution[1])-1):
        bin_min = spatial_distribution[1][i]
        bin_max = spatial_distribution[1][i+1]
        bins_mean_distance.append(( bin_max+bin_min ) / 2)

    plt.loglog(bins_mean_distance, spatial_distribution[0], 'ko', markerfacecolor='grey')
    plt.ylabel('Rain clusters pair frequency')
    plt.xlabel('Distance between pair (km)')
    plot_name = 'data_results/rain/{}/'.format(date) + file_name + '-rain-clusters-pair-distribution-loglog' + '.png'
    plt.savefig(plot_name)
    plt.close()


if __name__ == '__main__':
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
                print("Working on {}".format(file_specs))
                gpm_data = helper_functions.GPM('data_pmm/raw/{}/'.format(date)+filename, bounds)
                rain_grid = gpm_data.get_crop()
                grid_bounds = gpm_data.get_bounds()

                # plot_rain_distribution(rain_grid=rain_grid, grid_bounds=grid_bounds, file_name=file_specs)
                # plot_rain_clusters_distribution(rain_grid=rain_grid, grid_bounds=grid_bounds, file_name=file_specs)
                plot_rain_clusters_size_distribution(rain_grid=rain_grid, file_name=file_specs)
                # rain_pair_distribution(rain_grid=rain_grid, grid_bounds=grid_bounds, file_name=file_specs)
