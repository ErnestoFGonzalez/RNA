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

padma = pd.read_pickle('data_gloric/padma_gloric_1m3_final_no_geo.pkl')
padma = padma.set_index('Reach_ID',drop=True)
padma = padma[['Next_down', 'Length_km']]


class RainGrid:
    def __init__(self, filename, date):
        gpm_data = helper_functions.GPM('data_pmm/raw/{}/'.format(date)+filename, bounds)
        self.rain_grid = gpm_data.get_crop()
        self.grid_bounds = gpm_data.get_bounds()
        self.file_name = re.search('3B-HHR.MS.MRG.3IMERG.(.*?).V06B.HDF5', filename).group(1)
        self.date = date


    def plot_rain_distribution(self):
        """Plots rain distribution over distance around chosen center, inside
        Ganghes-Brahmaputra river basin.
        """
        print("\tRain distribution")
        # center is determined to be at coordinates with higher precipitation
        max_precip = 0
        center_lat_ind, center_lon_ind = 0, 0
        for i in range(len(self.rain_grid)):
            for j in range(len(self.rain_grid[i])):
                precip = self.rain_grid[i][j]
                if precip > max_precip:
                    center_lat_ind, center_lon_ind = i, j

        center_lat, center_lon = index_to_coords(ij=(center_lat_ind,center_lon_ind), grid_bounds=grid_bounds)

        # rain_to_distance = [...,[distance_to_center (km), precipitation (mm/h)],...]
        rain_to_distance = []

        for i in range(len(self.rain_grid)):
            for j in range(len(self.rain_grid[i])):
                precip = self.rain_grid[i][j]
                lat, lon = index_to_coords(ij=(i,j), grid_bounds=self.grid_bounds)
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
        plot_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-distribution' + '.png'
        plt.savefig(plot_name)
        plt.close()


    def hoshen_kopelman(self):
        """Hoshen-Kopelman algorithm for precipitation clusters labeling on rain grid
        """
        # rain grid with 1 where there is precipitation, 0 if there is not
        binary_rain_grid = [[1 if self.rain_grid[i][j]>0 else 0 for j in range(len(self.rain_grid[i]))]
                            for i in range(len(self.rain_grid))]
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
                        x_CM += i*self.rain_grid[i][j]
                        y_CM += j*self.rain_grid[i][j]
                        total_M += self.rain_grid[i][j]
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


    def get_rain_clusters_coords(self):
        M, clustered_rain_grid = self.hoshen_kopelman()
        df = pd.DataFrame.from_dict(M, orient='index')
        df.index.name = 'Cluster'
        df['Lat_Lon_CM'] = [index_to_coords(ij=ij, grid_bounds=self.grid_bounds) for ij in df.loc[:, 'CM']]
        df = df.drop(columns=['CM'])

        return df


    def plot_rain_clusters_size_distribution(self):
        """Plots rain clusters size histogram"""
        print("\tRain clusters size distribution")
        M, clustered_rain_grid = self.hoshen_kopelman()
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
        plot_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-clusters-size-distribution' + '.png'
        plt.savefig(plot_name)
        plt.close()


    def plot_rain_clusters_distribution(self):
        """Plots rain clusters distribution over space"""
        print("\tRain clusters distribution")
        M, clustered_rain_grid = self.hoshen_kopelman()

        df = pd.DataFrame.from_dict(M, orient='index',)
        df.index.name = 'Cluster'

        center_of_grid = (len(self.rain_grid[0])/2,len(self.rain_grid)/2)
        center_of_grid = index_to_coords(ij=center_of_grid, grid_bounds=self.grid_bounds)
        df['Lat_Lon_CM'] = [index_to_coords(ij=ij, grid_bounds=self.grid_bounds) for ij in df.loc[:, 'CM']]
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
        plot_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-clusters-distribution' + '.png'
        plt.savefig(plot_name)
        plt.close()


    def rain_pair_distribution(self):
        """Plots spatial distribution of rain pairs. A rain pair exists when two
        random points have precipitation."""
        print("\tRain pair distribution")

        rain_points = []
        for i in range(len(self.rain_grid)):
            for j in range(len(self.rain_grid[i])):
                if self.rain_grid[i][j] != 0:
                    latlon = index_to_coords(ij=[i,j], grid_bounds=self.grid_bounds)
                    rain_points.append(latlon)

        rain_pair_dists = []
        for latlon0 in rain_points:
            for latlon1 in rain_points:
                if latlon0 != latlon1:
                    dist = distance.distance(latlon0, latlon1).km
                    rain_pair_dists.append(dist)

        spatial_distribution = np.histogram(rain_pair_dists, bins=20)

        # Note: We have counted every rain pair twice, so now we'll halve
        # the frequency of distances in each bin
        spatial_distribution[0] = [freq/2 for freq in spatial_distribution[0]]

        bins_mean_distance = []
        for i in range(len(spatial_distribution[1])-1):
            bin_min = spatial_distribution[1][i]
            bin_max = spatial_distribution[1][i+1]
            bins_mean_distance.append(( bin_max+bin_min ) / 2)

        plt.plot(bins_mean_distance, spatial_distribution[0], 'ko', markerfacecolor='grey')
        plt.ylabel('Rain pair frequency')
        plt.xlabel('Distance between pair (km)')
        plot_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-pair-distribution' + '.png'
        plt.savefig(plot_name)
        plt.close()


    def rain_clusters_pair_distribution(self):
        """Plots spatial distribution of rain cluster pair."""
        print("\tRain clusters pair distribution")

        clusters = self.get_rain_clusters_coords()

        rain_clusters_pair_dists = []
        for latlon0 in clusters['Lat_Lon_CM']:
            for latlon1 in clusters['Lat_Lon_CM']:
                if latlon0 != latlon1:
                    dist = distance.distance(latlon0, latlon1).km
                    rain_clusters_pair_dists.append(dist)

        csv_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-clusters-pair-distances' + '.csv'
        with open(csv_name, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerows(map(lambda row: [row], rain_clusters_pair_dists))

        spatial_distribution = np.histogram(rain_clusters_pair_dists, bins=20)

        # Note: We have counted every cluster pair twice, so now we'll halve
        # the frequency of distances in each bin
        spatial_distribution[0] = [freq/2 for freq in spatial_distribution[0]]

        bins_mean_distance = []
        for i in range(len(spatial_distribution[1])-1):
            bin_min = spatial_distribution[1][i]
            bin_max = spatial_distribution[1][i+1]
            bins_mean_distance.append(( bin_max+bin_min ) / 2)

        plt.loglog(bins_mean_distance, spatial_distribution[0], 'ko', markerfacecolor='grey')
        plt.ylabel('Rain clusters pair frequency')
        plt.xlabel('Distance between pair (km)')
        plot_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-clusters-pair-distribution-loglog' + '.png'
        plt.savefig(plot_name)
        plt.close()


    def rain_pair_distribution_intra_clusters(self):
        """Plots spatial distribution of rain pairs which belong to same rain cluster.
        A rain pair exists when two random points have precipitation."""
        print("\tRain pair distribution intra clusters")

        M, clustered_rain_grid = self.hoshen_kopelman()

        rain_pair_dists = []
        for key in M:
            cluster_rain_points = []
            for i in range(len(clustered_rain_grid)):
                for j in range(len(clustered_rain_grid[i])):
                    if clustered_rain_grid[i][j] == key:
                        latlon = index_to_coords(ij=[i,j], grid_bounds=self.grid_bounds)
                        cluster_rain_points.append(latlon)

            for latlon0 in cluster_rain_points:
                for latlon1 in cluster_rain_points:
                    if latlon0 != latlon1:
                        dist = distance.distance(latlon0, latlon1).km
                        rain_pair_dists.append(dist)

        spatial_distribution = np.histogram(rain_pair_dists, bins=20)

        # Note: We have counted every rain pair twice, so now we'll halve
        # the frequency of distances in each bin
        spatial_distribution[0] = [freq/2 for freq in spatial_distribution[0]]

        bins_mean_distance = []
        for i in range(len(spatial_distribution[1])-1):
            bin_min = spatial_distribution[1][i]
            bin_max = spatial_distribution[1][i+1]
            bins_mean_distance.append(( bin_max+bin_min ) / 2)

        plt.plot(bins_mean_distance, spatial_distribution[0], 'ko', markerfacecolor='grey')
        plt.ylabel('Rain pair frequency')
        plt.xlabel('Distance between pair (km)')
        plot_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-pair-distribution-intra-clusters' + '.png'
        plt.savefig(plot_name)
        plt.close()


# pass this function to class
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


class RainOverRiver:
    def __init__(self, filename, date):
        rain = pd.read_csv('data_pmm/tif/{}/{}'.format(date,filename)).set_index('Reach_ID',drop=True)
        self.rain = pd.concat([rain, padma], axis=1)
        self.file_name = re.search('3B-HHR.MS.MRG.3IMERG.(.*?).V06B.HDF5', filename).group(1)


    def rain_pair_over_river_path(self):
        """Plots rain pair distribution over river path.
        Given a rain pair that belongs to the same river path, computes the
        distance between pair over path trajectory."""
        print("\tRain pair distribution over river path")

        rain_pair_dists = []
        for reach_id0 in self.rain.index:
            if self.rain.at[reach_id0, 'rain'] != 0:
                for reach_id1 in self.rain.index:
                    if ( self.rain.at[reach_id1,'rain'] != 0 ) and ( reach_id0 != reach_id1 ):
                        # go down starting at reach_id0 until finding reach_id1
                        # or reaching end of river network
                        next_down = self.rain.at[reach_id0,'Next_down']
                        path_lenght = self.rain.at[reach_id0,'Length_km']
                        while ( next_down != reach_id1 ) and ( next_down != 0 ): # next_down=0 means we reached end of river network
                            path_lenght += self.rain.at[next_down,'Length_km']
                            next_down = self.rain.at[next_down,'Next_down']

                        if next_down != reach_id1:
                            # go down starting at reach_id1 until finding reach_id0
                            # or reaching end of river network
                            next_down = self.rain.at[reach_id1,'Next_down']
                            path_lenght = self.rain.at[reach_id1,'Length_km']
                            while ( next_down != reach_id0 ) and ( next_down != 0 ):
                                path_lenght += self.rain.at[next_down,'Length_km']
                                next_down = self.rain.at[next_down,'Next_down']

                        # if reach_id0 and reach_id1 belong to same path
                        # i.e. algorithm found other reach starting from the one
                        if ( next_down == reach_id0 ) or ( next_down == reach_id1 ):
                            rain_pair_dists.append(path_lenght)

        spatial_distribution = np.histogram(rain_pair_dists, bins=20)

        # Note: We have counted every rain pair twice, so now we'll halve
        # the frequency of distances in each bin
        spatial_distribution[0] = [freq/2 for freq in spatial_distribution[0]]

        bins_mean_distance = []
        for i in range(len(spatial_distribution[1])-1):
            bin_min = spatial_distribution[1][i]
            bin_max = spatial_distribution[1][i+1]
            bins_mean_distance.append(( bin_max+bin_min ) / 2)

        plt.plot(bins_mean_distance, spatial_distribution[0], 'ko', markerfacecolor='grey')
        plt.ylabel('Rain pair frequency')
        plt.xlabel('Travelled Distance between pair (km)')
        plot_name = 'data_results/rain/{}/'.format(self.date) + self.file_name + '-rain-pair-over-river-path' + '.png'
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
                rain_grid = RainGrid(filename=filename, date=date)
                rain_grid.rain_pair_distribution()
                rain_grid.rain_clusters_pair_distribution()
                rain_grid.rain_pair_distribution_intra_clusters()


        # for (dirpath, dirnames, filenames_) in os.walk('data_pmm/tif/{}'.format(date)):
        #     for filename in filenames_:
        #         if filename.endswith('-masked-resampled.csv'):
        #             file_specs = re.search('3B-HHR.MS.MRG.3IMERG.(.*?).V06B.HDF5-masked-resampled.csv', filename).group(1)
        #             print("Working on {}".format(file_specs))
        #             rain_over_river = RainOverRiver(filename=filename, date=date)
        #             rain_over_river.rain_pair_over_river_path()

            # class methods untested!
            # Proceed to tests
