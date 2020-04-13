import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import csv


overflow = pd.read_pickle('data_results/overflow_130616_130617_2.pkl')
padma = pd.read_pickle('data_gloric/padma_gloric_1m3_final_no_geo.pkl')
padma = padma.set_index('Reach_ID',drop=False)


def build_avalanches_data():

    all_reaches_ids = overflow.index

    if not os.path.exists('data_results/avalanches'):
        os.makedirs('data_results/avalanches')

    # timesteps from 0 to 96
    # 30 minute timesteps
    for timestep in range(97):
        print('Counting avalanches for timestep {}...'.format(timestep))
        avalanches = []
        counted_reaches_ids = []
        for reach_id in all_reaches_ids:
            next_down_reach_id = reach_id
            avalanche = []
            while ( overflow.loc[next_down_reach_id, timestep] != 0 ) and ( next_down_reach_id not in counted_reaches_ids ):
                counted_reaches_ids.append(next_down_reach_id)
                avalanche.append(next_down_reach_id)
                next_down_reach_id = padma.loc[reach_id, 'Next_down']
            avalanche_is_registered = False
            for registered_avalanche in avalanches:
                if next_down_reach_id in registered_avalanche:
                    avalanche_is_registered = True
                    registered_avalanche_idx = avalanches.index(registered_avalanche)
                    for reach_id in avalanche:
                        avalanches[registered_avalanche_idx].append(reach_id)
            if not avalanche_is_registered:
                avalanches.append(avalanche)
                counted_reaches_ids.append(next_down_reach_id)

        all_counted_overflows = []
        for avalanche in avalanches:
            for reach in avalanche:
                all_counted_overflows.append(reach)
        all_overflows = []
        for reach_id in all_reaches_ids:
            if overflow.loc[reach_id, timestep] != 0:
                all_overflows.append(reach_id)
        if len(all_counted_overflows) != len(all_overflows):
            print('AvalancheCountError: Miscalculation in timestep {}!\n Avalanche counter found {} overflows, but there are actually {}.\n Please check algorithm for mistakes.'\
            .format(timestep, len(all_counted_overflows), len(all_overflows)))

        avalanches = filter(None, avalanches)

        avalanches_filename = 'data_results/avalanches/histogram/timestep-{}.csv'.format(timestep)
        with open(avalanches_filename, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerows(avalanches)


def plot_avalanche_size_hist(logarithmic_binning=True):
    """Plots Avalanche-size frequency vs Avalanche-size"""
    avalanche_size_frequency = [0 for i in range(200)]
    for timestep in range(97):
        filename = 'data_results/avalanches/histogram/timestep-{}.csv'.format(timestep)
        with open(filename, 'r') as infile:
            readlines = csv.reader(infile)
            avalanches = [row for row in readlines if row!=[]]
        for avalanche in avalanches:
            avalanche_size_frequency[len(avalanche)] += 1

    if logarithmic_binning:
        bins = np.logspace(0, 2.2, 10)
        hist = np.histogram([i for i in range(len(avalanche_size_frequency)) if i>1], bins=bins)
        log_bin_avalanche_size = []
        log_bin_avalanche_size_freq = [0 for i in range(9)]
        for i in range(len(hist[1])-1):
            log_bin_avalanche_size.append(( hist[1][i] + hist[1][i+1] ) / 2 )
            for j in range(len(avalanche_size_frequency)):
                if hist[1][i]<=j<hist[1][i+1]:
                    log_bin_avalanche_size_freq[i] += avalanche_size_frequency[j]
        recta = np.asarray(log_bin_avalanche_size)**(-1.7)*(10**5.65)
        plt.loglog(log_bin_avalanche_size,
                 log_bin_avalanche_size_freq, 'ko', markerfacecolor='grey')
        plt.loglog(log_bin_avalanche_size,
                 recta, '-r', label=r'$\alpha=1.7$')
        plt.ylabel('Avalanche-size frequency')
        plt.xlabel('Avalanche-size')
        plt.legend(fontsize=10)
        plt.show()

    else:
        avalanche_size_array = np.asarray([i for i in range(len(avalanche_size_frequency)) if 152>=i>1])\
                                        .reshape(-1, 1)
        avalanche_size_freq_array = np.asarray([freq for freq in avalanche_size_frequency[2:153]])
        # replaced all 0's with 1's
        avalanche_size_freq_array[avalanche_size_freq_array == 0] = 1
        log_avalanche_size_array = np.log10(avalanche_size_array)
        log_avalanche_size_freq_array = np.nan_to_num(np.log10(avalanche_size_freq_array))

        reg = LinearRegression().fit(log_avalanche_size_array, log_avalanche_size_freq_array)
        # print(reg.score(log_avalanche_size_array, log_avalanche_size_freq_array))
        # print(reg.coef_)
        # print(reg.intercept_)

        s_fit = np.linspace(1,152)
        H_fit = (10**reg.intercept_)*(s_fit**reg.coef_[0])

        plt.loglog(avalanche_size_array,
                 avalanche_size_freq_array, 'ko', markerfacecolor='grey')
        plt.loglog(s_fit, H_fit, 'k', label=r'$H(s)\sim s^{-\alpha}$')
        plt.text(5,80, r'$\alpha={:.2f}$'.format(abs(reg.coef_[0])))
        plt.ylabel('Avalanche-size frequency')
        plt.xlabel('Avalanche-size')
        plt.legend(fontsize=10)
        plt.show()


def avalanche_spatial_distribution_over_river():
    """Plots avalanche spatial distribution over river"""
    print("Avalanche Spatial Distribution Over River Path")
    avalanche_pairs_distances = []
    for timestep in range(97):
        print("\tWorking on timestep", timestep)
        for reach_id in padma.index:
            # check if reach has overflow
            if overflow.loc[reach_id, timestep] != 0:
                next_down_reach_id = padma.loc[reach_id, 'Next_down']
                if overflow.loc[next_down_reach_id, timestep] != 0:
                    while overflow.loc[next_down_reach_id, timestep] != 0:
                        prior_reach_id = next_down_reach_id
                        next_down_reach_id = padma.loc[next_down_reach_id, 'Next_down']
                    # reach from which we will measure the distance to first reach
                    # of all following avalanches (only downwards)
                    reference_reach_id = prior_reach_id
                    travelled_dist = padma.loc[prior_reach_id, 'Length_km']

                    # until river network reaches end
                    while next_down_reach_id != 0: # next_down_reach_id = 0 iff there is no next down reach
                        travelled_dist += padma.loc[next_down_reach_id, 'Length_km']
                        if padma.loc[next_down_reach_id, 'Next_down'] == 0:
                            break
                        is_avalanche = ( overflow.loc[next_down_reach_id, timestep] ) == 0 and ( overflow.loc[padma.loc[next_down_reach_id, 'Next_down'], timestep] )
                        if is_avalanche and overflow.loc[next_up_reach_id, timestep] == 0: # is avalanche and is not registered yet
                            avalanche_pairs_distances.append(travelled_dist)
                        next_up_reach_id = next_down_reach_id
                        next_down_reach_id = padma.loc[next_down_reach_id, 'Next_down']

        spatial_distribution = np.histogram(avalanche_pairs_distances, bins=20)

        bins_mean_distance = []
        for i in range(len(spatial_distribution[1])-1):
            bin_min = spatial_distribution[1][i]
            bin_max = spatial_distribution[1][i+1]
            bins_mean_distance.append(( bin_max+bin_min ) / 2)

        plt.plot(bins_mean_distance, spatial_distribution[0], 'ko', markerfacecolor='grey')
        plt.ylabel('Avalanche pair frequency')
        plt.xlabel('Travelled Distance between pair (km)')
        plot_name = 'data_results/avalanches/avalanche-pair-spatial-distribution-over-river.png'
        plt.savefig(plot_name)
        plt.close()


def plot_avalanche_under_rain_cluster_distribution():
    """Given a rain cluster, computes the distance of cluster center of mass to
    last (most down) reach that is part of an avalanche that starts in a reach
    under that rain cluster. Plots histogram of this distances."""
    import rain

    avalanches = pd.read_csv('data_results/avalanches/histogram/timestep-{}.csv'.format(2))
    rain_grid = rain.RainGrid(filename='data_pmm/raw/2013-06-16/3B-HHR.MS.MRG.3IMERG.'
        '20130616-S010000-E012959.0060.V06B.HDF5',
                              date='2013-06-16')
    M, clustered_rain_grid = rain_grid.hoshen_kopelman()

    # for all rain clusters find avalanches that start under cluster
    # compute distance of cluster most precipitous point to last avalanche reach
    for key in M:
        pass

    # plot distribution of this distances



if __name__ == '__main__':
    # build_avalanches_data()
    # plot_avalanche_size_hist()
    # avalanche_spatial_distribution_over_river()
    plot_avalanche_under_rain_cluster_distribution()
