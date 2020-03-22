import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import csv


def build_avalanches_data():
    overflow = pd.read_pickle('data_results/overflow_130616_130617_2.pkl')
    padma = pd.read_pickle('data_gloric/padma_gloric_1m3_final_no_geo.pkl')
    padma = padma.set_index('Reach_ID',drop=False)
    padma.head()

    all_reaches_ids = overflow.index

    if not os.path.exists('data_results/avalanches'):
        os.makedirs('data_results/avalanches')

    # timesteps from 0 to 96
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

        avalanches_filename = 'data_results/avalanches/timestep-{}.csv'.format(timestep)
        with open(avalanches_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerows(avalanches)


def plot_avalanche_size_hist():
    """Plots Avalanche-size frequency vs Avalanche-size"""
    avalanche_size_frequency = [0 for i in range(200)]
    for timestep in range(97):
        filename = 'data_results/avalanches/timestep-{}.csv'.format(timestep)
        with open(filename, 'r') as infile:
            readlines = csv.reader(infile)
            avalanches = [row for row in readlines if row!=[]]
        for avalanche in avalanches:
            avalanche_size_frequency[len(avalanche)] += 1

    avalanche_size_array = np.asarray([i for i in range(len(avalanche_size_frequency)) if 152>=i>1]).reshape(-1, 1)
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



if __name__ == '__main__':
    # build_avalanches_data()
    plot_avalanche_size_hist()
