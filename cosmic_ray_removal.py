'''
This program is hosted at:


Description:
This program will read Raman data from a .txt file, eliminate cosmic rays, and save the processed data in a .txt file
The technique used in this program is described in:
Uckert K., Bhartia B., et al. "A Semi-Autonomous Method to Detect Cosmic Rays in Raman Hyperspectral Datasets." Applied Spectroscopy Accepted March 22, 2019.

    A histogram is generated for all wavelength channels - intensity values are binned
    a cosmic ray is detected when a bin lies past the threshold value
    cosmic rays may be replaced by one of two functions:
        avg: the cosmic ray is replaced by an average of the surrounding values
        interp: the cosmic ray is replaced by a line of best fit to the surrounding points
            this method may fail in noisy regions of spectra
This program was tested using Python 3.6.1

License:
Copyright 2018, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws.
By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations.
User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Input:
An argument containing the path to data file
    --data_file 'path/to/data.txt'
    The program expects a data file with no header,
    with the wavelength channels listed in the first row (\t delimited),
    with the intensity values listed in subsequent rows (\t delimited)
An argument containing the path to store the desired output files
    --output_dir '/path/to/output'
An argument to define the replacement method ('A' for average, 'I' for interpolation
    --replace_method 'A'
An argument to set the threshold multiplier
    --threshold 10
An argument to set the wavelenegth range
    --wavelength_range [1000,2000]
        to only consider wavelength regions between 1000 and 2000 cm-1
    --wavelength_range 0
        to consider all wavelength regions

Example:
    python3 cosmic_ray_removal.py
        runs using the default data file value, default output path, and interpolated replacement values
    python3 cosmic_ray_removal.py --data_file '/path/to/data/file.txt' --output_dir '/path/to/output' --replace I --threshold 10 --wavelength_range 0

Output:
    a data file formatted in the same way as the input file, with cosmic rays removed
    a plot of the input spectra
    a plot of the output spectra

Author: Kyle Uckert
Email: kyle.uckert@jpl.nasa.gov
'''

import argparse
import os
import pandas as pd
import time
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip
from operator import itemgetter
from itertools import groupby
import matplotlib.pyplot as plt


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Read Raman hyperspectral map, remove cosmic rays, output cleaned hyperspectral dataset.')

    # default data path, if no argument is supplied
    default_data_file = os.path.join('input', 'BK7_chrome_target.txt')

    # default output path, if no argument is supplied
    default_output_path = os.path.join(os.getcwd(), 'output')

    # Arguments
    parser.add_argument('--data_file', metavar='D', type=str,
                        help='The full path to the data files',
                        default=default_data_file)

    parser.add_argument('--output_dir', metavar='O', type=str,
                        help='The full path to the directory to contain the output files',
                        default=default_output_path)

    parser.add_argument('--replace_method', metavar='R', type=str,
                        help='Technique to replace cosmic rays: I - interpolation, A - average',
                        default='A')

    parser.add_argument('--threshold', metavar='T', type=str,
                        help='Threshold value: integer',
                        default=10)

    parser.add_argument('--wavelength_range', metavar='W', type=str,
                        help='Range to apply algorithm to: 0 for all, [1000,2000] for wavelength channel range 1000 - 2000 cm-1',
                        default=0)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 0.1')

    # Parse arguments
    args = parser.parse_args()

    return args

# detects cosmic rays using the histogram method described in Uckert, 2019
# raw_spectra - pandas dataframe containing spectra intensity values and Raman shift values
# threshold - index for threshold multiplier
# method - 'A' or "I' representing replacement method
# spectra_index - list containing indicies of spectra to apply procedure to
# std_multiplier - True to multiply threshold index by standard deviation of histogram, False to use thresholdmultiplier as a constant
# CR_limit - limit of the number of comsic rays in any single spectrum
# sigma_val - the sigma clip factor
def cosmic_ray_execute(raw_spectra, threshold, method, wvl_ind, num_points, spectra_index, std_multiplier=True, CR_limit=5, sigma_val=4.0):
    start = time.time()
    len_of_file = wvl_ind[1] - wvl_ind[0] + 1
    # CR removal counter
    CRcounter = 0
    # copy spectra to replace cosmic ray removed intensity values with original values, if needed
    pre_CR_remove = raw_spectra.copy()

    # array to contain cosmic rays that have been removed (spectrum number, wavelength):
    # initialized with all values at -1 [number of spectra, max number of cosmic rays to be removed]
    CR_removed = np.full((num_points, CR_limit), -1)
    # [number of spectra, length of spectra] - assumes that all wavelength channels in a spectrum could be flagged
    CR_list = np.full((num_points, len_of_file), -1)
    sigma_factor = float(threshold)

    print('Finding cosmic rays in each wavelength channel...')
    # run for each wavelength channel
    # i is the index of the wavelength channel
    for i in range(0, len_of_file):
        exit_status = False
        while not exit_status:
            # calculate histogram, sigma-clipped mean, sigma-clipped std
            int_arr = raw_spectra.iloc[i + wvl_ind[0], range(len(spectra_index))]
            # CR_confirm contains a list of index values associated with spectra that have had cosmic rays flagged at this wavelength channel
            CR_confirm = []
            # repeat until sigma_clip array is no longer filtering elements
            delta_len = 99
            old_int = int_arr
            while delta_len > 1:
                new_int = sigmaclip(old_int, sigma_val, sigma_val)[0]
                delta_len = len(old_int) - len(new_int)
                old_int = new_int
            # if sigma-clipped array == original array, no cosmic rays will be found (as long as threshold is <= sigma_val
            # if no events are clipped, no events will be detected
            if len(new_int) < len(int_arr) or sigma_factor <= sigma_val:
                std = new_int.std()
                mean_hist = new_int.mean()
                hist, bins = np.histogram(int_arr, bins='fd')
                if not std_multiplier:
                    threshold_bin = np.argwhere(bins < (sigma_factor + mean_hist))[-1][0]
                else:
                    # index of bin associated with threshold
                    # don't consider closest bin (could fall to next bin higher) - find closest bin that is not higher
                    threshold_bin = np.argwhere(bins < ((sigma_factor * std) + mean_hist))[-1][0]
                CR_candidates = np.argwhere(hist[threshold_bin:] > 0)
                while len(CR_candidates) > 0:
                    # CR_index - index of intensity array associated with 1st occurrence of int values > outlier int value
                    # self.bins[gaps[-1]] = index of bin that has at least one intensity value associated with it
                    # this identifies the first occurence of an intensity array value larger than the bin+threshold value
                    CR_index = np.abs(int_arr - bins[CR_candidates[-1] + threshold_bin][0]).argmin()
                    CR_index = np.argwhere(CR_index == np.array(spectra_index))[0][0]

                    if hist[CR_candidates[-1] + threshold_bin] == 1:
                        CR_candidates = CR_candidates[0:-1]
                    else:
                        hist[CR_candidates[-1] + threshold_bin] -= 1
                        int_arr[CR_index] = -1
                    if CR_index not in CR_confirm:
                        CR_list[CR_index, np.argwhere(CR_list[CR_index, :] < 0)[0][0]] = i
            exit_status = True

    print('Calculating width of each event and removing cosmic rays...')
    # loop over all spectra
    for i in range(num_points):
        exit_status = False
        # CR_max_counter counts the number of cosmic rays removed for each spectrum
        CR_max_counter = 0
        while not exit_status:

            # get width of cosmic rays:
            if CR_list[i, 0] > -1:
                ranges = []
                for k, g in groupby(enumerate(CR_list[i, :]), lambda x: x[0] - x[1]):
                    group = list(map(itemgetter(1), g))
                    ranges.append((group[0], group[-1]))
                for j, CR in enumerate(ranges):
                    if CR[0] == -1:
                        ranges2 = ranges[0:j]
                        break
                for j in ranges2:
                    width = int(j[1] - j[0]) + 1
                    loc = j[0] + math.floor(width / 2)
                    # print('cosmic ray width: ' + str(width), 'at loc: ', str(self.raw_spectra['shift'][loc]))
                    if CR_max_counter < CR_limit and loc > -1:
                        if method=='A':
                            # remove cosmic rays and replace with mean of surrounding points
                            raw_spectra = CRreplace_avg(int(loc) + wvl_ind[0], i, width + 3, raw_spectra, spectra_index)
                        else:
                            # remove cosmic rays and replace with interpolated curve of nearby points
                            raw_spectra = CRreplace_interp(int(loc) + wvl_ind[0], i, width + 3, raw_spectra, spectra_index)
                        CRcounter += 1
                        CR_removed[i][CR_max_counter] = loc
                        CR_max_counter += 1
                        print('Removed cosmic ray in spectrum # ' + str(spectra_index[i] + 1) + ' at wavelength: ' + '{0:.1f}'.format(raw_spectra['shift'][loc + wvl_ind[0]]))
                    else:
                        exit_status = True
                exit_status = True
            else:
                exit_status = True

    # revert intensity array for spectra where the max number of cosmic rays have been removed:
    CR_count_subtract = 0
    for index, wvl_list in enumerate(CR_removed):
        if wvl_list[-1] != -1:
            event = 'Maximum number of cosmic rays detected in spectrum ' + str(spectra_index[index] + 1) + '. No cosmic rays removed.'
            raw_spectra[spectra_index[index]] = pre_CR_remove[spectra_index[index]]
            CR_count_subtract += 1
    stop = time.time()
    CR_time = stop - start
    # for proper grammar:
    if method == 'A':
        method_string='average'
    else:
        method_string = 'interpolation'
    if (CRcounter - (CR_count_subtract * CR_limit)) == 1:
        print(str(CRcounter - (CR_count_subtract * CR_limit)) + ' cosmic ray removed in ' + '{:0.2f}'.format(
            CR_time) + ' seconds and replaced with the ' + method_string + ' of nearby points\n')
    else:
        print(str(CRcounter - (CR_count_subtract * CR_limit)) + ' cosmic rays removed in ' + '{:0.2f}'.format(
            CR_time) + ' seconds and replaced with the ' + method_string + ' of nearby points\n')

    return raw_spectra

#remove cosmic rays by replacing adjacent cosmic ray hits with an average of surrounding values
def CRreplace_avg(loc, i, limit, raw_spectra, spectra_index):
    #limit=7
    avg = 0
    #if loc of cosmic ray is not near edge:
    #if loc < len(self.raw_spectra['shift']) - math.floor(limit / 2) - 1 and loc > math.ceil(limit / 2) - 1:
    if loc < len(raw_spectra['shift']) - math.floor(limit / 2) and loc > math.ceil(limit / 2):
        removal_list = list(range(math.ceil(-limit / 2), math.ceil(limit / 2)))
        if loc >= limit and loc <= len(raw_spectra['shift'])-limit-1:
            mean_list = list(range(-limit, math.ceil(-limit / 2))) + list(range(math.ceil(limit / 2), limit + 1))
        elif loc < limit:
            mean_list = list(range(-loc, math.ceil(-limit / 2))) + list(range(math.ceil(limit / 2), limit + 1))
        elif loc > len(raw_spectra['shift'])-limit-1:
            mean_list = list(range(-limit, math.ceil(-limit / 2))) + list(range(math.ceil(limit / 2), len(raw_spectra['shift'])-loc))
    #if loc is near beginning of spectrum:
    elif loc <= math.ceil(limit / 2):
        mean_list = list(range(math.ceil(limit/2), limit+1))
        removal_list = list(range(-loc, math.ceil(limit / 2)))
    #if loc is near end of spectrum:
    elif loc >= len(raw_spectra['shift']) - math.ceil(limit / 2):
        mean_list = list(range(-limit, math.ceil(-limit/2)))
        removal_list = list(range(math.ceil(-limit / 2), len(raw_spectra['shift'])-loc))
    #calculate mean of points around cosmic ray
    for f in mean_list:
        avg += raw_spectra[spectra_index[i]].ix[loc+f] / len(mean_list)
    #replace cosmic ray and surrounding points with mean
    for q in removal_list:
        g = int(loc+q)
        raw_spectra[spectra_index[i]].loc[g] = avg
    return raw_spectra

#replace cosmic ray hits by interpolating the remaining points
def CRreplace_interp(loc, i, limit, raw_spectra, spectra_index):
    #limit=7
    #if loc of cosmic ray is not near edge:
    if loc < len(raw_spectra['shift']) - math.floor(limit/2)-1 and loc > math.ceil(limit/2):
        shift_new = list(range(math.ceil(-limit / 2), math.ceil(limit / 2)))
        if loc >= limit and loc <= len(raw_spectra['shift'])-limit-1:
            interp_shift = list(range(-limit, math.ceil(-limit / 2))) + list(range(math.ceil(limit / 2), limit + 1))
        elif loc < limit:
            interp_shift = list(range(-loc, math.ceil(-limit / 2))) + list(range(math.ceil(limit / 2), limit + 1))
        elif loc > len(raw_spectra['shift'])-limit-1:
            interp_shift = list(range(-limit, math.ceil(-limit / 2))) + list(range(math.ceil(limit / 2), len(raw_spectra['shift'])-loc))
    #if loc is near beginning of spectrum:
    elif loc <= math.ceil(limit / 2):
        interp_shift = [-loc]+list(range(math.ceil(limit/2), limit+1))
        shift_new = list(range(-loc, math.ceil(limit / 2)))
    #if loc is near end of spectrum:
    elif loc >= len(raw_spectra['shift']) - math.ceil(limit / 2):
        interp_shift = list(range(-limit, math.ceil(-limit/2)))+[len(raw_spectra['shift'])-1-loc]
        shift_new = list(range(math.ceil(-limit / 2), len(raw_spectra['shift'])-loc))
    #calculate the new intensity values through the interpolation procedure
    #interp_shift = [item for item in interp_shift if item >= 0]
    loc = np.int64(loc)
    interp_int = interp1d(raw_spectra['shift'].ix[interp_shift+loc], raw_spectra[spectra_index[i]].ix[interp_shift+loc], kind='cubic')
    int_new = interp_int(raw_spectra['shift'].ix[shift_new+loc])
    raw_spectra[spectra_index[i]].ix[shift_new+loc] = int_new
    return raw_spectra



# get wavelength range from user input - convert to indicies
def get_wvl(shift, wvl):
    # check if user selected "0" representing all values are to be considered
    if wvl == 0:
        return [0, len(shift)-1]
    else:
        # if user supplied a wavelength below minimum, return 0
        if wvl[0] < shift.ix[0,0]:
            ind1 = 0
        # return index associated with selected wavelength
        else:
            ind1 = (np.abs(shift.as_matrix() - wvl[0])).argmin()
        # if user supplied a wavelength above the maximum, return last index
        if wvl[1] > shift.ix[len(shift)-1, 0]:
            ind2 = len(shift)-1
        else:
            ind2 = (np.abs(shift.as_matrix() - wvl[1])).argmin()
        return [ind1, ind2]

# plot every nth point (to reduce image file size and speed up plotting function)
# n=1 plots all spectra
def plot_nth_points(spectra_df, save_file, xlabel='', ylabel='', title='', n=10):
    shift = spectra_df['shift']
    intensity = spectra_df[spectra_index[::n]]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.18)
    fig.patch.set_facecolor('#E0E0E0')
    for y in intensity:
        plot = ax1.plot(shift, intensity[y])
    ax1.grid(b=True, which='major', linestyle='--')
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.set_xlabel(xlabel, fontsize=10)
    plt.title(title)
    fig.savefig(save_file, transparent=False, dpi=300)
    ax1.cla()
    plt.clf()
    plt.close()



if __name__ == '__main__':
    input_args = parseArguments()

    input_file = input_args.data_file
    output_dir = input_args.output_dir
    # if the output directory doesn't exist, create it
    print('Saving files to: ' + output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, os.path.split(input_file)[1][:-4]+'_output.txt')
    method = input_args.replace_method
    threshold = input_args.threshold
    wvl_range = input_args.wavelength_range
    # True: multiply threshold by STD
    std_multiplier = True
    # CR_limit defines number of successive cosmic ray removals for each spectrum
    CR_limit = 5
    # sigma clip factor
    sigma_val=4.0
    # plot every nth spectrum
    n_spectra_plot=10

    # read the file into a pandas dataframe
    print('Reading files from: ' + input_file)
    raw_file = pd.read_csv(input_file, sep='\t', low_memory=False, index_col=0, header=None)
    raw_file = raw_file.reset_index(drop=True)

    # contains the Raman shift values
    shift = raw_file.iloc[[0]]
    shift = shift.transpose()
    shift = shift.reset_index(drop=True)
    shift = shift.rename(columns={0: 'shift'})

    # drop 0 index (contains shift values)
    raw_file = raw_file.drop(raw_file.index[[0]])
    raw_file = raw_file.transpose()
    raw_file = raw_file.reset_index(drop=True)

    # the number of spectra collected
    num_points = len(raw_file.columns)
    # indicies of the spectra to process
    # default is to process all spectra
    spectra_index = list(range(num_points))

    # a list of the column identifiers [0,1,2,...n]
    cols = np.arange(0, len(raw_file.columns))
    raw_file.columns = cols

    # contains the intensity and shift data
    frames = [raw_file, shift]

    raw_spectra = pd.concat(frames, axis=1)
    # under some circumstances, a NaN line will be added to the shift array, if so, remove it
    raw_spectra = raw_spectra.dropna()

    # plot all initial spectra:
    print('Plotting input spectra, every '+str(n_spectra_plot) +'th spectrum, stacked.')
    save_file_init = os.path.join(output_dir, 'initial_spectra.png')
    # remove this file, if it already exists
    if os.path.exists(save_file_init):
        os.remove(save_file_init)
    title = 'Initial Spectra - Every '+str(n_spectra_plot) +'th Spectrum, Stacked'
    plot_nth_points(raw_spectra, save_file_init, xlabel='wavelength shift (cm$^{-1}$)', ylabel='Intensity (counts)', title=title, n=n_spectra_plot)
    print('Saved initial dataset plot to: ' + save_file_init)

    # perform cosmic ray detection and replacement
    print('Identifying cosmic rays.')
    if method == 'I':
        print('Replacing cosmic rays with interpolated value of nearest neighbors.')
    else:
        print('Replacing cosmic rays with average value of nearest neighbors.')
    wvl_ind = get_wvl(shift, wvl_range)

    processed_spectra = cosmic_ray_execute(raw_spectra, threshold, method, wvl_ind, num_points, spectra_index, std_multiplier=True, CR_limit=10, sigma_val=4.0)

    # save spectra to a new file
    # remove this file, if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)
    output_data=processed_spectra.transpose()
    output_data = output_data.reindex(['shift'] + spectra_index)
    output_data.index.name=None
    output_data.to_csv(output_file, sep='\t', header=False, index=False)
    print('Saved output dataset to: ' + output_file)

    # plot cosmic ray removed spectra
    print('Plotting cosmic ray-removed spectra, every '+str(n_spectra_plot) +'th spectrum, stacked.')
    save_file_out = os.path.join(output_dir, 'cosmic_ray_removed_spectra.png')
    # remove this file, if it already exists
    if os.path.exists(save_file_out):
        os.remove(save_file_out)
    title = 'Cosmic Ray Removed Spectra - Every '+str(n_spectra_plot) +'th Spectrum, Stacked'
    plot_nth_points(processed_spectra, save_file_out, xlabel='wavelength shift (cm$^{-1}$)', ylabel='Intensity (counts)', title=title, n=n_spectra_plot)
    print('Saved output dataset plot to: ' + save_file_out)
