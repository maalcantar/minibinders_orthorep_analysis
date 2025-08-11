#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 22 11:30:29 2025
@author: alcantar
example run: python 06_compute_Kds.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/normalized_tite-seq_scores/ -m mb171
"""
# activate virtual enviroment before running script
# source activate minibinders

import sys

import pandas as pd
import scipy as sp
import glob as glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math

from scipy.optimize import curve_fit
import scipy.stats as stats

from utils import *

import argparse

def plot_score_v_conc(scores,
                      concs,
                      fit_concs,
                      fit_scores,
                      out_path,
                      save=False):
    """
    Creates plots of tite-seq scores vs. antigen concentrations (on log10 scale).

    PARAMETERS
    -----------
    scores: list of continuous values
        Tite-seq scores (2D list: replicates x concentrations)
    concs: list of continuous values
        Concentrations used in tite-seq experiments
    fit_concs: list of continuous values
        Fitted antigen concentrations
    fit_scores: list of continuous values
        Fitted tite-seq scores
    outpath: str
        Name of output file, including path
    save: boolean
        Indicate whether to save

    RETURNS
    --------
    NONE
    """

    # Convert inputs to numpy arrays
    scores = np.array(scores)
    concs = np.array(concs)
    fit_concs = np.array(fit_concs)
    fit_scores = np.array(fit_scores)

    # Grab the values at no antigen concentration
    linear_concentrations = np.array([0])
    linear_scores_rep1 = np.array([scores[0, 0]])
    linear_scores_rep2 = np.array([scores[1, 0]])

    # Grab the values at antigen concentrations above 0nM
    log_concentrations = concs[1:]
    log_scores_rep1 = scores[0, 1:]
    log_scores_rep2 = scores[1, 1:]

    # Fit values
    log_concentrations_fit = fit_concs[1:]
    log_scores_fit_rep1 = fit_scores[0, 1:]
    log_scores_fit_rep2 = fit_scores[1, 1:]

    # Start plotting
    fig, ax = plt.subplots()

    # ====== REP 1 START ======
    ax.plot(linear_concentrations, linear_scores_rep1, marker='o',
            markersize=10, markeredgecolor='black', color='#D1A95F')
    ax.plot(log_concentrations, log_scores_rep1, linestyle='none',
            marker='o', markersize=10, markeredgecolor='black', color='#D1A95F', label='Replicate 1')
    # Plot the fit -- use solid lines with no markers
    ax.plot(log_concentrations_fit, log_scores_fit_rep1, linestyle='-',
            linewidth=3, color='#D1A95F')
    # ====== REP 1 END ======

    # ====== REP 2 START ======
    ax.plot(linear_concentrations, linear_scores_rep2,
            marker='o', markersize=10, markeredgecolor='black', color='#91B7DE')
    ax.plot(log_concentrations, log_scores_rep2, linestyle='none',
            marker='o', markersize=10, markeredgecolor='black', color='#91B7DE', label='Replicate 2')
    ax.plot(log_concentrations_fit, log_scores_fit_rep2, linestyle='-',
            linewidth=3, color='#91B7DE')
    # ====== REP 2 END ======

    # Customize x-axis to handle dual scaling
    ax.set_xscale('symlog', linthreshx=1e-3)  # Symlog scales allow a transition from linear to log
    ax.set_xticks([0, 0.01, 0.1, 1, 10, 100, 500])
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%.2f'))  # Format tick labels
    y_min = 1.5
    y_max = 4.05
    ax.set_ylim([y_min, y_max])

    # Add labels and legend
    ax.set_xlabel('Concentration (nM)')
    ax.set_ylabel('log10(Fluorescence)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper left')
    if save:
        plt.savefig(out_path + '.png', dpi=400, transparent=True)
        plt.savefig(out_path + '.pdf', dpi=400, transparent=True)
    plt.close()

def add_binary_list(bin_list,
                    parent_code):

    """
    Adds every binary string in a list. This is useful for assigning unique codes to
    each variant in a combinatorial library. For example
    wild type = 00
    mutation A = 10
    mutation B = 01
    mutation AB = 11

    PARAMETERS
    -----------
    bin_list: list of str
        list containing all binary strings that should be summed together
    parent_code: str
        base code for the parent -- usually [0]*number of possible mutations

    RETURNS
    --------
    sum of all binary strings or the parent code if the input list was empty
    """

    if bin_list:
        bin_list = [parent_code] + bin_list
        # Sum up the integer values of each binary string in the list
        total = sum(int(b, 2) for b in bin_list)
        # Convert the sum back to binary and format as needed
        return bin(total)[2:].zfill(len(max(bin_list, key=len)))
    else:
        return(parent_code)

def func_E50(concs, Kd, As, B):

    """
    Defines a general hill function that can be used to obtain an EC50 (Kd estimate) value

    PARAMETERS
    -----------
    concs: list of floats
        list of concentrations (independent variables) to be used in the fit
    Kd: float
        Kd (EC50) value
    As: float
        fluorescence level at saturation
    B: float
        Background fluorescence (fluorescence with no antigen present)

    RETURNS
    -----------
    fluorescence: floats
        log10 fluorescence
    """

    fluorescence = ((concs) / (concs + Kd)) * As + B

    # Return log10 of fluorescence
    return np.log10(fluorescence)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Path to directory with scores.')
    parser.add_argument('-m', help='minibinder name')

    args = parser.parse_args()

    path_to_norm_scores = args.i #'../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/normalized_tite-seq_scores/'
    mb = args.m #'mb171'

    if get_yes_no_input(mb):
            print("Great! Continuing with analysis ...")
    else:
        print("Please use the correct mb name.")
        sys.exit(1)

    # create output directories and paths for both replicates
    if path_to_norm_scores[-1] != '/':
        path_to_norm_scores = path_to_norm_scores + '/'

    output_dir = path_to_norm_scores.replace('normalized_tite-seq_scores', 'tite-seq_Kds')
    output_dir_plots = output_dir + 'score_vs_concs_plots/'
    make_dir(output_dir)
    make_dir(output_dir_plots)

    tite_seq_norm_scores_path_list = sorted(glob.glob(f"{path_to_norm_scores}*.csv"))
    outname_rep1 = tite_seq_norm_scores_path_list[0].split('/')[-1].replace('norm_scores', 'Kds').replace(".csv","")
    outname_plots = outname_rep1.split('_')[0]+'_'
    outname_rep2 = tite_seq_norm_scores_path_list[1].split('/')[-1].replace('norm_scores', 'Kds').replace(".csv","")

    outpath_rep1 = f"{output_dir}{outname_rep1}"
    outpath_rep2 = f"{output_dir}{outname_rep2}"
    outpath_plots = f"{output_dir_plots}{outname_plots}"

    if 'rep1' in tite_seq_norm_scores_path_list[0] and 'rep2' in tite_seq_norm_scores_path_list[1]:
        pass
    else:
        print('Error! Replicates are not being read in the proper order.')
        sys.exit(0)

    # read csv with raw counts as csv
    scores_rep1_df = pd.read_csv(tite_seq_norm_scores_path_list[0], index_col=None)
    scores_rep2_df = pd.read_csv(tite_seq_norm_scores_path_list[1], index_col=None)
    # set up output dirs


    if mb == 'mb171':
        mut_code_dict = {'N7D':'100000',
                         'F10Y':'010000',
                         'A14V':'001000',
                         'P29S':'000100',
                         'K45E':'000010',
                         'K52E': '000001'}
        parent_code='000000'
    else:
        print('minibinder {mb} is not acceptable')
        sys.exit(1)


    # loop through both dataframes simultaneously
    # (idxRow, s1), (_, s2) in zip(df0.iterrows(), df1.iterrows())
    concs_list = [0.0, 0.01, 0.1, 1, 10, 100, 500]
    # Kds_rep1_list = []
    # Kds_rep2_list = []
    Kd_rep1_list = []
    Kd_rep2_list = []
    As_rep1_list = []
    As_rep2_list = []
    Kd_avg_list = []
    As_mean_list = []
    variant_id_list = []

    # variables for fit
    # bounds = ([10e-5, 0, 0], [10e5, 1.5, 1.5]) -- this is if using simple normalization method
    # bounds = ([10e-5, 0, 0], [10e5, 10, 10]) -- this is if fitting without applying log10 to sigmoid function
    bounds = ([10e-5, 0, 0], [10e5, 10e6, 10e5])
    conc_fits = np.logspace(-4, 2.69897000434, num=1000)

    outname_EC50 = outname_rep1.split('_')[0] + '_EC50_df.csv'
    outpath_EC50 = f"{output_dir}{outname_EC50}"
    for (col_rep1,row_rep1), (col_rep2, row_rep2) in zip(scores_rep1_df.iterrows(), scores_rep2_df.iterrows()):
        assert row_rep1[0] == row_rep2[0], "Replicate 1 and 2 dataframes do not appear to be in the same order"

        if row_rep1['aa_mutations'] =='[]':
            aa_mutations_rep1 = []
            list_of_mutations_rep1 = []
        else:
            aa_mutations_rep1 = extract_mutations_from_string(row_rep1['aa_mutations'])
            list_of_mutations_rep1 = [mut_code_dict[key.replace(" ", "")] for key in aa_mutations_rep1]

        variant_id_rep1 = add_binary_list(list_of_mutations_rep1,
                        parent_code)
        variant_id_list.append(str(variant_id_rep1))

        print(f"Processing variant {str(variant_id_rep1)}")

        scores_rep1 = row_rep1[-7:]
        scores_rep2 = row_rep2[-7:]
        scores_list = [scores_rep1, scores_rep2]
        outpath_plots_tmp = f"{outpath_plots}{variant_id_rep1}"

        popt_rep1, pcov_rep1 = sp.optimize.curve_fit(func_E50, np.array(concs_list),
                                           np.array(scores_rep1), bounds=bounds)
        Kd_temp_rep1, As_temp_rep1, B_temp_rep1 = popt_rep1
        scores_fit_rep1 = (((conc_fits)/(conc_fits + Kd_temp_rep1))*As_temp_rep1 + B_temp_rep1)
        scores_fit_rep1 = np.log10(scores_fit_rep1)

        popt_rep2, pcov_rep2 = sp.optimize.curve_fit(func_E50, np.array(concs_list),
                                           np.array(scores_rep2), bounds=bounds)
        Kd_temp_rep2, As_temp_rep2, B_temp_rep2 = popt_rep2
        scores_fit_rep2 = (((conc_fits)/(conc_fits + Kd_temp_rep2))*As_temp_rep2 + B_temp_rep2)
        scores_fit_rep2 = np.log10(scores_fit_rep2)

        max_signal_rep1_tmp = As_temp_rep1 + B_temp_rep1
        max_signal_rep2_tmp = As_temp_rep2 + B_temp_rep2
        Kd_rep1_list.append(Kd_temp_rep1)
        As_rep1_list.append(max_signal_rep1_tmp) # max signal
        Kd_rep2_list.append(Kd_temp_rep2)
        As_rep2_list.append(max_signal_rep2_tmp)

        Kd_avg_list.append(np.mean([Kd_temp_rep1,Kd_temp_rep2]))
        As_mean_list.append(np.mean([max_signal_rep1_tmp, max_signal_rep2_tmp]))


        plot_score_v_conc(scores = scores_list,
                         concs=concs_list,
                         fit_concs = conc_fits,
                         fit_scores = [scores_fit_rep1, scores_fit_rep2],
                         out_path=outpath_plots_tmp,
                         save=True)

    output_plot_correlations_path = f"{outpath_plots}EC50_correlations"

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(Kd_rep1_list,Kd_rep2_list,
               marker='.', s=200, color='black', alpha=0.8)
    ax.set_xlabel('Replicate 1 EC50s (nM)', size=16)
    ax.set_ylabel('Replicate 2 EC50s (nM)', size=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)# remove top and right borders
    correlation_coefficient, p_value = stats.pearsonr(Kd_rep1_list,Kd_rep2_list)
    correlation_coefficient_spear, p_value_spear = stats.spearmanr(Kd_rep1_list,Kd_rep2_list)
    plt.text(.6, 50, f'PearsonR = {correlation_coefficient:.5f}, pval={p_value:.2e}', fontsize=12)
    plt.text(.6, 30, f'SpearmanR = {correlation_coefficient_spear:.5f}, pval={p_value_spear:.2e}', fontsize=12)

    plt.savefig(output_plot_correlations_path+'.png', dpi=400)
    plt.savefig(output_plot_correlations_path+'.pdf', dpi=400)
    plt.close()

    Kd_dataframe = scores_rep1_df.copy()[['dna_sequence',
                                           'aa_sequence',
                                           'dna_mutations',
                                           'aa_mutations',
                                           'number_dna_mutations',
                                           'number_aa_mutations']]

    Kd_dataframe['variant_id'] = variant_id_list
    Kd_dataframe['EC50_rep1'] = Kd_rep1_list
    Kd_dataframe['EC50_rep2'] = Kd_rep2_list
    Kd_dataframe['EC50_mean'] = Kd_avg_list
    Kd_dataframe['max_signal_rep1'] = As_rep1_list
    Kd_dataframe['max_signal_rep2'] = As_rep2_list
    Kd_dataframe['max_signal_mean'] = As_mean_list
    Kd_dataframe.to_csv(outpath_EC50)

if __name__ == "__main__":
    main()
