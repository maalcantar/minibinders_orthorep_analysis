#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 15 12:16:54 2025
@author: alcantar
example run: python 05_create_normalized_tite_seq_scores.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/normalized_tite-seq_counts/ -m mb171 -n fmean

"""
# activate virtual enviroment before running script
# source activate minibinders

import sys
import glob as glob
import pandas as pd
import numpy as np

import argparse

from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Directory containing normalized counts')
    parser.add_argument('-m', help='minibinder name')
    parser.add_argument('-n', help='\'simple\' or \'fmean\'')

    args = parser.parse_args()

    path_input = args.i
    mb = args.m
    norm_method = args.n

    if get_yes_no_input(mb):
        print("Great! Continuing with analysis ...")
    else:
        print("Please use the correct mb name.")
        sys.exit(1)

    # create output directories and paths for both replicates
    if path_input[-1] != '/':
        path_input = path_input + '/'

    output_dir = path_input.replace('normalized_tite-seq_counts', 'normalized_tite-seq_scores')
    make_dir(output_dir)

    tite_seq_norm_counts_path_list = sorted(glob.glob(f"{path_input}*.csv"))
    outname_rep1 = tite_seq_norm_counts_path_list[0].split('/')[-1].replace('normalized', f'{norm_method}-norm_scores')
    outname_rep2 = tite_seq_norm_counts_path_list[1].split('/')[-1].replace('normalized', f'{norm_method}-norm_scores')

    outpath_rep1 = f"{output_dir}{outname_rep1}"
    outpath_rep2 = f"{output_dir}{outname_rep2}"

    if 'rep1' in tite_seq_norm_counts_path_list[0] and 'rep2' in tite_seq_norm_counts_path_list[1]:
        pass
    else:
        print('Error! Replicates are not being read in the proper order.')
        sys.exit(0)

    # read csv with raw counts as csv
    normed_rep1_df = pd.read_csv(tite_seq_norm_counts_path_list[0], index_col=None)
    normed_rep2_df = pd.read_csv(tite_seq_norm_counts_path_list[1], index_col=None)

    # normalization values
    conc_list = ['0nM', '0pt01nM', '0pt1nM', '1nM', '10nM', '100nM', '500nM']
    if norm_method == 'simple':
        # this method basically normalizes binding to be between 0 and 1
        norm_list = [0,(1/3), (2/3), (3/3)]

        geom_means_rep1 = {key: [] for key in conc_list}
        geom_means_rep2 = geom_means_rep1.copy()

        geom_means_rep1 = {conc: norm_list.copy() for conc in geom_means_rep1}
        geom_means_rep2 = {conc: norm_list.copy() for conc in geom_means_rep2}

    elif norm_method == 'fmean':
        # this method uses the geometric mean of each bin to estimate
        # fluorescence values for each variant at each concentration

        if mb == 'mb171':
            # in an eariler version, these were names 'cell_counts_raw_repX'
            # that variable name was inaccurate, as these are actually the
            # geometric means of fluorescence in each bin
            geom_means_rep1 = {'0nM': np.log10([35, 174, 0.01, 0.01]),
                                    '0pt01nM': np.log10([43, 266, 1064, 0.01]),
                                    '0pt1nM': np.log10([54, 388, 2178, 6944]),
                                   '1nM': np.log10([61, 448, 2631, 8713]),
                                   '10nM': np.log10([70, 474, 2970, 9326]),
                                   '100nM': np.log10([80, 533, 3095, 9755]),
                                   '500nM': np.log10([81, 550, 3230, 10080])}

            geom_means_rep2 = {'0nM': np.log10([35, 176, 0.01, 0.01]),
                                '0pt01nM': np.log10([46, 359, 1309, 0.01]),
                                '0pt1nM': np.log10([56, 406, 2392, 7582]),
                                '1nM': np.log10([62, 450, 2709, 9063]),
                                '10nM': np.log10([70, 474, 2956, 9646]),
                                '100nM': np.log10([80, 531, 3084, 10044]),
                                '500nM': np.log10([85, 562, 3325, 10690])}

        else:
            print("Error! normalization method should be in [simple, fmean].")
            sys.exit(1)

    for conc in geom_means_rep1:
        normed_rep1_df_tmp = normed_rep1_df.copy().filter(regex=rf"^{conc}_bin.*_normed_counts$", axis=1)
        bin_names_list = list(normed_rep1_df_tmp.columns)
        norm_list_tmp = geom_means_rep1[conc]
        normed_rep1_df[f'score_{conc}'] = (normed_rep1_df_tmp[bin_names_list[0]]*norm_list_tmp[0] \
                             + normed_rep1_df_tmp[bin_names_list[1]]*norm_list_tmp[1] \
                             + normed_rep1_df_tmp[bin_names_list[2]]*norm_list_tmp[2] \
                             + normed_rep1_df_tmp[bin_names_list[3]]*norm_list_tmp[3])

    for conc in geom_means_rep2:
        normed_rep2_df_tmp = normed_rep2_df.copy().filter(regex=rf"^{conc}_bin.*_normed_counts$", axis=1)
        bin_names_list = list(normed_rep2_df_tmp.columns)
        norm_list_tmp = geom_means_rep2[conc]
        normed_rep2_df[f'score_{conc}'] = (normed_rep2_df_tmp[bin_names_list[0]]*norm_list_tmp[0] \
                             + normed_rep2_df_tmp[bin_names_list[1]]*norm_list_tmp[1] \
                             + normed_rep2_df_tmp[bin_names_list[2]]*norm_list_tmp[2] \
                             + normed_rep2_df_tmp[bin_names_list[3]]*norm_list_tmp[3])
    normed_rep1_df.to_csv(outpath_rep1, index=False)
    normed_rep2_df.to_csv(outpath_rep2, index=False)

if __name__ == "__main__":
    main()
