#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 14 15:42:19 2025
@author: alcantar
example run: python 04_create_normalized_tite_seq_counts.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/raw_tite-seq_counts/ -m mb171

"""
# activate virtual enviroment before running script
# source activate minibinders

import sys

import pandas as pd
import numpy as np
import glob as glob

from utils import *

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Directory containing raw counts')
    parser.add_argument('-m', help='minibinder name')

    args = parser.parse_args()

    path_input = args.i
    mb = args.m

    if get_yes_no_input(mb):
        print("Great! Continuing with analysis ...")
    else:
        print("Please use the correct mb name.")
        sys.exit(1)

    # create output directories and paths for both replicates
    if path_input[-1] != '/':
        path_input = path_input + '/'

    output_dir = path_input.replace('raw_tite-seq_counts', 'normalized_tite-seq_counts')
    make_dir(output_dir)

    tite_seq_raw_counts_path_list = sorted(glob.glob(f"{path_input}*.csv"))
    outname_rep1 = tite_seq_raw_counts_path_list[0].split('/')[-1].replace('raw', 'normalized')
    outname_rep2 = tite_seq_raw_counts_path_list[1].split('/')[-1].replace('raw', 'normalized')

    outpath_rep1 = f"{output_dir}{outname_rep1}"
    outpath_rep2 = f"{output_dir}{outname_rep2}"

    if 'rep1' in tite_seq_raw_counts_path_list[0] and 'rep2' in tite_seq_raw_counts_path_list[1]:
        pass
    else:
        print('Error! Replicates are not being read in the proper order.')
        sys.exit(0)

    # read csv with raw counts as csv
    tite_seq_raw_counts_rep1_df = pd.read_csv(tite_seq_raw_counts_path_list[0], index_col=None)
    tite_seq_raw_counts_rep2_df = pd.read_csv(tite_seq_raw_counts_path_list[1], index_col=None)

    # use the first 6 columns to initialize a dataframe without counts
    first_n_columns = 6
    tite_seq_norm_counts_rep1_df = tite_seq_raw_counts_rep1_df.iloc[:, :first_n_columns]
    tite_seq_norm_counts_rep2_df = tite_seq_raw_counts_rep2_df.iloc[:, :first_n_columns]


    # initialize new dataframe
    tite_seq_norm_counts_rep1_df_tmp = tite_seq_norm_counts_rep1_df.copy()
    tite_seq_norm_counts_rep2_df_tmp = tite_seq_norm_counts_rep2_df.copy()

    # number of sorted cells per minibinder / replicate

    if mb=='mb171':
        cell_counts_raw_rep1 = {'0nM': [37565, 190, 0, 0],
                            '0pt01nM': [44373, 4369, 19, 0],
                            '0pt1nM': [43460, 26167, 16709, 241],
                            '1nM': [21191, 31537, 50681, 11190],
                            '10nM': [8034, 25419, 66617, 30548],
                            '100nM': [1227, 12675, 65364, 42433],
                            '500nM': [657, 8298, 55093, 50923]}

        cell_counts_raw_rep2 = {'0nM': [36256, 151, 0, 0],
                            '0pt01nM': [43919, 11790, 1321, 0],
                            '0pt1nM': [35195, 28816, 21039, 1154],
                            '1nM': [16691, 26230, 47786, 13338],
                            '10nM': [7347, 21445, 56268, 27807],
                            '100nM': [1230, 11112, 56187, 37742],
                            '500nM': [317, 5202, 47974, 58254]}
    else:
        print("Error: Minibinder should be mb171.")
        sys.exit(1)  # Exit with a non-zero exit code

    concs_list = list(cell_counts_raw_rep1.keys())

    for conc in concs_list:

        # create list with fraction of cells sorted into each bin
        cell_counts_raw_rep1_tmp = cell_counts_raw_rep1[conc]
        cell_counts_raw_rep2_tmp = cell_counts_raw_rep2[conc]
        cell_counts_norm_rep1_tmp = [cell_count_bin/sum(cell_counts_raw_rep1_tmp) for cell_count_bin in cell_counts_raw_rep1_tmp]
        cell_counts_norm_rep2_tmp = [cell_count_bin/sum(cell_counts_raw_rep2_tmp) for cell_count_bin in cell_counts_raw_rep2_tmp]

        # rep1 start
        # use regular expressions to extract columns, from raw count dataframe,
        # that correspond to thecurrent concentration being analyzed
        # this is not to be confused with the other tmp dataframe, which is used
        # to update the main dataframe
        tite_seq_raw_counts_rep1_tmp_df = tite_seq_raw_counts_rep1_df.copy().filter(regex=rf"^{conc}_bin.*_raw_counts$", axis=1)

        # compute total number of reads per bin (per concentration)
        column_sums_rep1 = tite_seq_raw_counts_rep1_tmp_df.sum().tolist()
        for bn, bin_no in enumerate(list(tite_seq_raw_counts_rep1_tmp_df.columns)):
            # normalize by number of reads and fraction of cells in each bin
            tite_seq_norm_counts_rep1_df_tmp[bin_no] = tite_seq_raw_counts_rep1_tmp_df[bin_no] * cell_counts_norm_rep1_tmp[bn]/ column_sums_rep1[bn]
        # normalize each row by the sum of each row
        tite_seq_norm_counts_rep1_df_tmp.iloc[:, -4:] = (
        tite_seq_norm_counts_rep1_df_tmp.iloc[:, -4:]
        .div(tite_seq_norm_counts_rep1_df_tmp.iloc[:, -4:].sum(axis=1), axis=0)
        )

        # change column names to reflect that they are now nornalized
        tite_seq_norm_counts_rep1_df_tmp.columns = tite_seq_norm_counts_rep1_df_tmp.columns.str.replace("raw_counts", "normed_counts", regex=True)

        # update original dataframe
        tite_seq_norm_counts_rep1_df = pd.concat([tite_seq_norm_counts_rep1_df,tite_seq_norm_counts_rep1_df_tmp.iloc[:, -4:]], axis=1)
        tite_seq_norm_counts_rep1_df = tite_seq_norm_counts_rep1_df.fillna(0)
        # rep 1 end

        # rep 2 start -- same procedure as above
        tite_seq_raw_counts_rep2_tmp_df = tite_seq_raw_counts_rep2_df.copy().filter(regex=rf"^{conc}_bin.*_raw_counts$", axis=1)

        column_sums_rep2 = tite_seq_raw_counts_rep2_tmp_df.sum().tolist()
        for bn, bin_no in enumerate(list(tite_seq_raw_counts_rep2_tmp_df.columns)):
            tite_seq_norm_counts_rep2_df_tmp[bin_no] = tite_seq_raw_counts_rep2_tmp_df[bin_no] * cell_counts_norm_rep2_tmp[bn]/ column_sums_rep2[bn]
        tite_seq_norm_counts_rep2_df_tmp.iloc[:, -4:] = (
        tite_seq_norm_counts_rep2_df_tmp.iloc[:, -4:]
        .div(tite_seq_norm_counts_rep2_df_tmp.iloc[:, -4:].sum(axis=1), axis=0)
        )

        tite_seq_norm_counts_rep2_df_tmp.columns = tite_seq_norm_counts_rep2_df_tmp.columns.str.replace("raw_counts", "normed_counts", regex=True)

        tite_seq_norm_counts_rep2_df = pd.concat([tite_seq_norm_counts_rep2_df,tite_seq_norm_counts_rep2_df_tmp.iloc[:, -4:]], axis=1)
        tite_seq_norm_counts_rep2_df = tite_seq_norm_counts_rep2_df.fillna(0)
        # rep 2 end

        # export dataframes
        tite_seq_norm_counts_rep1_df.to_csv(outpath_rep1, index=False)
        tite_seq_norm_counts_rep2_df.to_csv(outpath_rep2, index=False)

if __name__ == "__main__":
    main()
