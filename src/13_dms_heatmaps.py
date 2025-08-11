#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 22 12:16:22 2025
@author: alcantar
example run: python 13_dms_heatmaps.py

"""

# activate virtual enviroment before running script
# source activate minibinders

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils import *

import argparse

# helper function
def swap_keys_values(input_dict):
    return{value: key for key, value in input_dict.items()}

def main():
    output_dir = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/dms_heatmaps/'
    output_dir_10nM = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/dms_heatmaps/dms_10nM/'
    output_dir_1nM = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/dms_heatmaps/dms_1nM/'

    make_dir(output_dir)
    make_dir(output_dir_10nM)
    make_dir(output_dir_1nM)

    ### 10nM DMS ###
    hi_bind_10nM_dms_path = '../data/dms/mb376/mb376_dms_hi-bind.xlsx'
    wt_bind_10nM_dms_path = '../data/dms/mb376/mb376_dms_wt-bind.xlsx'
    lo_bind_10nM_dms_path = '../data/dms/mb376/mb376_dms_lo-bind.xlsx'
    no_bind_10nM_dms_path = '../data/dms/mb376/mb376_dms_no-bind.xlsx'

    hi_bind_10nM_dms_df = pd.read_excel(hi_bind_10nM_dms_path, index_col=0)
    wt_bind_10nM_dms_df = pd.read_excel(wt_bind_10nM_dms_path, index_col=0)
    lo_bind_10nM_dms_df = pd.read_excel(lo_bind_10nM_dms_path, index_col=0)
    no_bind_10nM_dms_df = pd.read_excel(no_bind_10nM_dms_path, index_col=0)



    hi_bind_columns_dict = dict(enumerate(hi_bind_10nM_dms_df.columns.to_list(),1))
    hi_bind_columns_reverse_dict = swap_keys_values(hi_bind_columns_dict)
    hi_bind_10nM_dms_df = hi_bind_10nM_dms_df.rename(columns=hi_bind_columns_reverse_dict)

    wt_bind_columns_dict = dict(enumerate(wt_bind_10nM_dms_df.columns.to_list(),1))
    wt_bind_columns_reverse_dict = swap_keys_values(wt_bind_columns_dict)
    wt_bind_10nM_dms_df = wt_bind_10nM_dms_df.rename(columns=wt_bind_columns_reverse_dict)

    lo_bind_columns_dict = dict(enumerate(lo_bind_10nM_dms_df.columns.to_list(),1))
    lo_bind_columns_reverse_dict = swap_keys_values(lo_bind_columns_dict)
    lo_bind_10nM_dms_df = lo_bind_10nM_dms_df.rename(columns=lo_bind_columns_reverse_dict)

    no_bind_columns_dict = dict(enumerate(no_bind_10nM_dms_df.columns.to_list(),1))
    no_bind_columns_reverse_dict = swap_keys_values(no_bind_columns_dict)
    no_bind_10nM_dms_df = no_bind_10nM_dms_df.rename(columns=no_bind_columns_reverse_dict)

    parent_seq = 'SLGYYKVTFLPDAHPQAVEILALAFLDNGLEVKEVVTEEGNKYVIAELDEITLEAVKEAIGEIIESVEPVVE'
    for pos, residue in enumerate(parent_seq,1):
        hi_bind_10nM_dms_df.loc[residue, pos] = 0.0
        wt_bind_10nM_dms_df.loc[residue, pos] = 0.0
        lo_bind_10nM_dms_df.loc[residue, pos] = 0.0
        no_bind_10nM_dms_df.loc[residue, pos] = 0.0


    concat_data_10nM = pd.concat([hi_bind_10nM_dms_df, wt_bind_10nM_dms_df, lo_bind_10nM_dms_df, no_bind_10nM_dms_df], axis=1).fillna(0).astype('float')
    max_enrich_10nM = concat_data_10nM.max().max()
    min_enrich_10nM = concat_data_10nM.min().min()

    value_to_check = 0.0  # Value to check for

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    hi = sns.heatmap(hi_bind_10nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_10nM, vmax=max_enrich_10nM)
    hi.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(hi_bind_10nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        hi.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_10nM + 'mb376_dms_10nM_hi_bind.pdf')

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    wt = sns.heatmap(wt_bind_10nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_10nM, vmax=max_enrich_10nM)
    wt.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(wt_bind_10nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        wt.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_10nM + 'mb376_dms_10nM_wt_bind.pdf')

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    lo = sns.heatmap(lo_bind_10nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_10nM, vmax=max_enrich_10nM)
    lo.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(lo_bind_10nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        lo.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_10nM + 'mb376_dms_10nM_lo_bind.pdf')

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    no = sns.heatmap(no_bind_10nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_10nM, vmax=max_enrich_10nM)
    no.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(no_bind_10nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        no.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_10nM + 'mb376_dms_10nM_no_bind.pdf')


    ### 0.1nM DMS ###
    hi_bind_1nM_dms_path = '../data/dms/mb376_1nM/mb376_dms_hi-bind_1nM.xlsx'
    wt_bind_1nM_dms_path = '../data/dms/mb376_1nM/mb376_dms_wt-bind_1nM.xlsx'
    lo_bind_1nM_dms_path = '../data/dms/mb376_1nM/mb376_dms_lo-bind_1nM.xlsx'
    no_bind_1nM_dms_path = '../data/dms/mb376_1nM/mb376_dms_no-bind_1nM.xlsx'

    hi_bind_1nM_dms_df = pd.read_excel(hi_bind_1nM_dms_path, index_col=0)
    wt_bind_1nM_dms_df = pd.read_excel(wt_bind_1nM_dms_path, index_col=0)
    lo_bind_1nM_dms_df = pd.read_excel(lo_bind_1nM_dms_path, index_col=0)
    no_bind_1nM_dms_df = pd.read_excel(no_bind_1nM_dms_path, index_col=0)

    hi_bind_columns_dict = dict(enumerate(hi_bind_1nM_dms_df.columns.to_list(),1))
    hi_bind_columns_reverse_dict = swap_keys_values(hi_bind_columns_dict)
    hi_bind_1nM_dms_df = hi_bind_1nM_dms_df.rename(columns=hi_bind_columns_reverse_dict)

    wt_bind_columns_dict = dict(enumerate(wt_bind_1nM_dms_df.columns.to_list(),1))
    wt_bind_columns_reverse_dict = swap_keys_values(wt_bind_columns_dict)
    wt_bind_1nM_dms_df = wt_bind_1nM_dms_df.rename(columns=wt_bind_columns_reverse_dict)

    lo_bind_columns_dict = dict(enumerate(lo_bind_1nM_dms_df.columns.to_list(),1))
    lo_bind_columns_reverse_dict = swap_keys_values(lo_bind_columns_dict)
    lo_bind_1nM_dms_df = lo_bind_1nM_dms_df.rename(columns=lo_bind_columns_reverse_dict)

    no_bind_columns_dict = dict(enumerate(no_bind_1nM_dms_df.columns.to_list(),1))
    no_bind_columns_reverse_dict = swap_keys_values(no_bind_columns_dict)
    no_bind_1nM_dms_df = no_bind_1nM_dms_df.rename(columns=no_bind_columns_reverse_dict)

    for pos, residue in enumerate(parent_seq,1):
        hi_bind_1nM_dms_df.loc[residue, pos] = 0.0
        wt_bind_1nM_dms_df.loc[residue, pos] = 0.0
        lo_bind_1nM_dms_df.loc[residue, pos] = 0.0
        no_bind_1nM_dms_df.loc[residue, pos] = 0.0

    concat_data_1nM = pd.concat([hi_bind_1nM_dms_df, wt_bind_1nM_dms_df, lo_bind_1nM_dms_df, no_bind_1nM_dms_df], axis=1).fillna(0).astype('float')
    max_enrich_1nM = concat_data_1nM.max().max()
    min_enrich_1nM = concat_data_1nM.min().min()

    value_to_check = 0.0  # Value to check for

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    hi = sns.heatmap(hi_bind_1nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_1nM, vmax=max_enrich_1nM)
    hi.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(hi_bind_1nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        hi.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_1nM + 'mb376_dms_1nM_hi_bind.pdf')

    value_to_check = 0.0  # Value to check for

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    wt = sns.heatmap(wt_bind_1nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_1nM, vmax=max_enrich_1nM)
    wt.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(wt_bind_1nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        wt.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_1nM + 'mb376_dms_1nM_wt_bind.pdf')

    value_to_check = 0.0  # Value to check for

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    lo = sns.heatmap(lo_bind_1nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_1nM, vmax=max_enrich_1nM)
    lo.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(lo_bind_1nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        lo.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_1nM + 'mb376_dms_1nM_lo_bind.pdf')

    value_to_check = 0.0  # Value to check for

    plt.figure(figsize=(25, 5))  # Adjust figure size as needed
    no = sns.heatmap(no_bind_1nM_dms_df, annot=False, cmap="RdBu_r", center=0, vmin=min_enrich_1nM, vmax=max_enrich_1nM)
    no.set_facecolor('black')

    # Get indices where values are 0.0
    zero_indices = np.where(no_bind_1nM_dms_df.to_numpy() == value_to_check)

    # Annotate only the zero values
    for i, j in zip(*zero_indices):
        no.text(j + 0.5, i + 0.5, "X", ha='center', va='center', color='black', fontsize=12)
    plt.savefig(output_dir_1nM + 'mb376_dms_1nM_no_bind.pdf')

if __name__ == "__main__":
    main()
