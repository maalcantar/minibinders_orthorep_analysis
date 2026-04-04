#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 19 19:03:20 2026
@author: alcantar
example run 1: python 18_intermediate_affinity_maturation.py
"""
# activate virtual enviroment before running script
# source activate minibinders

import pandas as pd
import numpy as np
import glob as glob
import tqdm
import logomaker
from matplotlib import pyplot as plt

from utils import *

def build_ppm(df):

    '''
    Create a frequency matrix for amino acids at each minibinder position.

    PARAMETERS
    -----------
    df: pandas dataframe
        mutation dataframe with amino acid sequences and read counts

    RETURNS
    --------
    ppm: pandas dataframe
        frequency matrix
    '''

    seqs = df["aa_sequence"].values
    weights = df["read_count"].values
    L = len(seqs[0])
    aas = list("ACDEFGHIKLMNPQRSTVWY") # note that this does not consider stop codons

    # 1-indexed rows
    pfm = pd.DataFrame(0, index=range(1, L + 1), columns=aas)

    for seq, w in tqdm.tqdm(zip(seqs, weights)):
        for i, aa in enumerate(seq):
            if aa in pfm.columns:
                pfm.loc[i + 1, aa] += w  # +1 for 1-based indexing

    ppm = pfm.div(pfm.sum(axis=1), axis=0)
    return ppm

def plot_all_seq_logos(mb_to_paths_dict, positions_to_highlight_dict,
                        outpath, save=False, fig_width=20):

    '''
    Creates sequence logos based on frequency of amino acids at each position.

    PARAMETERS
    -----------
    mb_to_paths_dict: dictionary
        maps each minibinder name to paths to mutation dataframes
    outpath: str
        path to save figure
    save: boolean
        indicate whether to save figure
    fig_width: int
        adjust the width of the figure (adjusting this helps makes sure each position is clear)

    RETURNS
    --------
    NONE -- just plots / saves sequence logos
    '''

    for mb, round_paths in mb_to_paths_dict.items():
        n_rows = len(round_paths)
        positions_to_highlight = positions_to_highlight_dict.get(mb)
        if positions_to_highlight is None:
            print(f'Warning: no highlight positions defined for {mb}')
            positions_to_highlight = []

        fig, axes = plt.subplots(n_rows, 1,
                                figsize=(fig_width, n_rows * 2.5),
                                squeeze=False)

        for row_idx, round_path in enumerate(round_paths):
            ax = axes[row_idx][0]
            df = pd.read_csv(round_path, index_col=0)
            ppm = build_ppm(df)

            round_label = round_path.split('/')[-1].replace('.csv', '')

            logo = logomaker.Logo(ppm, color_scheme='black', ax=ax)
            for pos in positions_to_highlight:
                logo.highlight_position_range(pmin=pos, pmax=pos, color='#6ECFF6')

            logo.style_spines(visible=False)
            logo.style_spines(spines=['left', 'bottom'], visible=True)
            ax.set_title(f"{mb} — {round_label}", fontsize=9)

        plt.suptitle(mb, fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save:
            fig.savefig(f"{outpath}{mb}_sequence_logos.pdf", dpi=400)
            fig.savefig(f"{outpath}{mb}_sequence_logos.png", dpi=400)

def main():
    mb_names = ['mb171', 'mb317', 'mb340', 'mb376']
    path_prefix = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_004/mutation_dfs/'
    outpath = '../figs/intermediate_populations_seq_logos/'
    make_dir(outpath)

    mb_to_paths_dict = dict()
    for mb in mb_names:
        mb_aff_mat_files = glob.glob(f'{path_prefix}{mb}*')
        mb_aff_mat_files.sort()
        mb_to_paths_dict.update({mb: mb_aff_mat_files})

    for mb, round_paths in mb_to_paths_dict.items():
        for round_temp in round_paths:
            seqs_temp = pd.read_csv(round_temp, index_col=0)

    positions_to_highlight_dict = {
        'mb171': [7, 10, 14, 29, 45, 52],
        'mb317': [29, 32, 46, 56, 57],
        'mb340': [10, 26, 41],
        'mb376': [13, 24, 25, 40],
    }
    plot_all_seq_logos(mb_to_paths_dict,
                        positions_to_highlight_dict=positions_to_highlight_dict,
                        outpath=outpath, save=True, fig_width=20)

if __name__ == "__main__":
    main()
