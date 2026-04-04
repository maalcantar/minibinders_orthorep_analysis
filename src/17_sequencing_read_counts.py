#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 16 12:10:28 2026
@author: alcantar
example run: python 17_sequencing_read_counts.py

"""
# activate virtual enviroment before running script
# conda activate minibinders

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

from utils import *

def plot_tite_seq_reads(tite_seq_dfs, outdir, reps):

    '''
    Create stacked barplot for tite-seq reads.

    PARAMETERS
    -----------
    tite_seq_dfs: list of pandas dataframes
        tite-seq raw counts dataframes
    reps: list of ints
        replicate numbers
    RETURNS
    --------
        None -- just saves plots
    '''

    pool_prefixes = ['0nM', '0pt01nM','0pt1nM',
                     '1nM','10nM', '100nM','500nM']

    bin_colors = ['#213B65', '#406EA7', '#A8C8DD',
                  '#FFFFFF', '#F4E1D1', '#BD6857',
                  '#5F1A24']

    fig, axes = plt.subplots(len(tite_seq_dfs), 1, figsize=(16, 11))

    plt.rcParams.update({'font.size': 12, 'font.sans-serif': 'Arial'}) # font size 12
    for ax, tite_seq_df, rep in zip(axes, tite_seq_dfs, reps):
        variant_to_reads_dict = dict()
        for idx, row in tite_seq_df.iterrows():
            variant_temp = row['aa_mutations'].strip("[]").replace("'", "") or 'Parent'
            variant_reads_temp = []
            for pool in pool_prefixes:
                pool_reads_temp = sum(row[row.index.str.startswith(pool)])
                variant_reads_temp.append(pool_reads_temp)
            variant_to_reads_dict[variant_temp] = variant_reads_temp

        df = pd.DataFrame.from_dict(variant_to_reads_dict, orient='index', columns=pool_prefixes)
        bottom = np.zeros(len(df))
        for col, color in zip(df.columns, bin_colors):
            ax.bar(df.index, df[col], bottom=bottom, color=color, label=col,
                   edgecolor='black', linewidth=0.5, rasterized=True) # rasterizing; otherwise Illustrator has issues with figures
            bottom += df[col].values

        ax.set_yscale('log')
        ax.set_xlabel('Amino acid mutations')
        ax.set_ylabel('Number of reads')
        ax.set_ylim(bottom=1e2, top=1e5)
        ax.set_title(f'rep {rep}')
        ax.legend(title='Pool', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{outdir}/tite_seq_all_reps.pdf', dpi=400,  bbox_inches='tight')
    plt.savefig(f'{outdir}/tite_seq_all_reps.svg', dpi=400,  bbox_inches='tight')


def tite_seq_reads(outdir= '../figs/raw_counts/'):

    '''
    imports tite-seq results and plots number of tite-seq reads

    PARAMETERS
    -----------
    outdir: str
        path to output

    RETURNS
    --------
        None -- just saves plots
    '''

    tite_seq_path_rep1 = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/raw_tite-seq_counts/mb171_tite-seq_raw_rep1.csv'
    tite_seq_path_rep2 = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/raw_tite-seq_counts/mb171_tite-seq_raw_rep2.csv'
    tite_seq_rep1_df = pd.read_csv(tite_seq_path_rep1, index_col=None)
    tite_seq_rep2_df = pd.read_csv(tite_seq_path_rep2, index_col=None)

    plot_tite_seq_reads(
        tite_seq_dfs=[tite_seq_rep1_df, tite_seq_rep2_df],
        outdir=outdir,
        reps=[1, 2]
    )

def plot_dms_reads(reads_df, mb376_parental, outdir, outname):

    '''
    plots dms reads

    PARAMETERS
    -----------
    reads_df: pandas dataframe
        dms reads
    mb376_parental: str
        parental amino acid sequence
    outdir: str
        output directory
    outname: str
        output name

    RETURNS
    --------
        None -- just saves plots
    '''

    reads_df = reads_df.dropna().reset_index(drop=True)
    variants_to_count_display_dict = dict()
    variants_to_count_binding_dict = dict()

    for idx, row in reads_df.iterrows():
        variant_temp = row['aa_sequences']
        if len(variant_temp) == len(mb376_parental):
            mutations_temp = find_mutations(mb376_parental, variant_temp)
            if len(mutations_temp) <= 1:
                mutation = mutations_temp[0] if mutations_temp else 'parent'
                display_reads = row['display_pseudocounts']
                binding_reads = (row['high_bind_pseudocounts'] + row['low_bind_pseudocounts']
                                 + row['no_bind_pseudocounts'] + row['wildtype_bind_pseudocounts'])
                if mutation not in variants_to_count_display_dict or \
                   display_reads > variants_to_count_display_dict[mutation]:
                    variants_to_count_display_dict[mutation] = display_reads
                    variants_to_count_binding_dict[mutation] = binding_reads

    # Sort by display_reads magnitude (descending) and reindex binding dict to match
    variants_to_count_display_dict = dict(
        sorted(variants_to_count_display_dict.items(), key=lambda x: x[1], reverse=True)
    )
    variants_to_count_binding_dict = {k: variants_to_count_binding_dict[k]
                                      for k in variants_to_count_display_dict}

    categories = list(variants_to_count_display_dict.keys())
    parent_idx = categories.index('parent')

    fig, (ax_display, ax_bind) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.rcParams.update({'font.size': 12, 'font.sans-serif': 'Arial'})

    # Display plot (top)
    ax_display.bar(categories, list(variants_to_count_display_dict.values()),
                   color='black', linewidth=0, rasterized=True)
    ax_display.set_xticks([parent_idx])
    ax_display.set_xticklabels(['parent'])
    ax_display.set_ylabel('Read counts')
    ax_display.set_yscale('log')
    ax_display.axhline(y=100, color='grey', linewidth=1, linestyle='--')
    ax_display.set_ylim(bottom=1e0, top=1e6)

    # Binding plot (bottom)
    ax_bind.bar(categories, list(variants_to_count_binding_dict.values()),
                color='black', linewidth=0, rasterized=True)
    ax_bind.set_xticks([parent_idx])
    ax_bind.set_xticklabels(['parent'])
    ax_bind.set_ylabel('Read pseudocounts')
    ax_bind.set_yscale('log')
    ax_bind.set_ylim(bottom=1e0, top=1e6)

    plt.tight_layout()
    plt.savefig(f'{outdir}{outname}_reads.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'{outdir}{outname}_reads.svg', dpi=400, bbox_inches='tight')
    plt.close()

def main():
    outdir = '../figs/raw_counts/'
    make_dir(outdir)
    tite_seq_reads()

    mb376_parental = 'SLGYYKVTFLPDAHPQAVEILALAFLDNGLEVKEVVTEEGNKYVIAELDEITLEAVKEAIGEIIESVEPVVE'
    mb376_dms_0pt1_path = '../data/dms/mb376_1nM/mb376_dms_raw_counts_0pt1nm.xlsx'
    mb376_dms_10_path = '../data/dms/mb376/mb376_dms_raw_counts_10nm.xlsx'
    mb376_dms_0pt1_df = pd.read_excel(mb376_dms_0pt1_path, engine='openpyxl')
    mb376_dms_10_df = pd.read_excel(mb376_dms_10_path, engine='openpyxl')
    outname_0pt1 = '0pt1nm_reads'
    outname_10 = '10nm_reads'

    plot_dms_reads(mb376_dms_0pt1_df, mb376_parental, outdir  = '../figs/raw_counts/', outname=outname_0pt1)
    plot_dms_reads(mb376_dms_10_df, mb376_parental, outdir  = '../figs/raw_counts/', outname=outname_10)

if __name__ == "__main__":
    main()
