#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 2 15:22:28 2026
@author: alcantar
example run: python 20_stringent_vs_neutral.py
"""

# activate virtual enviroment before running script
# source activate minibinders

import pandas as pd
from collections import Counter
from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import pacmap
import seaborn as sns

from utils import *

def aggregate_by_aa_sequence(initial_df):

    """
    aggregate variants by their amino acid sequence. this is meant to consolidate
    sequences that have different dna sequences but the same amino acid sequence.
    this ignores subtle effects that different dna encodings can have on expression

    NOTE: for each amino acid sequence, this returns the most abundant DNA sequence mapping
    to that amino acid sequence

    PARAMETERS
    -----------
    initial_df: pandas dataframe
        dataframe with dna sequences, amino acid sequences, and read counts

    RETURNS
    -----------
    consolidated_df: pandas dataframe
        dataframe with consolidated dna sequences
    """

    df = initial_df.copy()

    consolidated_df = (
    df
    .sort_values(by='read_count', ascending=False)  # Sort by read_count in descending order
    .groupby('aa_sequence', as_index=False)
    .agg({
        'dna_sequence': list, # create a list of all dna sequences which map to an amino acid sequence
        'read_count': 'sum', # sum the read counts
        'dna_mutations': list, # create a list of all dna mutations which map to an amino acid sequence
        'aa_mutations': 'first', # should be the same across all variants, so only take one representative entry
        'number_dna_mutations': list, # create a list of all the number of dna mutations in each dna variant
        'number_aa_mutations': 'first' # should be the same across all variants, so only take one representative entry

    })
)

    return(consolidated_df)

def one_hot_encode_sequences_flat(sequences, max_length=None):

    """
    One-hot encodes a list of amino acid sequences and flattens each sequence into a 1D vector.
    This is equivalent to what is defined in '10_neutral_drift_clustering' but can't import
    a script whose name begins with a number

    PARAMETERS
    ------------
    sequences: list of str
        list of amino acid sequences
    max_length: int, optional
        Fixed length for padding/truncation. If None, uses the max length in input sequences.

    RETURNS
    ------------
    one_hot_matrix: np.ndarray
        one-hot encoded matrix of shape (N, max_length * 20), where N is the number of sequences.
    sequence_lengths: list of int
        Original sequence lengths before padding.
    """

    # Define standard amino acid alphabet
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Determine sequence length for padding/truncation
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    # Initialize one-hot encoded matrix (flattened)
    num_sequences = len(sequences)
    one_hot_matrix = np.zeros((num_sequences, max_length * len(amino_acids)), dtype=np.float32)

    # Encode sequences
    sequence_lengths = []  # Store original lengths before padding
    for i, seq in enumerate(sequences):
        sequence_lengths.append(len(seq))
        for j, aa in enumerate(seq[:max_length]):  # Truncate if longer than max_length
            if aa in aa_to_index:
                index = j * len(amino_acids) + aa_to_index[aa]  # Flattened index
                one_hot_matrix[i, index] = 1.0

    return one_hot_matrix, sequence_lengths

def plot_pairwise_hamming(neutral_df, stringent_df, outdir, save=False):

    """
    Make a  histogram with pairwise hamming distances (hamming distance between
    all possible pairs of variants)

    PARAMETERS
    ------------
    neutral_df: pandas dataframe
        neutral drift mutations dataframe (using the same dataframe used to analyze the library)
    stringent_df: pandas dataframe
        stringent selection dataframe
    outdir: str
        output directory
    save: boolean
        indicate whether to save

    RETURNS
    ------------
    None - just plots the histogram
    """

    dataset_params = [
        (neutral_df,   'Neutral Drift', '#EFA641'),
        (stringent_df, 'Stringent',     '#A73B41')
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    hamming_maxes = []
    for df, label, color in dataset_params:
        aa_sequences = df['aa_sequence'].to_list()
        hamming_pairwise_aa = [
            hamming_distance(a, b)
            for a, b in combinations(aa_sequences, 2)
        ]
        median_val = np.median(hamming_pairwise_aa)
        hamming_maxes.append(max(hamming_pairwise_aa))
        ax.hist(hamming_pairwise_aa, bins=list(range(0,18)), alpha=0.5, color=color,density=True,
                label=f'{label} (median={median_val:.1f})', edgecolor='black')
        ax.axvline(median_val, color=color, linestyle='--', linewidth=1.5)
    ax.set_xticks(range(0, max(hamming_maxes), 2))
    ax.set_xlabel('Pairwise Hamming Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('Pairwise AA Hamming Distance Distributions')
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f'{outdir}pairwise_hamming.pdf', dpi=400)
        plt.close()

def plot_pacmap(neutral_df, stringent_df, outdir, save=False):

    """
    Make scatterplot of PACMAP embeddings

    PARAMETERS
    ------------
    neutral_df: pandas dataframe
        neutral drift mutations dataframe (using the same dataframe used to analyze the library)
    stringent_df: pandas dataframe
        stringent selection dataframe
    outdir: str
        output directory
    save: boolean
        indicate whether to save

    RETURNS
    ------------
    None - just plots the scatterplot
    """

    neutral_df = neutral_df.copy()
    stringent_df = stringent_df.copy()
    neutral_df['source']   = 'Neutral Drift'
    stringent_df['source'] = 'Stringent'
    merged_df = pd.concat([neutral_df, stringent_df], ignore_index=True)

    # One-hot encode all sequences together
    sequences = merged_df['aa_sequence'].to_list()
    one_hot_encoded, _ = one_hot_encode_sequences_flat(sequences)

    # Fit PACMap on the full merged matrix
    pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    embedded_sequences = pacmap_reducer.fit_transform(one_hot_encoded)

    # Assign colors by source
    source_colors = {
        'Neutral Drift': '#EFA641',
        'Stringent':     '#A73B41'}
    colors = merged_df['source'].map(source_colors).to_list()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for source, color in source_colors.items():
        mask = merged_df['source'] == source
        ax.scatter(
            embedded_sequences[mask, 0],
            embedded_sequences[mask, 1],
            alpha=0.5, c=color, label=source, s=75,
        )

    ax.set_xlabel('PACMap 1')
    ax.set_ylabel('PACMap 2')
    ax.legend(markerscale=2)
    plt.tight_layout()

    if save:
        plt.savefig(f'{outdir}pacmap.pdf', dpi=400)
        plt.close()

def find_num_unique_muts(mut_df):
    """
    Find all unique mutations given a dataframe with a column
    containing mutations in a given sequence

    PARAMETERS
    ------------
    mut_df: pandas dataframe
        dataframe with a column containing mutations

    RETURNS
    ------------
    num_unique_muts: int
        number of unique mutations
    """

    mut_list_all = []
    for mut_list in mut_df['aa_mutations']:
        mut_list_temp = extract_mutations_from_string(mut_list)
        mut_list_temp = [mut.replace(' ','') for mut in mut_list_temp]
        for mut in mut_list_temp:
            # handle parent sequence which does not have any mutations, but would otherwise be added to this list as an empty list
            if mut:
                mut_list_all.append(mut)
    num_unique_muts = len(set(mut_list_all))

    return(num_unique_muts)

def plot_unique_muts_bootstrap(neutral_df, stringent_df, outdir, num_bootstraps = 100,
                               num_samples = 100, save=False):

    """
    Bootstrap sequences to compare unique mutations. I am bootstrapping because the number
    of sequences in each condition differ. this only returns results at one
    sequencing depth. 'plot_rarefaction_unique_muts' is an upgraded function that
    returns a whole rarefaction curve.


    PARAMETERS
    ------------
    neutral_df: pandas dataframe
        neutral drift mutations dataframe (using the same dataframe used to analyze the library)
    stringent_df: pandas dataframe
        stringent selection dataframe
    outdir: str
        output directory
    num_bootstraps: int
        number of bootstrap iterations to perform (i.e., how many times to sample dataframes)
    num_samples: int
        number of sequences to compare (i.e., how big of a sample to take per bootstrap iteration)
    save: boolean
        indicate whether to save

    RETURNS
    ------------
    None - just plots the barplots
    """

    neutral_unique_list = []
    stringent_unique_list = []
    for bootstrap_iter in range(num_bootstraps):
        neutral_df_temp = neutral_df.copy().sample(n=num_samples, random_state=bootstrap_iter)
        stringent_df_temp = stringent_df.copy().sample(n=num_samples)

        neutral_unique_temp = find_num_unique_muts(neutral_df_temp)
        stringent_unique_temp = find_num_unique_muts(stringent_df_temp)

        neutral_unique_list.append(neutral_unique_temp)
        stringent_unique_list.append(stringent_unique_temp)

    # for whole df (no bootstrap)
    neutral_unique_all = find_num_unique_muts(neutral_df)
    stringent_unique_all = find_num_unique_muts(stringent_df)

    df = pd.DataFrame()
    df['neutral'] = neutral_unique_list
    df['stringent'] = stringent_unique_list

    fig, ax = plt.subplots(figsize=(3, 3))
    df_long = df.melt(var_name='condition', value_name='unique_muts')
    palette = {'neutral': '#EFA641', 'stringent': '#A73B41'}

    # sns.barplot(df_long, x='condition', y='unique_muts',
    #             errorbar=('sd', 2), palette=palette, edgecolor='black',
    #            capsize=0.1, ax=ax)

    sns.violinplot(df_long, x='condition', y='unique_muts',
               palette=palette, edgecolor='black', ax=ax)
    sns.stripplot(df_long, x='condition', y='unique_muts',
              color='black', size=2, alpha=1.0, jitter=True, ax=ax)

    ax.set_ylim(30, 100)
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(f'{outdir}unique_mutations.pdf', dpi=400)
        plt.close()

    # plt.axhline(neutral_unique_all, color='#EFA641', linestyle='--', linewidth=1.5, label='Neutral (all data)')
    # plt.axhline(stringent_unique_all, color='#A73B41', linestyle='--', linewidth=1.5, label='Stringent (all data)')

    print(f'Neutral drift dataset has {neutral_unique_all} total unique mutations')
    print(f'Stringent dataset has {stringent_unique_all} total unique mutations')

def plot_rarefaction_unique_muts(neutral_df, stringent_df, outdir, num_bootstraps=100,
                                  depth_step=5, max_depth=None, save=False):
    """
    generate a rarefaction to compate unique mutations at various sampling depths.
    this includes bootstrapping at each sampling depth. each sample will be
    plotted to its own respective max sampling depth.

    PARAMETERS
    ------------
    neutral_df: pandas dataframe
        neutral drift mutations dataframe (using the same dataframe used to analyze the library)
    stringent_df: pandas dataframe
        stringent selection dataframe
    outdir: str
        output directory
    num_bootstraps: int
        number of bootstrap iterations to perform (i.e., how many times to sample dataframes)
    depth_step : int
       step size between sampling depths
    max_depth : int or None
        maximum sampling depth. If None, each condition is rarefied to its own
        dataframe length. If specified, the same max depth is applied to both.
        this should be < max dataframe length
    save : boolean
        indicate whether to save figures

    RETURNS
    ------------
    None -- just plots the rarefaction curbe
    """

    palette    = {'neutral': '#EFA641', 'stringent': '#A73B41'}
    conditions = {'neutral':   neutral_df,
                  'stringent': stringent_df}

    max_depths = {
        cond: (max_depth if max_depth is not None else len(df))
        for cond, df in conditions.items()
    }

    records = []
    for cond, df in conditions.items():
        depths = range(depth_step, max_depths[cond] + 1, depth_step)
        for depth in depths:
            for bootstrap_iter in range(num_bootstraps):
                sample = df.sample(n=depth, random_state=bootstrap_iter)
                unique_muts = find_num_unique_muts(sample)
                records.append({'depth': depth, 'condition': cond,
                                 'unique_muts': unique_muts})

    rarefaction_df = pd.DataFrame(records)

    summary = (
        rarefaction_df
        .groupby(['depth', 'condition'])['unique_muts']
        .agg(mean='mean', sd='std')
        .reset_index()
    )
    summary['lower'] = summary['mean'] - 2 * summary['sd']
    summary['upper'] = summary['mean'] + 2 * summary['sd']

    neutral_unique_all   = find_num_unique_muts(neutral_df)
    stringent_unique_all = find_num_unique_muts(stringent_df)

    # plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    for cond in conditions:
        color = palette[cond]
        sub   = summary[summary['condition'] == cond]

        ax.fill_between(sub['depth'].values, sub['lower'].values, sub['upper'].values,
                        color=color, alpha=0.20)
        ax.plot(sub['depth'].values, sub['mean'].values,
                color=color, linewidth=2, label=cond.capitalize())

    # dashed horizontal lines for the full-dataset totals
    ax.axhline(neutral_unique_all,   color=palette['neutral'],
               linestyle='--', linewidth=1.2, label='Neutral (all data)')
    ax.axhline(stringent_unique_all, color=palette['stringent'],
               linestyle='--', linewidth=1.2, label='Stringent (all data)')

    ax.set_xlabel('Sampling depth (# sequences)')
    ax.set_ylabel('Unique mutations')
    ax.set_title('Rarefaction curve – unique mutations')
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig(f'{outdir}rarefaction_unique_mutations.pdf', dpi=400)
        plt.close()

    print(f'Neutral drift dataset  → {neutral_unique_all} total unique mutations '
          f'({len(neutral_df)} sequences, max rarefaction depth: {max_depths["neutral"]})')
    print(f'Stringent dataset      → {stringent_unique_all} total unique mutations '
          f'({len(stringent_df)} sequences, max rarefaction depth: {max_depths["stringent"]})')

def main():
    mb4_neutral_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/neutral_drift_library/library_scores.csv'
    mb4_stringent_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_004/mutation_dfs/mb376_stringent-all_mutation_analysis.csv'

    mb4_neutral_df = pd.read_csv(mb4_neutral_path, index_col=0)
    mb4_stringent_df = pd.read_csv(mb4_stringent_path, index_col=0)
    mb4_stringent_df = aggregate_by_aa_sequence(mb4_stringent_df)

    outdir = '../figs/neutral_v_stringent/'
    make_dir(outdir)

    plot_pairwise_hamming(mb4_neutral_df, mb4_stringent_df, outdir, save=True)
    plot_pacmap(mb4_neutral_df, mb4_stringent_df, outdir, save=True)
    plot_rarefaction_unique_muts(mb4_neutral_df, mb4_stringent_df, outdir, num_bootstraps=100,
                               depth_step=5, max_depth=None, save=True)

if __name__ == "__main__":
    main()
