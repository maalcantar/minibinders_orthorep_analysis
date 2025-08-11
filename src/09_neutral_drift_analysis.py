#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 29 08:24:51 2025
@author: alcantar
example run: python 09_neutral_drift_analysis.py
"""
# activate virtual enviroment before running script
# source activate minibinders

import pandas as pd
import tqdm
import math
from matplotlib import pyplot as plt
import scipy.stats as stats
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator
from itertools import combinations
import seaborn as sns

from utils import *

def retrieve_filter_counts(master_df,
                           bin1_dict,
                           bin2_dict,
                           bin3_dict,
                           bin4_dict,
                           count_threshold=10):

    """
    Populate a dataframe with read counts for each amino acid sequence in each bin.
    Practically, this takes in several dictionaries which map an amino acid sequence
    with it's corresponding read count in a given bin (i.e., WT-like binding) which are
    used to populate a dataframe.

    bin_1 - no binding
    bin_2 - low binding
    bin_3 - WT-like binding
    bin_4 - high binding

    PARAMETERS
    -----------
    master_df: pandas dataframe
        dataframe with all the amino acid sequences that will be analyzed / for which
        we want read counts
    bin1_dict: dictionary
        dictionary mapping aa sequence to read count in no binding bin
    bin2_dict: dictionary
        dictionary mapping aa sequence to read count in low binding bin
    bin3_dict: dictionary
        dictionary mapping aa sequence to read count in wt-like binding bin
    bin4_dict: dictionary
        dictionary mapping aa sequence to read count in high binding bin
    count_threshold: int
        minimun number of reads across conditions to consider sequence to analysis

    RETURNS
    -----------
    df: pandas dataframe
        dataframe with read counts per bin populated
    """

    # Use the 'get' method of dictionaries on the 'dna_sequence' column with vectorized operations
    df = master_df.copy()

    # Add new columns directly using pandas' Series mapping
    df['bin_1'] = df['aa_sequence'].map(bin1_dict).fillna(0).astype(int)
    df['bin_2'] = df['aa_sequence'].map(bin2_dict).fillna(0).astype(int)
    df['bin_3'] = df['aa_sequence'].map(bin3_dict).fillna(0).astype(int)
    df['bin_4'] = df['aa_sequence'].map(bin4_dict).fillna(0).astype(int)

    # Filter rows where the sum of the last four columns meets the threshold
    df = df[df[['bin_1', 'bin_2', 'bin_3', 'bin_4']].sum(axis=1) >= count_threshold]

    # Reset the index
    df.reset_index(inplace=True, drop=True)
    return(df)

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

def plot_hamming_distances(annotated_counts_df,
                           out_name_1,
                           out_name_2):
    """
    Plot hamming distance to parent sequence and pairwise hamming distance
    as a histogram.


    annotated_counts_df: pandas dataframe
        dataframe with different variants and number of mutations
    out_name_1: str
        output name for hamming distance plot versus parent
    out_name_2: str
        output name for pairwise hamming distance plot

    RETURNS
    -----------
    NONE



    """

    # plot versus parent
    num_sequences = annotated_counts_df.shape[0]
    # Set the Seaborn style and context
    sns.set(style="white")

    # Set global font size and family
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial'
    })
    # Set the Seaborn style
    sns.set_style("ticks")


    g = sns.histplot(data=annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='probability',
                         edgecolor='black', linewidth=0.5)
    ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = g.get_xticks()
    new_ticks = ticks[:-1] + 0.5
    g.set_xticks(new_ticks)
    g.set_xticklabels([str(int(t)) for t in new_ticks])

    g.set_xlabel('amino acid mutations')
    g.set_ylabel(f'frequency (n={num_sequences} sequences)')
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    for spine in ['left', 'bottom']:
        g.spines[spine].set_linewidth(0.5)
    g.tick_params(width=0.5)
    g.set_xlim([new_ticks[0]+0.5, new_ticks[-1]-0.5])
    outname_hamming_dir = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/neutral_drift_library/plots/'
    outname_hamming_to_parent_plot =outname_hamming_dir + out_name_1
    # Show the plot
    plt.savefig(f'{outname_hamming_to_parent_plot}.png', dpi=400)
    plt.savefig(f'{outname_hamming_to_parent_plot}.pdf', dpi=400)
    plt.savefig(f'{outname_hamming_to_parent_plot}.svg', dpi=400)
    plt.close()

    #plot pairwise hamming
    aa_sequences = annotated_counts_df['aa_sequence'].to_list()
    hamming_pairwise_aa = []
    for aa_seq_combo in combinations(aa_sequences, 2):
        hamming_pairwise_aa.append(hamming_distance(aa_seq_combo[0], aa_seq_combo[1]))

    g = sns.histplot(data=hamming_pairwise_aa,
                             binwidth=1,stat='probability',
                             edgecolor='black', linewidth=0.5)
    # sns.histplot(data=norm_counts_annotated_df, x="number_aa_mutations",
    #                  binwidth=1,stat='probability',
    #                  edgecolor='black', linewidth=0.5, color='gray')
    ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = g.get_xticks()
    new_ticks = ticks[:-1] + 0.5
    g.set_xticks(new_ticks)
    g.set_xticklabels([str(int(t)) for t in new_ticks])
    g.set_ylabel(f'frequency (n={num_sequences} sequences)')

    g.set_xlabel('amino acid mutations')
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    for spine in ['left', 'bottom']:
        g.spines[spine].set_linewidth(0.5)
    g.tick_params(width=0.5)
    g.set_xlim([new_ticks[0]+0.5, new_ticks[-1]-0.5])


    outname_hamming_dir = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/neutral_drift_library/plots/'
    outname_hamming_pairwise_plot =outname_hamming_dir + out_name_2
    # Show the plot
    plt.savefig(f'{outname_hamming_pairwise_plot}.png', dpi=400)
    plt.savefig(f'{outname_hamming_pairwise_plot}.pdf', dpi=400)
    plt.savefig(f'{outname_hamming_pairwise_plot}.svg', dpi=400)
    plt.close()

def main():
    # define parent / wt sequence
    parent_dna_seq = "TCTTTAGGTTATTATAAAGTTACTTTTTTACCTGATGCTCATCCTCAAGCTGTTGAAATTTTAGCTTTAGCTTTTTTAGATAATGGTTTAGAAGTTAAAGAAGTTGTTACTGAAGAAGGTAATAAATATGTTATTGCTGAATTAGATGAAATTACTTTAGAAGCTGTTAAAGAAGCTATTGGTGAAATTATTGAATCTGTTGAACCTGTTGTTGAA"

    # read in relevant dataframes
    wt_rep1_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-wtbinding-rep1_mutation_analysis.csv'
    wt_rep2_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-wtbinding-rep2_mutation_analysis.csv'

    low_rep1_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-lowbinding-rep1_mutation_analysis.csv'
    low_rep2_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-lowbinding-rep2_mutation_analysis.csv'

    hi_rep1_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-highbinding-rep1_mutation_analysis.csv'
    hi_rep2_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-highbinding-rep2_mutation_analysis.csv'

    no_rep1_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-nobinding-rep1_mutation_analysis.csv'
    no_rep2_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/mutation_dfs/mb376-drift-nobinding-rep2_mutation_analysis.csv'

    out_dir = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/neutral_drift_library/'
    make_dir(out_dir)
    out_dir_figs = out_dir+'plots/'
    make_dir(out_dir_figs)

    wt_rep1_df = pd.read_csv(wt_rep1_path, index_col=0)
    wt_rep2_df = pd.read_csv(wt_rep2_path, index_col=0)

    low_rep1_df = pd.read_csv(low_rep1_path, index_col=0)
    low_rep2_df = pd.read_csv(low_rep2_path, index_col=0)

    hi_rep1_df = pd.read_csv(hi_rep1_path, index_col=0)
    hi_rep2_df = pd.read_csv(hi_rep2_path, index_col=0)

    no_rep1_df = pd.read_csv(no_rep1_path, index_col=0)
    no_rep2_df = pd.read_csv(no_rep2_path, index_col=0)

    # consilidate by amino acid sequence
    wt_rep1_df = aggregate_by_aa_sequence(wt_rep1_df)
    wt_rep2_df = aggregate_by_aa_sequence(wt_rep2_df)

    low_rep1_df = aggregate_by_aa_sequence(low_rep1_df)
    low_rep2_df = aggregate_by_aa_sequence(low_rep2_df)

    hi_rep1_df = aggregate_by_aa_sequence(hi_rep1_df)
    hi_rep2_df = aggregate_by_aa_sequence(hi_rep2_df)

    no_rep1_df = aggregate_by_aa_sequence(no_rep1_df)
    no_rep2_df = aggregate_by_aa_sequence(no_rep2_df)

    # create master dataframe with all sequences that will be considered
    # these sequences will be filtered using a read count threshold
    master_rep1_df = pd.concat([wt_rep1_df,
                               low_rep1_df,
                               hi_rep1_df,
                               no_rep1_df]).reset_index(drop=True).drop(['read_count'], axis=1)

    master_rep2_df = pd.concat([wt_rep2_df,
                               low_rep2_df,
                               hi_rep2_df,
                               no_rep2_df]).reset_index(drop=True).drop(['read_count'], axis=1)

    # drop duplicate rows with the same amino acid sequence
    master_rep1_df = master_rep1_df.drop_duplicates(subset=['aa_sequence'],).reset_index(drop=True)
    master_rep2_df = master_rep2_df.drop_duplicates(['aa_sequence']).reset_index(drop=True)


    # populate master_dfs with the read counts and also apply read count threshold filter
    count_threshold = 21

    master_rep1_df = master_rep1_df.assign(bin_1=0, bin_2=0, bin_3=0, bin_4=0)
    master_rep2_df = master_rep2_df.assign(bin_1=0, bin_2=0, bin_3=0, bin_4=0)

    no_rep1_counts_dict = dict(zip(no_rep1_df['aa_sequence'],no_rep1_df['read_count']))
    no_rep2_counts_dict = dict(zip(no_rep2_df['aa_sequence'],no_rep2_df['read_count']))

    low_rep1_counts_dict = dict(zip(low_rep1_df['aa_sequence'],low_rep1_df['read_count']))
    low_rep2_counts_dict = dict(zip(low_rep2_df['aa_sequence'],low_rep2_df['read_count']))

    wt_rep1_counts_dict = dict(zip(wt_rep1_df['aa_sequence'],wt_rep1_df['read_count']))
    wt_rep2_counts_dict = dict(zip(wt_rep2_df['aa_sequence'],wt_rep2_df['read_count']))

    hi_rep1_counts_dict = dict(zip(hi_rep1_df['aa_sequence'],hi_rep2_df['read_count']))
    hi_rep2_counts_dict = dict(zip(hi_rep2_df['aa_sequence'],hi_rep2_df['read_count']))

    raw_counts_rep1_df = retrieve_filter_counts(master_rep1_df,
                               no_rep1_counts_dict,
                               low_rep1_counts_dict,
                               wt_rep1_counts_dict,
                               hi_rep1_counts_dict,
                               count_threshold=count_threshold)

    raw_counts_rep2_df = retrieve_filter_counts(master_rep2_df,
                               no_rep2_counts_dict,
                               low_rep2_counts_dict,
                               wt_rep2_counts_dict,
                               hi_rep2_counts_dict,
                               count_threshold=count_threshold)


    # calculate cell fractions in each bin
    rep1_cell_counts = [53830, 64696, 298413, 138]
    rep1_cell_fracs = [count/sum(rep1_cell_counts) for count in rep1_cell_counts]

    rep2_cell_counts = [48727, 51098, 244362, 372]
    rep2_cell_fracs = [count/sum(rep2_cell_counts) for count in rep2_cell_counts]

    # create new dataframes that will contain normalized scores
    first_n_columns = 6
    norm_counts_rep1_df = raw_counts_rep1_df.copy().iloc[:, :first_n_columns]
    norm_counts_rep2_df = raw_counts_rep2_df.copy().iloc[:, :first_n_columns]

    # create dataframes which only contain read counts and will be used to calculate normalized scores
    raw_counts_only_rep1_df = raw_counts_rep1_df.copy().iloc[:,-4:]
    raw_counts_only_rep2_df = raw_counts_rep2_df.copy().iloc[:,-4:]

    # apply sort-seq-esque normalization to both replicates

    # compute total number of reads per bin (per concentration)
    column_sums_rep1 = raw_counts_only_rep1_df.sum().tolist()
    for bn, bin_no in enumerate(list(raw_counts_only_rep1_df.columns)):
        # normalize by number of reads and fraction of cells in each bin
        norm_counts_rep1_df[bin_no] = raw_counts_only_rep1_df[bin_no] * rep1_cell_fracs[bn]/ column_sums_rep1[bn]
    # normalize each row by the sum of each row
    norm_counts_rep1_df.iloc[:, -4:] = (
    norm_counts_rep1_df.iloc[:, -4:]
    .div(norm_counts_rep1_df.iloc[:, -4:].sum(axis=1), axis=0)
    )

    norm_counts_rep1_df['normalized_score_rep1'] = (norm_counts_rep1_df['bin_1']*0 \
                                                                           + norm_counts_rep1_df['bin_2']*(1/3) \
                                                                           + norm_counts_rep1_df['bin_3']*(2/3) \
                                                                           + norm_counts_rep1_df['bin_4']*(3/3))

    # compute total number of reads per bin (per concentration)
    column_sums_rep2 = raw_counts_only_rep2_df.sum().tolist()
    for bn, bin_no in enumerate(list(raw_counts_only_rep2_df.columns)):
        # normalize by number of reads and fraction of cells in each bin
        norm_counts_rep2_df[bin_no] = raw_counts_only_rep2_df[bin_no] * rep2_cell_fracs[bn]/ column_sums_rep2[bn]
    # normalize each row by the sum of each row
    norm_counts_rep2_df.iloc[:, -4:] = (
    norm_counts_rep2_df.iloc[:, -4:]
    .div(norm_counts_rep2_df.iloc[:, -4:].sum(axis=1), axis=0)
    )

    norm_counts_rep2_df['normalized_score_rep2'] = (norm_counts_rep2_df['bin_1']*0 \
                                                                           + norm_counts_rep2_df['bin_2']*(1/3) \
                                                                           + norm_counts_rep2_df['bin_3']*(2/3) \
                                                                           + norm_counts_rep2_df['bin_4']*(3/3))

    # merge both replicates for amino acid sequences that appear in both replicates
    # Get the last column of norm_counts_rep2_df
    last_column = norm_counts_rep2_df[["aa_sequence", "normalized_score_rep2"]]

    # Merge norm_counts_rep1_df with the last column from norm_counts_rep2_df based on dna_sequence
    norm_counts_df = norm_counts_rep1_df.copy().merge(last_column, on="aa_sequence", how="inner")



    # remove highly variant sequences differing by 1/3 a.u. -- note that
    # differing by 1/3 a.u. means that, on average, the bins
    # were sorted into different bins
    for idx, row in tqdm.tqdm(norm_counts_df.iterrows(), total = norm_counts_df.shape[0]):
        replicate_difference = abs(row['normalized_score_rep1'] - row['normalized_score_rep2'])
        if replicate_difference >= (1/3):
            norm_counts_df = norm_counts_df.drop(idx)

    norm_counts_annotated_df = norm_counts_df.copy()

    # Prepare feature matrix (normalized scores)
    X = norm_counts_annotated_df[['normalized_score_rep1', 'normalized_score_rep2']].values

    # Use KMeans clustering to group points into 3 categories
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X)

    # Map cluster labels to descriptive categories (assign based on mean locations)
    cluster_centers = kmeans.cluster_centers_
    sorted_indices = np.argsort(cluster_centers[:, 0])  # Sort clusters by normalized_score_rep1
    cluster_map = {sorted_indices[0]: 'Inactive',
                   sorted_indices[1]: 'Active',
                   sorted_indices[2]: 'WT-like'}

    # Assign labels to dataframe
    norm_counts_annotated_df['predicted_category'] = [cluster_map[label] for label in kmeans_labels]

    # Map categories to colors for visualization
    category_colors = {
        'Inactive': 'black',
        'Active': '#3B71B2',
        'WT-like': 'orange'
    }
    norm_counts_annotated_df['color'] = norm_counts_annotated_df['predicted_category'].map(category_colors)

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with predicted colors
    ax.scatter(
        norm_counts_annotated_df['normalized_score_rep1'],
        norm_counts_annotated_df['normalized_score_rep2'],
        c=norm_counts_annotated_df['color'],
        edgecolors='black',
        s=100,
        alpha=0.75,
        label="Predicted categories"
    )

    # Highlight parent DNA sequence
    # parent_mask = norm_counts_annotated_df['dna_sequence'] == parent_dna_seq
    parent_mask = norm_counts_annotated_df['dna_sequence'].apply(lambda seq_list: parent_dna_seq in seq_list)

    ax.scatter(
        norm_counts_annotated_df['normalized_score_rep1'][parent_mask],
        norm_counts_annotated_df['normalized_score_rep2'][parent_mask],
        c='orange',
        edgecolors='black',
        s=750,
        linewidth=2,
        label="Parent DNA sequence"
    )

    # Add labels, legend, and title
    ax.set_xlabel('Normalized Score Replicate 1', fontsize=16)
    ax.set_ylabel('Normalized Score Replicate 2', fontsize=16)
    ax.set_title('K-means clustering of minibinder activity', fontsize=16)
    ax.legend(fontsize=12, loc="best")

    rep1_scores = norm_counts_annotated_df['normalized_score_rep1']
    rep2_scores = norm_counts_annotated_df['normalized_score_rep2']
    correlation_coefficient, p_value = stats.pearsonr(rep1_scores, rep2_scores)
    correlation_coefficient_spear, p_value_spear = stats.spearmanr(rep1_scores, rep2_scores)

    textstr = (f"Pearson's r: {correlation_coefficient:.2f}, p = {p_value:.2e}\n"
               f"Spearman's r: {correlation_coefficient_spear:.2f}, p = {p_value_spear:.2e}")
    plt.gca().text(0.01, 0.8, textstr, transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    out_name = out_dir + 'library_scores.csv'
    out_name_fig = out_dir_figs + 'library_scores'

    # save plot
    plt.savefig(out_name_fig+'.png', dpi=400)
    plt.savefig(out_name_fig+'.pdf', dpi=400)
    plt.savefig(out_name_fig+'.svg', dpi=400)
    plt.close()
    # save dataframe
    norm_counts_annotated_df.drop('color', axis=1).to_csv(out_name)

    total_seqs = norm_counts_annotated_df.shape[0]
    num_wt_like_seqs = (norm_counts_annotated_df['predicted_category'] == 'WT-like').sum()
    num_active_seqs = (norm_counts_annotated_df['predicted_category'] == 'Active').sum()
    active_wt_seqs = num_wt_like_seqs + num_active_seqs
    num_unactive_sequences = (norm_counts_annotated_df['predicted_category'] == 'Inactive').sum()

    percent_wt = num_wt_like_seqs/total_seqs*100
    percent_active = num_active_seqs/total_seqs*100

    percent_wt_active = (num_wt_like_seqs + num_active_seqs) / total_seqs * 100

    percent_inactive = num_unactive_sequences / total_seqs * 100

    text_file_name = out_dir + 'results_summary.txt'
    with open(text_file_name, "w") as file:
        file.write(f"{percent_wt:.2f}% of sequences ({num_wt_like_seqs}/{total_seqs}) are WT-like.\n")
        file.write(f"{percent_wt_active:.2f}% of sequences ({num_wt_like_seqs + num_active_seqs}/{total_seqs}) are active.\n")
        file.write(f"{percent_inactive:.2f}% of sequences ({num_unactive_sequences}/{total_seqs}) are inactive.")

    # plot hamming distances
    # plot all hamming
    plot_hamming_distances(annotated_counts_df=norm_counts_annotated_df,
                           out_name_1='hamming_vs_parent_all' ,
                           out_name_2='pairwise_hamming_all')

    wt_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="WT-like"]
    active_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="Active"]
    inactive_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="Inactive"]


    plot_hamming_distances(annotated_counts_df=wt_annotated_counts_df,
                           out_name_1='hamming_vs_parent_WT-like' ,
                           out_name_2='pairwise_hamming_WT-like')
    plot_hamming_distances(annotated_counts_df=active_annotated_counts_df,
                           out_name_1='hamming_vs_parent_active' ,
                           out_name_2='pairwise_hamming_active')
    plot_hamming_distances(annotated_counts_df=inactive_annotated_counts_df,
                           out_name_1='hamming_vs_parent_inactive' ,
                           out_name_2='pairwise_hamming_inactive')
    # plot versus parent
    num_sequences = norm_counts_annotated_df.shape[0]
    # Set the Seaborn style and context
    sns.set(style="white")

    # Set global font size and family
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial'
    })
    # Set the Seaborn style
    sns.set_style("ticks")

    wt_mean = np.mean(wt_annotated_counts_df['number_aa_mutations'])
    wt_median = np.median(wt_annotated_counts_df['number_aa_mutations'])

    active_mean = np.mean(active_annotated_counts_df['number_aa_mutations'])
    active_median = np.median(active_annotated_counts_df['number_aa_mutations'])

    inactive_mean = np.mean(inactive_annotated_counts_df['number_aa_mutations'])
    inactive_median = np.median(inactive_annotated_counts_df['number_aa_mutations'])

    g = sns.histplot(data=wt_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='probability',
                         edgecolor='black', color='orange', linewidth=0.5,
                         alpha=0.5,
                         label=f"WT-like (n={wt_annotated_counts_df.shape[0]} [{wt_mean:.1f}|{wt_median:.1f}])")
    sns.histplot(data=active_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='probability',
                         edgecolor='black',color='#3B71B2', linewidth=0.5,
                         alpha=0.5,
                         label=f"Active (n={active_annotated_counts_df.shape[0]} [{active_mean:.1f}|{active_median:.1f}])")
    sns.histplot(data=inactive_annotated_counts_df, x="number_aa_mutations",
                         binwidth=1,stat='probability',
                         edgecolor='black', color='grey',linewidth=0.5,
                         alpha=0.5,
                         label=f"Inactive (n={inactive_annotated_counts_df.shape[0]}[{inactive_mean:.1f}|{inactive_median:.1f}])")
    ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = g.get_xticks()
    new_ticks = ticks[:-1] + 0.5
    g.set_xticks(new_ticks)
    g.set_xticklabels([str(int(t)) for t in new_ticks])

    g.set_xlabel('amino acid mutations')
    g.set_ylabel(f'frequency (n={num_sequences} sequences)')
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    for spine in ['left', 'bottom']:
        g.spines[spine].set_linewidth(0.5)
    g.tick_params(width=0.5)
    g.set_xlim([new_ticks[0]+0.5, new_ticks[-1]-0.5])
    plt.legend(loc='upper left')
    outname_hamming_dir = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/neutral_drift_library/plots/'
    outname_histogram_all_classes =outname_hamming_dir + 'hamming_all_classes'
    plt.savefig(outname_histogram_all_classes + '.png', dpi=400)
    plt.savefig(outname_histogram_all_classes + '.pdf', dpi=400)
    plt.savefig(outname_histogram_all_classes + '.svg', dpi=400)
    plt.close()
if __name__ == "__main__":
    main()
