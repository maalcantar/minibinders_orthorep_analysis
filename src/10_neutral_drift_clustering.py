#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on February 26 13:13:19 2025
@author: alcantar
example run: python 10_neutral_drift_clustering.py
"""
# activate virtual enviroment before running script
# source activate minibinders

import pacmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import Counter
from sklearn.cluster import KMeans
import logomaker

from utils import *

import argparse

def retrieve_dms_scores(dms_score_df):

    """
    Converts DMS dataframe into a dictionary where the key is a mutation
    and the value is the DMS enrichment score

    PARAMETERS
    ------------
    df: pandas dataframe
        dms results where column is original residue and index is mutation

    RETURNS
    ------------
    dms_dict: dictionary
        dictionary where key is a mutation and the value is the DMS enrichment score

    """

    dms_dict = {
    f"{col}{idx}": dms_score_df.at[idx, col]
    for col in dms_score_df.columns
    for idx in dms_score_df.index
    }

    return(dms_dict)

def check_for_nan(list_to_check):

    """
    Checks whether a list contains an NaN value

    PARAMETERS
    ------------
    list_to_check: list
        list that will be checked for NaNs

    PARAMETERS
    ------------
    True is list contains NaN
    False if list does NOT contain NaN
    """

    return(any(np.isnan(x) for x in list_to_check if isinstance(x, (int, float))))

def find_enriched_dms_bin(mut_list,
                          hi_bind_scores_dict,
                          wt_bind_scores_dict,
                          low_bind_scores_dict,
                          no_bind_scores_dict):

    """
    Given a list of mutations, returns the most enriched DMS bin

    PARAMETERS
    ------------
    mut_list: list
        list of mutations to check
    hi_bind_scores_dict: dictionary
        maps mutations to enrichhment in high binding bin
    wt_bind_scores_dict: dictionary
        maps mutations to enrichhment in wild type binding bin
    lo_bind_scores_dict: dictionary
        maps mutations to enrichhment in low type binding bin
    no_bind_scores_dict: dictionary
        maps mutations to enrichhment in no type binding bin

    PARAMETERS
    ------------
    dms_mutation_bin_list: list
        list of bin assignments
    """

    dms_mutation_bin_list = []
    index_to_bin_dict = {0: 'No binding',
                         1: 'Low binding',
                         2: 'WT binding',
                         3: 'High binding'}
    for mut in mut_list:
        hi_score_tmp = hi_bind_scores_dict.get(mut, np.nan)
        wt_score_tmp = wt_bind_scores_dict.get(mut, np.nan)
        lo_score_tmp = low_bind_scores_dict.get(mut, np.nan)
        no_score_tmp = no_bind_scores_dict.get(mut, np.nan)

        scores_list_tmp = [no_score_tmp, lo_score_tmp, wt_score_tmp, hi_score_tmp]
        max_index = np.argmax(scores_list_tmp)

        bin_loc = index_to_bin_dict[max_index]

        if check_for_nan(scores_list_tmp):
            bin_loc = 'Not in DMS data'
        dms_mutation_bin_list.append(bin_loc)

    return(dms_mutation_bin_list)

def one_hot_encode_sequences_flat(sequences, max_length=None):

    """
    One-hot encodes a list of amino acid sequences and flattens each sequence into a 1D vector.

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

def plot_enriched_dms_bins(unique_mut_dict,
                           outdir, outname,
                           save=False):
    """
    Create a barplot indicating where the unique mutations are
    most enriched in the DMS data.

    PARAMETERS
    ------------
    unique_mut_dict: dict
        Dictionary mapping the occurrences of unique mutations in DMS bins.
    outdir: str
        output directory
    outname: str
        Name of output file.
    save: boolean
        Whether to save the plot.

    RETURNS
    ------------
    None - just plots and saves.
    """

    category_order = ['High binding', 'WT binding',
                      'Low binding', 'No binding',
                      'Not in DMS data']

    # Ensure all categories are present in the same order
    occurances_in_bins = [unique_mut_dict.get(category, 0) for category in category_order]
    total_mutations = sum(occurances_in_bins)

    # First plot: All categories (including missing ones)
    plt.figure(figsize=(8, 6))
    plt.bar(category_order, occurances_in_bins, color='grey',
            edgecolor='black', linewidth=1)

    # Add percentage labels at the bottom of each bar
    for i, count in enumerate(occurances_in_bins):
        if total_mutations > 0:
            percentage = (count / total_mutations) * 100
            plt.text(i, 1.5, f'{percentage:.1f}%',
                     ha='center', va='top', fontsize=16, color='black')

    plt.xlabel('Most enriched \nDMS bin')
    plt.ylabel('Number of unique mutations')
    plt.tight_layout()

    outpath_full = f"{outdir}/{outname}_full"
    if save:
        plt.savefig(outpath_full+'.pdf', dpi=400)
        plt.savefig(outpath_full+'.png', dpi=400)

    plt.close()  # Close the first figure to prevent overlap

    # Second plot: Only present categories
    present_bins = [category for category in category_order if category in unique_mut_dict]
    occurances_in_bins_partial = [unique_mut_dict[category] for category in present_bins]

    plt.figure(figsize=(8, 6))
    plt.bar(present_bins, occurances_in_bins_partial, color='grey',
            edgecolor='black', linewidth=1)

    plt.xlabel('Most enriched \nDMS bin')
    plt.ylabel('Number of unique mutations')
    plt.tight_layout()

    outpath_partial = f"{outdir}/{outname}_partial"
    if save:
        plt.savefig(outpath_partial+'.pdf', dpi=400)
        plt.savefig(outpath_partial+'.png', dpi=400)
    plt.close()

def plot_logos(annotated_counts_df, outdir,
              outname, save=False):

    """
    plot amino acid sequence logos

    PARAMETERS
    ------------
    annotated_counts_df: pandas dataframe
        annotated counts dataframe which contains a column named "aa_sequence"
    outdir: str
        output directory
    outname: str
        name of the output file
    save: boolean
        whether to save plot

    RETURNS
    ------------
    None - just saves plots
    """

    sequences = annotated_counts_df['aa_sequence'].to_list()
    counts_df = logomaker.alignment_to_matrix(sequences, to_type="counts")

    prob_df = counts_df.div(counts_df.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(50, 5))

    # Create the sequence logo
    logo = logomaker.Logo(prob_df, font_name='Arial', ax=ax, color_scheme='dmslogo_charge')

    if save:
        plt.savefig(f"{outdir}{outname}.png", dpi=400)
        plt.savefig(f"{outdir}{outname}.pdf", dpi=400)
    plt.close()

def main():
    norm_counts_annotated_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/neutral_drift_library/library_scores.csv'
    norm_counts_annotated_df = pd.read_csv(norm_counts_annotated_path, index_col=0)

    wt_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="WT-like"]
    active_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="Active"]
    inactive_annotated_counts_df = norm_counts_annotated_df.copy()[norm_counts_annotated_df['predicted_category']=="Inactive"]

    wt_mutations = [(extract_mutations_from_string(mutations)) \
                   for mutations in wt_annotated_counts_df['aa_mutations']]

    wt_mutations = set(flatten_mutations(wt_mutations))
    wt_mutations.remove('')

    active_mutations = [(extract_mutations_from_string(mutations)) \
                   for mutations in active_annotated_counts_df['aa_mutations']]

    active_mutations = set(flatten_mutations(active_mutations))

    inactive_mutations = [(extract_mutations_from_string(mutations)) \
                           for mutations in inactive_annotated_counts_df['aa_mutations']]

    inactive_mutations = set(flatten_mutations(inactive_mutations))

    wt_only = [x for x in wt_mutations if x not in inactive_mutations]
    inactive_only = [x for x in inactive_mutations if x not in wt_mutations and x not in active_mutations]
    overlap_inactive_wt = [x for x in active_mutations if x in inactive_mutations]

    outdir = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/neutral_drift_analysis_plots/'
    outdir_bars = f"{outdir}dms_comparisons/"
    outdir_logos = f"{outdir}sequence_logos/"
    outdir_pacmap = f"{outdir}PACMap/"

    make_dir(outdir)
    make_dir(outdir_bars)
    make_dir(outdir_logos)
    make_dir(outdir_pacmap)

    hi_bind_dms_path = '../data/dms/mb376/mb376_dms_hi-bind.xlsx'
    wt_bind_dms_path = '../data/dms/mb376/mb376_dms_wt-bind.xlsx'
    lo_bind_dms_path = '../data/dms/mb376/mb376_dms_lo-bind.xlsx'
    no_bind_dms_path = '../data/dms/mb376/mb376_dms_no-bind.xlsx'

    hi_bind_dms_df = pd.read_excel(hi_bind_dms_path, index_col=0)
    wt_bind_dms_df = pd.read_excel(wt_bind_dms_path, index_col=0)
    lo_bind_dms_df = pd.read_excel(lo_bind_dms_path, index_col=0)
    no_bind_dms_df = pd.read_excel(no_bind_dms_path, index_col=0)

    original_residues = [residue[0]+str(aa_no) for aa_no, residue in enumerate(hi_bind_dms_df.columns.to_list(),1)]
    hi_bind_dms_df = hi_bind_dms_df.rename(columns=dict(zip(hi_bind_dms_df.columns.to_list(), original_residues)))
    wt_bind_dms_df = wt_bind_dms_df.rename(columns=dict(zip(wt_bind_dms_df.columns.to_list(), original_residues)))
    lo_bind_dms_df = lo_bind_dms_df.rename(columns=dict(zip(lo_bind_dms_df.columns.to_list(), original_residues)))
    no_bind_dms_df = no_bind_dms_df.rename(columns=dict(zip(no_bind_dms_df.columns.to_list(), original_residues)))

    hi_bind_scores_dict = retrieve_dms_scores(hi_bind_dms_df)
    wt_bind_scores_dict = retrieve_dms_scores(wt_bind_dms_df)
    lo_bind_scores_dict = retrieve_dms_scores(lo_bind_dms_df)
    no_bind_scores_dict = retrieve_dms_scores(no_bind_dms_df)
    wt_only_assignment = find_enriched_dms_bin(wt_only,
                              hi_bind_scores_dict,
                              wt_bind_scores_dict,
                              lo_bind_scores_dict,
                              no_bind_scores_dict)

    inactive_only_assignment = find_enriched_dms_bin(inactive_only,
                              hi_bind_scores_dict,
                              wt_bind_scores_dict,
                              lo_bind_scores_dict,
                              no_bind_scores_dict)

    wt_all_assigment = find_enriched_dms_bin(wt_mutations,
                              hi_bind_scores_dict,
                              wt_bind_scores_dict,
                              lo_bind_scores_dict,
                              no_bind_scores_dict)

    overlap_assigment = find_enriched_dms_bin(overlap_inactive_wt,
                              hi_bind_scores_dict,
                              wt_bind_scores_dict,
                              lo_bind_scores_dict,
                              no_bind_scores_dict)

    inactive_all_assigment = find_enriched_dms_bin(inactive_mutations,
                              hi_bind_scores_dict,
                              wt_bind_scores_dict,
                              lo_bind_scores_dict,
                              no_bind_scores_dict)

    wt_only_counter = Counter(wt_only_assignment)
    inactive_only_counter = Counter(inactive_only_assignment)
    overlap_inactive_wt_counter = Counter(overlap_assigment)

    wt_all_counter = Counter(wt_all_assigment)
    inactive_all_counter = Counter(inactive_all_assigment)

    # plot wt only mutations (not in inactive)
    plot_enriched_dms_bins(wt_only_counter, outdir=outdir_bars,
                           outname='wt_only',
                          save=True)
    # plot inactive only mutations (not in WT or active)
    plot_enriched_dms_bins(inactive_only_counter, outdir=outdir_bars,
                           outname='inactive_only',
                          save=True)

    # plot overlap between wt and inactive bins
    plot_enriched_dms_bins(overlap_inactive_wt_counter, outdir=outdir_bars,
                           outname='wt_and_inactive',
                          save=True)

    # plot all wt
    plot_enriched_dms_bins(wt_all_counter, outdir=outdir_bars,
                           outname='wt_all',
                          save=True)

    # plot all inactive
    plot_enriched_dms_bins(inactive_all_counter, outdir=outdir_bars,
                           outname='inactive_all',
                          save=True)

    plot_logos(wt_annotated_counts_df, outdir=outdir_logos,
                  outname='wt_like_sequences', save=True)


    plot_logos(inactive_annotated_counts_df, outdir=outdir_logos,
              outname='inactive_sequences', save=True)

    # one-hot encode sequences in prep for PACMap mapper
    sequences = norm_counts_annotated_df['aa_sequence'].to_list()
    one_hot_encoded, seq_lengths = one_hot_encode_sequences_flat(sequences)

    classes = norm_counts_annotated_df['predicted_category'].to_list()
    colors = []
    for seq in classes:
        if seq == 'WT-like':
            colors.append('orange')
        elif seq == 'Active':
            colors.append('blue')
        else:
            colors.append('grey')
    # Initialize PACMap
    pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)

    # Fit PACMap
    embedded_sequences = pacmap_reducer.fit_transform(one_hot_encoded)

    plt.scatter(embedded_sequences[:, 0], embedded_sequences[:, 1], alpha=0.5, c=colors)
    plt.xlabel("PACMap Dimension 1")
    plt.ylabel("PACMap Dimension 2")
    plt.title("PACMap Projection of Amino Acid Sequences")
    plt.savefig(f"{outdir_pacmap}pacmap_by_class.png", dpi=400)
    plt.savefig(f"{outdir_pacmap}pacmap_by_class.pdf", dpi=400)
    plt.close()

    # perform K-means clustering
    num_clusters = 4 # you have to specify this
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # this trains the model
    clusters = kmeans.fit_predict(embedded_sequences) # this makes the predictions on your data points

    # Convert to pandas DataFrame
    df = pd.DataFrame(embedded_sequences, columns=["PACMap_1", "PACMap_2"]) # converts PACMap embeddings to a dataframe
    df["Cluster"] = clusters + 1  # start cluster names at 1 as opposed to 0

    cluster_colors = plt.get_cmap("Dark2", num_clusters)  # Get only `num_clusters` colors
    norm = BoundaryNorm(np.arange(0.5, num_clusters + 1.5), num_clusters)  # Ensure proper color mapping

    # Plot results with cluster colors
    plt.scatter(df["PACMap_1"], df["PACMap_2"], c=df["Cluster"], cmap=cluster_colors, norm=norm, alpha=0.6)
    plt.xlabel("PACMap Dimension 1")
    plt.ylabel("PACMap Dimension 2")
    plt.title("PACMap Projection with K-Means Clustering (K=4)")
    plt.colorbar(label="Cluster")
    plt.savefig(f"{outdir_pacmap}pacmap_by_kmean_cluster.png", dpi=400)
    plt.savefig(f"{outdir_pacmap}pacmap_by_kmean_cluster.pdf", dpi=400)
    plt.close()

    norm_counts_annotated_clustered_df = norm_counts_annotated_df.copy()
    norm_counts_annotated_clustered_df['clusters'] = df["Cluster"].to_list()

    # Create figure with subplots (parental at the top + clusters below)
    fig, axes = plt.subplots(nrows=num_clusters + 1, figsize=(50, 5 * (num_clusters + 1)))

    # Plot parental sequence logo in the first subplot
    parental_seq = ['SLGYYKVTFLPDAHPQAVEILALAFLDNGLEVKEVVTEEGNKYVIAELDEITLEAVKEAIGEIIESVEPVVE']
    counts_df = logomaker.alignment_to_matrix(parental_seq, to_type="counts")
    prob_df = counts_df.div(counts_df.sum(axis=1), axis=0)

    logomaker.Logo(prob_df, font_name='Arial', ax=axes[0], color_scheme='dmslogo_charge')
    axes[0].set_title("Parental Sequence", fontsize=20)

    # Iterate over clusters and plot each sequence logo below the parental logo
    for idx, cluster in enumerate(range(1, num_clusters + 1)):
        cluster_seqs_df = norm_counts_annotated_clustered_df[norm_counts_annotated_clustered_df['clusters'] == cluster]
        print(f'cluster {idx}: {cluster_seqs_df.shape[0]} sequences')

        sequences = cluster_seqs_df['aa_sequence'].to_list()
        counts_df = logomaker.alignment_to_matrix(sequences, to_type="counts")
        prob_df = counts_df.div(counts_df.sum(axis=1), axis=0)

        logomaker.Logo(prob_df, font_name='Arial', ax=axes[idx + 1], color_scheme='dmslogo_charge')
        axes[idx + 1].set_title(f"Cluster {cluster}", fontsize=20)

        # Remove x-axis labels and ticks for all except the last plot
        if idx + 1 < num_clusters:
            axes[idx + 1].set_xticks([])
            axes[idx + 1].set_xticklabels([])
            axes[idx + 1].set_xlabel("")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{outdir_logos}cluster_logos.png", dpi=400)
    plt.savefig(f"{outdir_logos}cluster_logos.pdf", dpi=400)
    plt.close()

if __name__ == "__main__":
    main()
