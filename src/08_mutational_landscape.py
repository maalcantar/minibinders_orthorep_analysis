#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 22 12:23:09 2025
@author: alcantar
example run: python 08_mutational_landscape.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/tite-seq_Kds/mb171_EC50_df.csv
"""
# activate virtual enviroment before running script
# source activate minibinders

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import pandas as pd

from utils import *

import argparse

def generate_mutational_graph(values, outpath, cmap_name='RdBu_r', save=False):

    """
    Generate and visualize a mutational landscape graph
    with nodes organized by mutation count and ordered lexicographically.

    PARAMETERS
    -----------
    values: dictionary
        Dictionary where keys are binary strings representing mutations / variant IDs (e.g., '000000'),
        and values are -log10(EC50).
    cmap_name: str
        Name of the colormap to use for node coloring.
    save: bool
        Whether to save the figure as a file.

    RETURNS
    -----------
    NONE - just plots the graph.
    """

    # Initialize graph
    G = nx.Graph()

    # Add nodes with continuous values as attributes
    for subset, value in values.items():
        G.add_node(subset, value=value)

    # Add edges between nodes differing by one mutation (Hamming distance = 1)
    subsets = list(values.keys())
    for i, subset1 in enumerate(subsets):
        for subset2 in subsets[i + 1:]:
            if sum(c1 != c2 for c1, c2 in zip(subset1, subset2)) == 1:
                G.add_edge(subset1, subset2)

    # Extract node values and normalize for colormap
    node_values = np.array([data['value'] for _, data in G.nodes(data=True)])
    norm = Normalize(vmin=node_values.min(), vmax=node_values.max())
    cmap = plt.get_cmap(cmap_name)

    # Map values to colors
    node_colors = [cmap(norm(value)) for value in node_values]

    # Define a custom layout based on the number of mutations (Hamming weight)
    def mutation_count(binary_string):
        return sum(int(bit) for bit in binary_string)

    # Organize nodes by mutation levels
    levels = {}
    for subset in subsets:
        level = mutation_count(subset)
        levels.setdefault(level, []).append(subset)

    # Sort nodes within each level lexicographically in descending order
    for level in levels:
        levels[level].sort(reverse=True)

    # Find the maximum row width
    max_level_width = max(len(nodes) for nodes in levels.values())

    # Assign positions for each node, centering rows and explicitly centering `000000` and `111111`
    layout = {}
    for level, nodes in levels.items():
        n = len(nodes)
        x_positions = np.linspace(-max_level_width / 2, max_level_width / 2, n, endpoint=True)

        for i, node in enumerate(nodes):
            # Explicitly center `000000` and `111111`
            if node == '000000' and level == 0:
                layout[node] = (0, level)  # Center at the bottom
            elif node == '111111' and level == max(levels):
                layout[node] = (0, level)  # Center at the top
            else:
                layout[node] = (x_positions[i], level)  # Standard positioning

    # Draw the graph
    plt.figure(figsize=(18, 12))
    nx.draw(
        G, layout, node_color=node_colors, edgecolors='black', with_labels=True,
        labels={node: f"{node}" for node in G.nodes()}, font_size=12,
        node_size=1800, edge_color="grey"
    )

    # Add a colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(node_values)
    plt.colorbar(sm, label="-log10(EC50)")
    plt.title("Mutational Landscape for mb171")

    if save:
        plt.savefig(f"{outpath}.pdf", dpi=400)
        plt.savefig(f"{outpath}.png", dpi=400)
        plt.savefig(f"{outpath}.svg", dpi=400)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Path to csv with EC50/Kd values')

    args = parser.parse_args()

    EC50_path=args.i #'../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/tite-seq_Kds/mb171_EC50_df.csv'
    split_path_list = EC50_path.split('/')
    mbname = split_path_list[-1].split('_')[0]
    output_dir = '/'.join(EC50_path.split('/')[:-1]).replace('tite-seq_Kds', 'network_analysis') + '/'
    make_dir(output_dir)
    outpath = f"{output_dir}mutational_landscape"
    outpath_signal = f"{output_dir}mutational_landscape_max_signal"

    EC50_df = pd.read_csv(EC50_path, index_col=0, dtype={"variant_id": str})

    X_df = EC50_df[['variant_id']]
    y_df = EC50_df['EC50_mean']
    y_df = -np.log10(y_df)

    y_signal_df = EC50_df['max_signal_mean']
    y_signal_df = np.log10(y_signal_df)

    values_dict = dict(zip(X_df['variant_id'], y_df))
    values_signal_dict = dict(zip(X_df['variant_id'], y_signal_df))

    generate_mutational_graph(values_dict, outpath=outpath, save=True)
    generate_mutational_graph(values_signal_dict, outpath=outpath_signal, save=True)

if __name__ == "__main__":
    main()
