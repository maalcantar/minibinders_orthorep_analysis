#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 14 18:09:26 2026
@author: alcantar
example run 1: python 16_rosetta_esm_comparison.py -i monomer
example run 2: python 16_rosetta_esm_comparison.py -i complex
"""
# virtual environment: minibinders

from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys

import argparse

from utils import *

def calc_best_ddG_score(parent_pickle_path,
                       mutant_pickle_path):

    '''
    Compare the top 5 lowest energy mutant structures against the top 5 lowest
    energy parental structures.

    PARAMETERS
    -----------
    parent_pickle_path: str
        path to pickle file with parent dG scores
    mutant_pickle_path: str
        path to pickle file with mutant dG scores

    RETURNS
    --------
    mutation_to_ddG_dict: dict
        mutation with ddG scores for all single and higher order mutants
    '''
    parent_dGs = load_pickle(parent_pickle_path)
    mutant_dGs = load_pickle(mutant_pickle_path)

    mutation_to_ddG_dict = {}
    for mutation in parent_dGs:
        parent_dG_curr_mut = parent_dGs[mutation]
        mutant_dG_curr_mut = mutant_dGs[mutation]

        best_parents = sorted(parent_dG_curr_mut)[0:5] # default is lowest to highest
        best_mutants = sorted(mutant_dG_curr_mut)[0:5]

        ddG_curr_mut = np.median(best_mutants) - np.median(best_parents)

        mutation_to_ddG_dict[mutation] = ddG_curr_mut

    return(mutation_to_ddG_dict)

def load_pickle(path):

    '''
    loads pickle file

    PARAMETERS
    -----------
    path: str
        path to pickle file

    RETURNS
    --------
    pickle_file_loaded: dict
        loaded pickle file -- should be a dictionary

    '''
    with open(path, 'rb') as file_to_load:
        pickle_file_loaded = pickle.load(file_to_load)

    return(pickle_file_loaded)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='monomer or complex')
    args = parser.parse_args()

    structure_type=args.i

    esm3_prefix   = '../data/minibinders_orthorep_esm3/'
    outdir = '../figs/esm_rosetta/'
    make_dir(outdir)

    # rosetta pickle files
    if structure_type=='monomer':
        rosetta_prefix = '../data/rosetta_analysis/monomer_results/'
        mb1_parent_rosetta_path = f'{rosetta_prefix}mb1_parent_dG_scores_dict_with_fastrelax_v2.pickle'
        mb1_mutant_rosetta_path = f'{rosetta_prefix}mb1_mutant_dG_scores_dict_with_fastrelax_v2.pickle'

        mb2_parent_rosetta_path = f'{rosetta_prefix}mb2_parent_dG_scores_dict_with_fastrelax_v2.pickle'
        mb2_mutant_rosetta_path = f'{rosetta_prefix}mb2_mutant_dG_scores_dict_with_fastrelax_v2.pickle'

        mb3_parent_rosetta_path = f'{rosetta_prefix}mb3_parent_dG_scores_dict_with_fastrelax_v2.pickle'
        mb3_mutant_rosetta_path = f'{rosetta_prefix}mb3_mutant_dG_scores_dict_with_fastrelax_v2.pickle'

        mb4_parent_rosetta_path = f'{rosetta_prefix}mb4_parent_dG_scores_dict_with_fastrelax_v2.pickle'
        mb4_mutant_rosetta_path = f'{rosetta_prefix}mb4_mutant_dG_scores_dict_with_fastrelax_v2.pickle'

    elif structure_type=='complex':
        rosetta_prefix = '../data/rosetta_analysis/complex_results/'
        mb1_parent_rosetta_path = f'{rosetta_prefix}mb1_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle'
        mb1_mutant_rosetta_path = f'{rosetta_prefix}mb1_mutant_dG_scores_complex_with_fastrelax_v2.pickle'

        mb2_parent_rosetta_path = f'{rosetta_prefix}mb2_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle'
        mb2_mutant_rosetta_path = f'{rosetta_prefix}mb2_mutant_dG_scores_complex_with_fastrelax_v2.pickle'

        mb3_parent_rosetta_path = f'{rosetta_prefix}mb3_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle'
        mb3_mutant_rosetta_path = f'{rosetta_prefix}mb3_mutant_dG_scores_complex_with_fastrelax_v2.pickle'

        mb4_parent_rosetta_path = f'{rosetta_prefix}mb4_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle'
        mb4_mutant_rosetta_path =f'{rosetta_prefix}mb4_mutant_dG_scores_complex_with_fastrelax_v2.pickle'
    else:
        print("Error: Please specify monomer or complex.")
        sys.exit(1)

    # esm3 pickle files
    mb1_esm3_path = '../data/minibinders_orthorep_esm3/mb1_esm3_scores.pickle'
    mb2_esm3_path = '../data/minibinders_orthorep_esm3/mb2_esm3_scores.pickle'
    mb3_esm3_path = '../data/minibinders_orthorep_esm3/mb3_esm3_scores.pickle'
    mb4_esm3_path = '../data/minibinders_orthorep_esm3/mb4_esm3_scores.pickle'

    minibinders = {
        'mb1': {
            'parent_rosetta': mb1_parent_rosetta_path,
            'mutant_rosetta': mb1_mutant_rosetta_path,
            'esm3': mb1_esm3_path,
        },
        'mb2': {
            'parent_rosetta': mb2_parent_rosetta_path,
            'mutant_rosetta': mb2_mutant_rosetta_path,
            'esm3': mb2_esm3_path,
        },
        'mb3': {
            'parent_rosetta': mb3_parent_rosetta_path,
            'mutant_rosetta': mb3_mutant_rosetta_path,
            'esm3': mb3_esm3_path,
        },
        'mb4': {
            'parent_rosetta': mb4_parent_rosetta_path,
            'mutant_rosetta': mb4_mutant_rosetta_path,
            'esm3':mb4_esm3_path,
        },
    }

    for mb_name, paths in minibinders.items():
        ddG = calc_best_ddG_score(paths['parent_rosetta'], paths['mutant_rosetta'])
        esm3_scores = load_pickle(paths['esm3'])

        mutations = list(ddG.keys())
        x_vals = np.array([esm3_scores[m] * -1 for m in mutations])  # NLL
        y_vals = np.array([ddG[m] for m in mutations])

        is_single = np.array(['_' not in m for m in mutations])
        is_multi  = ~is_single

        plt.figure(figsize=(6, 6))

        plt.scatter(x_vals[is_multi], y_vals[is_multi], alpha=0.7, s=100)
        plt.scatter( x_vals[is_single], y_vals[is_single], edgecolors='black',
                     linewidths=1.2, s=200, zorder=3)

        for m, x, y in zip(np.array(mutations)[is_single], x_vals[is_single], y_vals[is_single]):
            plt.text(x, y, m, fontsize=24, ha='right', va='bottom')

        plt.xlabel('ESM3 P(parent residue) - P(mutant residue)')
        plt.ylabel('Rosetta ddG (REU)')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{outdir}{mb_name}_{structure_type}_score_comparison.pdf', dpi=400)

if __name__ == "__main__":
    main()
