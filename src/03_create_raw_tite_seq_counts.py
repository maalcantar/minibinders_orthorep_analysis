#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 14 11:29:53 2025
@author: alcantar
example run: python 03_create_raw_tite_seq_counts.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/mutation_dfs/ -m mb171

"""
# activate virtual enviroment before running script
# source activate minibinders
import sys

import pandas as pd
import glob as glob

from utils import *

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Directory containing mutation_dfs')
    parser.add_argument('-m', help='minibinder name')

    args = parser.parse_args()

    mutation_df_dir = args.i #'../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/mutation_dfs/'
    mb = args.m #'mb171'

    if get_yes_no_input(mb):
        print("Great! Continuing with analysis ...")
    else:
        print("Please use the correct mb name.")
        sys.exit(1)

    if mutation_df_dir[-1] != '/': # add '/' if missing
        mutation_df_dir = mutation_df_dir + '/'

    # directory and path prefix for output data
    output_dir = mutation_df_dir.replace('mutation_dfs', 'raw_tite-seq_counts')
    make_dir(output_dir)
    output_name_prefix = f"{output_dir}{mb}_tite-seq"

    conc_names = ['0nM', '0pt01nM', '0pt1nM', '1nM', '10nM', '100nM', '500nM']
    bin_names = ['bin1', 'bin2', 'bin3', 'bin4']

    conc_bins_list = []
    for concentration_no in conc_names:
        for bin_no in bin_names:
            conc_bins_list.append(f"{concentration_no}_{bin_no}_raw_counts")

    # create combinatorial library in silico -- these parts correspond to the
    # oligos used to create the combinatorial libraries.

    if mb == 'mb171':
        part1_seqs = ['ACGCATGTTGTCTAACTTTTGAAAACCCTGAGTTCGCCGAACAAGCTGTGCTGATT',
                 'ACGCATGTTGTCTAACTTTTGAAGACCCTGAGTTCGCCGAACAAGCTGTGCTGATT',
                 'ACGCATGTTGTCTAACTTTTGAAAACCCTGAGTACGCCGAACAAGCTGTGCTGATT',
                 'ACGCATGTTGTCTAACTTTTGAAAACCCTGAGTTCGCCGAACAAGTTGTGCTGATT',
                 'ACGCATGTTGTCTAACTTTTGAAGACCCTGAGTACGCCGAACAAGCTGTGCTGATT',
                 'ACGCATGTTGTCTAACTTTTGAAGACCCTGAGTTCGCCGAACAAGTTGTGCTGATT',
                 'ACGCATGTTGTCTAACTTTTGAAAACCCTGAGTACGCCGAACAAGTTGTGCTGATT',
                 'ACGCATGTTGTCTAACTTTTGAAGACCCTGAGTACGCCGAACAAGTTGTGCTGATT']
        part2_seqs = ['GCCGAAGATGCTGGCTTGAAAGTCGAGGTAAAGCCAGGTGAAGACT',
                     'GCCGAAGATGCTGGCTTGAAAGTCGAGGTAAAGTCAGGTGAAGACT']
        part3_seqs = ['CATTAACCGTTTGCCTTTCTAATGAAGCAGCATGTAAGTATTTTGC',
                     'CATTAACCGTTTGCCTTTCTAATGAAGCAGCATGTGAGTATTTTGC']
        part4_seqs = ['TGAAAGAGCGAAAGAGATGGGAGTTCCCGTTACAGTCGAACCAG',
                     'TGAAAGAGCGGAAGAGATGGGAGTTCCCGTTACAGTCGAACCAG']
    else:
        print("Error: Minibinder should be mb171.")
        sys.exit(1)  # Exit with a non-zero exit code

    mb_combo_library_dna = []
    for part1 in part1_seqs:
        part1 = part1[5:]
        for part2 in part2_seqs:
            for part3 in part3_seqs:
                for part4 in part4_seqs:
                    part4 = part4[:-1]
                    combo = part1+part2+part3+part4
                    mb_combo_library_dna.append(combo)

    # create aa sequences from dna
    mb_combo_library_aa = []
    for mb_var_dna in mb_combo_library_dna:
        mb_combo_library_aa.append(translate_sequence(mb_var_dna))
    parental_dna = mb_combo_library_dna[0]
    parental_aa = mb_combo_library_aa[0]

    # initialize dataframe containing all mb variants
    mutant_df = dict()
    dna_mutations = []
    aa_mutations = []
    no_dna_mutations = []
    no_aa_mutations = []

    for dna_var, aa_var in zip(mb_combo_library_dna, mb_combo_library_aa):
        dna_mutations.append(find_mutations(parental_dna, dna_var))
        aa_mutations.append(find_mutations(parental_aa, aa_var))
        no_dna_mutations.append(len(dna_mutations[-1]))
        no_aa_mutations.append(len(aa_mutations[-1]))

    mutant_df = {'dna_sequence': mb_combo_library_dna,
                 'aa_sequence': mb_combo_library_aa,
                 'dna_mutations': dna_mutations,
                 'aa_mutations': aa_mutations,
                 'number_dna_mutations': no_dna_mutations,
                 'number_aa_mutations': no_aa_mutations}
    mutations_core_df = pd.DataFrame(mutant_df)

    # add new columns and initialize with zeros
    for col in conc_bins_list:
        mutations_core_df[col] = 0

    for rep in ['rep1', 'rep2']:
#     '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/mutation_dfs/*-rep2_*.csv'
        mutation_dfs_to_parse = glob.glob(f"{mutation_df_dir}*-{rep}_*.csv")

        # for sample in
        mutations_core_filled_df = mutations_core_df.copy()
        for sample_name in mutation_dfs_to_parse:
            sample_info_list = sample_name.split('/')[-1].split('-')[:-1]
            conc_bin_temp = '_'.join(sample_info_list[1:]) + '_raw_counts'
            sample_df = pd.read_csv(sample_name).drop('Unnamed: 0', axis=1)[['dna_sequence', 'read_count']].rename(columns = {'read_count': conc_bin_temp})
            mutations_core_filled_df.update(mutations_core_filled_df[['dna_sequence']].merge(sample_df, on='dna_sequence', how='left')[conc_bin_temp])

        out_name = f"{output_name_prefix}_raw_{rep}.csv"
        mutations_core_filled_df.to_csv(out_name, index=False)

if __name__ == "__main__":
    main()
