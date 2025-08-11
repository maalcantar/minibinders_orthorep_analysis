#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 22 12:00:57 2025
@author: alcantar
example run: python 07_model_fitting.py -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/tite-seq_Kds/mb171_EC50_df.csv
"""
# activate virtual enviroment before running script
# source activate minibinders

import numpy as np
import csv
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from utils import *

from itertools import combinations
import scipy.stats as stats
import pandas as pd

import argparse

def encode_mutations(mut_df):

    """
    converts a genotype from binary format to categorical format. for example,
    the genotypoes '000'and '101' would return [0,0,0] and [1,0,1] where the
    elements of the list are dtype=int

    PARAMETERS
    -----------
    mut_df: pandas dataframe
        a dataframe containing a column named 'variant_id' which
        contains genotypes in binary string format

    RETURNS
    -----------
    categorical_encodings: list of in
        categorical encoding of a genotype

    """

    categorical_encodings = [[int(mutation) for mutation in variant_id] for variant_id in mut_df['variant_id']]
#     categorical_encodings = [list(variant_id_str) for variant_id_str in mut_df['variant_id']]
#     categorical_encodings = [[int(x) for x in sublist] for sublist in categorical_encodings]

    return(categorical_encodings)


def generate_combinations(input_list, order):

    """
    creates all possible (n Choose order) orders of a list. this is useful for assigning
    weights to their cognate mutation(s)

    PARAMETERS
    -----------
    input_list: list of str
        list of mutations (e.g., ['A14V', 'P29S'])
    order: int
        max order to combinations (e.g., N choose Order)

    RETURNS
    -----------
    mut_combos: list of str
        list of mutation combinations
    """

    # Start with a list containing just the number 1
    mut_combos = ['coef_0']

    # Loop through all combination orders from 1 to the specified order
    for i in range(1, order + 1):
        # Generate combinations of the current order and add them to the result
        mut_combos.extend('_'.join(combo) for combo in combinations(input_list, i))

    return mut_combos

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Path to csv with EC50/Kd values')

    args = parser.parse_args()

    EC50_path=args.i#'../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/tite-seq_Kds/mb171_EC50_df.csv'

    split_path_list = EC50_path.split('/')
    mbname = split_path_list[-1].split('_')[0]
    output_dir = '/'.join(EC50_path.split('/')[:-1]).replace('tite-seq_Kds', 'model_results') + '/'
    make_dir(output_dir)

    EC50_df = pd.read_csv(EC50_path, index_col=0, dtype={"variant_id": str})



    muts = extract_mutations_from_string(list(EC50_df['aa_mutations'])[-1])
    muts = [mut.replace(" ","") for mut in muts]


    X_df = EC50_df[['variant_id']]
    y_df = EC50_df['EC50_mean']
    y_df = -np.log10(y_df)
    test_size=0.2

    alpha=0.0001

    for order in range(1,4):
        print(f"Testing / training models of order {str(order)}")
        print("==================================================")

        results_current_dict = dict()
        interaction_terms_current = generate_combinations(muts, order)
        # random_state=42
        pearson_corr_train_list = []
        pearson_corr_test_list = []

        r2_score_train_list = []
        r2_score_test_list = []

        coefficient_list = []

        outpath_all = f"{output_dir}{mbname}_all_results_order{str(order)}.csv"
        outpath_mean = f"{output_dir}{mbname}_mean_results_order{str(order)}.csv"

        n_trials = 100
        for trial in range(0,n_trials):

            X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                                y_df,
                                                                test_size=test_size,
                                                                random_state=trial)

            X_cat_train = encode_mutations(X_train)
            X_cat_test = encode_mutations(X_test)

            model_current = linear_model.Ridge(alpha=alpha,
                                               solver='lsqr',
                                               fit_intercept=False)

            poly_features_transformer = PolynomialFeatures(order,
                                                           interaction_only=True)
            X_train_current = poly_features_transformer.fit_transform(X_cat_train)
            X_test_current = poly_features_transformer.fit_transform(X_cat_test)
            model_current.fit(X_train_current, y_train)


            model_coefs_current  = model_current.coef_

            preds_train_current = model_current.predict(X_train_current)
            preds_test_current = model_current.predict(X_test_current)

            pearsonR_corr_train, pearsonR_pval_train = stats.pearsonr(y_train, preds_train_current)
            pearsonR_corr_test, pearsonR_pval_test = stats.pearsonr(y_test, preds_test_current)

            R2_score_train = r2_score(y_train, preds_train_current)
            R2_score_test = r2_score(y_test, preds_test_current)

            pearson_corr_train_list.append(pearsonR_corr_train)
            pearson_corr_test_list.append(pearsonR_corr_test)

            r2_score_train_list.append(R2_score_train)
            r2_score_test_list.append(R2_score_test)
            coefficient_list.append(list(model_coefs_current))

            if order == 1 and trial ==0:
                linear_model_to_store = model_current

        results_current_dict.update({'pearsonR_train': pearson_corr_train_list,
                                    'pearsonR_test': pearson_corr_test_list,
                                    'R2_train': r2_score_train_list,
                                    'R2_test': r2_score_test_list})
        results_df = pd.DataFrame(results_current_dict)

        for idx, interaction_term in enumerate(interaction_terms_current):
            results_df[interaction_term] = [coeff_tmp[idx] for coeff_tmp in coefficient_list]

        mean_df = results_df.mean().to_frame().T

        # Ensure the column names are preserved
        mean_df.columns = results_df.columns

        results_df.to_csv(outpath_all, index=False)
        mean_df.to_csv(outpath_mean, index=False)

# plot linear model results

    X = encode_mutations(X_df)
    poly_features_transformer = PolynomialFeatures(1, interaction_only=True)
    X = poly_features_transformer.fit_transform(X)
    lin_predictions = linear_model_to_store.predict(X)
    model_example_results_df = pd.DataFrame()
    model_example_results_df['measurements'] = list(y_df)
    model_example_results_df['predictions'] = list(lin_predictions)
    output_csv_path = '../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/model_results/linear_model_predictions.csv'
    model_example_results_df.to_csv(output_csv_path)
    print(r2_score(model_example_results_df['measurements'], model_example_results_df['predictions']))
    print(stats.pearsonr(model_example_results_df['measurements'], model_example_results_df['predictions'])[0])
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_df, lin_predictions,
               marker='.', s=200, color='black', alpha=0.8)
    ax.set_xlabel('Measured -log(EC50)', size=16)
    ax.set_ylabel('Predicted -log(EC50)', size=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.savefig('../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/model_results/linear_model_predictions.pdf', dpi=400)

if __name__ == "__main__":
    main()
