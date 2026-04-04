#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 14 16:14:47 2026
@author: alcantar
example run: python 14_esm3_scores.py
"""
# virtual environment: minibinders_orthorep_esm3
# requires huggingface token

import itertools

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
import pickle

from utils import *

def define_token_dicts():

    """
    Create a mapping between ESM tokens and amino acids

    PARAMETERS:
    -------------
    NONE

    RETURNS:
    -------------
    idx_to_token_dict: dict
        index to amino acid/token
    token_to_idx_dict: dict
        amino acid/token to index
    """

    # create dictionaries mapping index in logits output to token (including 20 amino acids)
    Tokenizer = EsmSequenceTokenizer()
    idx_to_token_dict = dict() # index to token / amino acid
    token_to_idx_dict = dict() # index / amino acid to token

    # there are 64 tokens, althought only the first 33 have meaning (e.g., amino acid, mask, special token)
    # this code was adapted from the following: https://github.com/evolutionaryscale/esm/issues/252
    for i in range(64):
        idx_to_token_dict.update({i:Tokenizer.decode(i)})
        token_to_idx_dict.update({Tokenizer.decode(i): i}) # redundant empty tokens ('') will cause this to collapse empty token to one key

    return (idx_to_token_dict, token_to_idx_dict)

def parse_mutation(mut):

    """
    Parse a mutation (e.g., A14V)
    i) original residue (A)
    ii) index (13; residue number - 14)
    iii) new residue (V)

    PARAMETERS:
    -------------
    mut: str
        mutation in the form XNY, where X is the original residue, N is the
        residue number, and Y is the new residue

    RETURNS:
    -------------
    orig: str
        original residue
    idx: int
        mutated residue index
    new: str
        new residue
    """

    orig = mut[0]
    new = mut[-1]
    idx = int(mut[1:-1]) - 1

    return(orig, idx, new)

def validate_mutation(orig, idx, new, sequence, token_to_idx_dict):

    '''
    Check whether a given mutation is valid. This means:
    i) The mutated residue is a valid token
    ii) The original residue in 'orig' matches the residue in the
    given protein sequence

    PARAMETERS:
    -------------
    orig: str
        original residue
    idx: int
        mutated residue index
    new: str
        new residue
    sequence: str
        original (non-mutated) protein sequence
    token_to_idx_dict: dict
        dictionary mapping token to index

    RETURNS:
    -------------
        NONE. Just raises errors if invalid token provided as the mutated residue
        or if the provided original residue doesn't match the given protein sequence.
    '''

    if new not in token_to_idx_dict:
        raise ValueError(f"Invalid amino acid: {new}")

    if sequence[idx] != orig:
        raise KeyError(
            f"Mutation mismatch at position {idx+1}: "
            f"expected {orig}, found {sequence[idx]}"
        )

def mask_sequence(sequence, indices):

    '''
    applies mask tokens at given indices.

    PARAMETERS:
    -------------
    sequence: str
        protein sequence
    indices: list of int
        indicies where mask token will be applied

    RETURNS:
    -------------
    seq_mask: str
        protein sequence with mask token applied at specified residues

    '''
    seq = list(sequence)
    for idx in indices:
        seq[idx] = "<mask>"
    seq_mask = "".join(seq)

    return seq_mask

def delta_logp_for_position(logits, orig, new, idx, token_to_idx_dict):

    '''
    Compute delta residue-wise delta logP between wild type and mutated residue.

    PARAMETERS:
    -------------
    logits: tensor
        logits provided by esm model
    orig: str
        original residue
    idx: int
        mutated residue index
    new: str
        new residue
    token_to_idx_dict: dict
        dictionary mapping token to index

    RETURNS:
    -------------
        deltaP: float
            difference in likelihood between mutant and wild type
    '''
    log_probs = torch.log_softmax(logits[idx], dim=-1)

    deltaP = (
        log_probs[token_to_idx_dict[new]]
        - log_probs[token_to_idx_dict[orig]]
    ).item()

    return deltaP

def calc_deltaP(protein_seq, muts_list, token_to_idx_dict, client):

    '''
    Calculate likelihoods for a given set of mutations.
    This includes single mutations and higher order mutations

    PARAMETERS:
    -------------
    protein_seq: str
        original protein sequence
    muts_list: list of str
        list of mutations to consider

    RETURNS:
    -------------
    mut_to_deltaP: dict
        mapping of a given set of mutations to predicted total delta P

    '''

    # parse mutations into original residue, new residue, and residue number
    # also, make sure mutation is valid (original matches sequences and new residue
    # is a valid token
    parsed_muts = []
    for mut in muts_list:
        orig, idx, new = parse_mutation(mut)
        validate_mutation(orig, idx, new, protein_seq, token_to_idx_dict)
        parsed_muts.append((orig, idx, new))

    # initialize dictionary which will contain all summed delta P values
    mut_to_deltaP = {}

    # generate all possibe combinations of mutations
    for r in range(1, len(parsed_muts) + 1):
        for mut_set in itertools.combinations(parsed_muts, r):

            origs, idxs, news = zip(*mut_set)

            masked_seq = mask_sequence(protein_seq, idxs)
            protein_masked = ESMProtein(sequence=masked_seq)

            protein_tensor = client.encode(protein_masked)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True)
            )

            logits = logits_output.logits.sequence[0]

            deltaPs = [
                delta_logp_for_position(logits, orig, new, idx, token_to_idx_dict)
                for orig, idx, new in mut_set
            ]

            mut_labels = tuple(
                f"{orig}{idx+1}{new}" for orig, idx, new in mut_set
            )

            mut_to_deltaP['_'.join(mut_labels)] = sum(deltaPs)

            print('_'.join(mut_labels), mut_to_deltaP['_'.join(mut_labels)])

    return(mut_to_deltaP)

def main():

    idx_to_token_dict, token_to_idx_dict = define_token_dicts()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = ESMC.from_pretrained("esmc_300m").to(device)

    output_dir = '../data/minibinders_orthorep_esm3/'
    make_dir(output_dir)

    mb_1_seq = 'CCLTFENPEFAEQAVLIAEDAGLKVEVKPGEDSLTVCLSNEAACKYFAERAKEMGVPVTVEP'
    mb_1_muts_list = ['N7D', 'F10Y', 'A14V', 'P29S', 'K45E', 'K52E']
    mb_1_deltaP_scores = calc_deltaP(protein_seq=mb_1_seq, muts_list = mb_1_muts_list, token_to_idx_dict=token_to_idx_dict,client=client)
    with open(f'{output_dir}mb1_esm3_scores.pickle', 'wb') as handle:
         pickle.dump(mb_1_deltaP_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('')

    mb_2_seq = 'SAEDELWELLDECRAAYFEGKFKEAKKCLKKVLELAKKNGNKAFEEAAKALLKAVEAAE'
    mb_2_muts_list = ['L29S', 'V32I', 'E46K', 'E56A', 'A57T']
    mb_2_deltaP_scores = calc_deltaP(protein_seq=mb_2_seq, muts_list = mb_2_muts_list, token_to_idx_dict=token_to_idx_dict,client=client)
    with open(f'{output_dir}mb2_esm3_scores.pickle', 'wb') as handle:
         pickle.dump(mb_2_deltaP_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('')

    mb_3_seq = 'SMFEEMVEKMIEYLKACKESDYDKADDLYGEIIDLADEAAKGNSEKARELIEKAYEEAKKRYE'
    mb_3_muts_list = ['M10I', 'D26G', 'K41E']
    mb_3_deltaP_scores = calc_deltaP(protein_seq=mb_3_seq, muts_list = mb_3_muts_list, token_to_idx_dict=token_to_idx_dict,client=client)
    with open(f'{output_dir}mb3_esm3_scores.pickle', 'wb') as handle:
         pickle.dump(mb_3_deltaP_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('')

    mb_4_seq = 'SLGYYKVTFLPDAHPQAVEILALAFLDNGLEVKEVVTEEGNKYVIAELDEITLEAVKEAIGEIIESVEPVVE'
    mb_4_muts_list = ['A13T', 'A24V', 'F25L', 'G40D']
    mb_4_deltaP_scores = calc_deltaP(protein_seq=mb_4_seq, muts_list = mb_4_muts_list, token_to_idx_dict=token_to_idx_dict,client=client)
    with open(f'{output_dir}mb4_esm3_scores.pickle', 'wb') as handle:
         pickle.dump(mb_4_deltaP_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('')

if __name__ == "__main__":
    main()
