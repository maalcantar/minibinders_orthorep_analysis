#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on March 14 16:14:47 2026
@author: alcantar
example run 1: python 15_rosetta_mutation_stability.py -i monomer
example run 2: python 15_rosetta_mutation_stability.py -i complex
"""
# virtual environment: minibinders_orthorep_rosetta

import tqdm
import pickle
import itertools

import pyrosetta
from pyrosetta.toolbox import pose_from_rcsb
from pyrosetta.toolbox import cleanATOM
from Bio.PDB import MMCIFParser, PDBIO
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.toolbox import mutants
from pyrosetta import *

import argparse

from utils import *

init("-mute all") # silence verbose outputs from fastrelax

def fast_relax_iterations(initial_pose, n_iter, fastrelax_protocol, scorefxn):

    '''
    Creates a given number of fastrelax models (n=n_iter) and then returns the lowest energy model

    PARAMETERS
    -----------
    initial_pose: pyrosetta pose
        pose that will undergo relaxation
    n_iter: int
        number of fastrelax models to test
    fastrelax_protocol: pyrosetta fastrelax
        fastrelax protocol, which mainly just needs to include the score function
    scorefxn: pyrosetta score function
        pyrosetta score function

    RETURNS
    --------
    final_pose: pyrosetta pose
        lowest energy pose from all tested
    scores: list of float
        scores for all models tested

    '''
    min_score = 0 # initialize score as 0. Technically would be better to have -np.inf but I don't expect any pose to be above 0
    scores = [] # initialize empty list to store all scores
    print('Finding best fastrelaxed model')
    for i in tqdm.tqdm(range(n_iter)):
        pose_temp = initial_pose.clone()
        fastrelax_protocol.apply(pose_temp)
        score_tmp = scorefxn(pose_temp)
        scores.append(score_tmp)
        if score_tmp < min_score:
            min_score = score_tmp
            final_pose = pose_temp.clone()

    print(f'Best model score is: {scorefxn(final_pose)}')

    return(final_pose, scores)


def apply_mutations(base_pose, mutation_list, offset,
                    scorefxn, fastrelax, include_relax=False):
    '''

    applies mutations to chosen positions. either does side chain repacking (if include_relax=False)
    or fastrelaxes the mutated model. include_relax probably better for more disruptive mutations.
    for examples, I've found that prolines lead to massive increases in enegy if you only allow repacking

    PARAMETERS
    -----------
    base_pose: pyrosetta pose
        pose that will undergo mutation and or relaxation
    mutation_list: list of str
        list of mutations to apply in the form AXY
        (A=original residue; X=residue to change; 1-indexed; Y=new residue)
    offset: int
        offset to apply to residue numbering if minibinder is not chain A.
        If it is chain A, then offset should be 0.
    scorefxn: pyrosetta score function
        score function to optimize
    include_relax: bool
        indicate whether to fastrelax mutated pose (if True) or only implement
        side chain repacking (if False)

    RETURNS
    --------
    new_pose: pyrosetta pose
        mutated pose
    wt_pose_repack: pyrosetta pose
        wild type pose

    '''

    new_pose = base_pose.clone()
    wt_pose_repack = base_pose.clone()
    for mut in mutation_list:
        old_res = mut[0]
        new_res = mut[-1]
        res_num = int(mut[1:-1]) + offset # apply offset in case minibinder is not part of chain A

        # Validate residue identity
        pose_res = new_pose.residue(res_num).name1()

        if pose_res != old_res:
            raise ValueError(
                f"Mismatch at position {int(mut[1:-1])} "
                f"(pose position {res_num}): expected {old_res}, found {pose_res}.")

        if include_relax:
            # apply mutate_residue to both the wild type and mutated variant, where the wild type just gets
            # changed back to the original residue.

            mutants.mutate_residue(new_pose, res_num, new_res)
            mutants.mutate_residue(wt_pose_repack, res_num, old_res)

        else:
            # repack residue within 12 Angstroms of mutated residue
            mutants.mutate_residue(new_pose, res_num, new_res,
                                 pack_radius=12.0, pack_scorefxn=scorefxn)
            mutants.mutate_residue(wt_pose_repack, res_num, old_res,
                                 pack_radius=12.0, pack_scorefxn=scorefxn)
    if include_relax:
        fastrelax.apply(new_pose) # apply fastrelax on the new pose
        fastrelax.apply(wt_pose_repack) # apply fastrelax on the old pose

    return(new_pose, wt_pose_repack)

def ddG_calc(path_to_af_model, pdb_output_name,
                     muts_list, chain, n_repacks=100, n_relaxs=10,
                     score_eqn='ref2015', constrain_bb=True,
                     fastrelax_max_iter=200, include_relax=True):

    '''
    Calculate deltadeltaG of protein structure after introducing mutations

    PARAMETERS
    -----------
    path_to_af_model: str
        path to af model to be used as base pose
    pdb_output_name: str
        path of af model after conversion to pdb format
    muts_list: list of str
        list of mutations to test (both individually and in combination)
    chain: str
        indicate which string contain the minibinder. For the monomer,
        this should be 'A'. For the complex, this should be 'B'
    n_repacks: int
        number of trials to score mutation. the repacking / fastrelax
        process is stochastic, so this is meant to help calculate a median
        deltadeltaG
    n_relaxs: int
        number of fastrelax models of the base pose to test.
        the lowest energy pose is then used
    score_eqn: str
        score function to use. highly recommend using the default ('ref2015')
    constrain_bb: bool
        indicate whether to constrain backbone coordinates. highly recommend
        keeping this as True because the AlphaFold models tend to be highly accurate
        when predicting minibinder structure.
    fastrelax_max_iter: int
        max number of fastrelax interations
    include_relax: bool
        indicate whether to fastrelax mutated pose (if True) or only implement
        side chain repacking (if False)

    RETURNS
    --------
    mut_ddG_dict: dictionary of list of floats
        dictionary mapping each mutation to deltadeltaG scores.
    parent_dG_dict: dictionary of list of floats
        dictionary mapping each mutation to a parent deltaG
    mutant_dG_dict: dictionary of list of floats
        dictionary mapping each mutation to a mutant deltaG
    '''

    # Load structure
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("af3", path_to_af_model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_output_name)

    # Load pose and score function
    pose = pose_from_pdb(pdb_output_name)
    scorefxn = create_score_function(score_eqn)

    # FastRelax on wild-type
    fastrelax = FastRelax(scorefxn)
    if constrain_bb:
        fastrelax.constrain_relax_to_start_coords(constrain_bb)
    fastrelax.max_iter(fastrelax_max_iter)
    pose, scores_list = fast_relax_iterations(pose, n_relaxs, fastrelax, scorefxn)

    # Score the relaxed wild-type once
    wt_score = scorefxn(pose)

    # Generate all non-redundant combinations
    mut_ddG_dict = dict()
    parent_dG_dict = dict()
    mutant_dG_dict = dict()

    # calculate offset needed to mutate proper residue
    # offset will be 0 if chain minibinder chain is 'A'. That's taken
    # care of with the else.
    chain_id = ord(chain) - ord('A') + 1
    if chain_id > 1:
        offset = pose.chain_end(chain_id - 1)
    else:
        offset = 0

    for r in range(1, len(muts_list) + 1):
        for mut_combo in tqdm.tqdm((itertools.combinations(muts_list, r))):
            print(f"Analyzing mutation: {mut_combo}")
            ddG_scores = []
            parent_dG_scores = []
            mutant_dG_scores = []
            for i in tqdm.tqdm((range(n_repacks))):
                # Apply mutations to relaxed WT
                mut_pose, wt_pose_repack = apply_mutations(pose, mut_combo, offset, scorefxn,
                                                           fastrelax, include_relax=include_relax)
                mutant_score = scorefxn(mut_pose)
                wt_score_repack = scorefxn(wt_pose_repack)

                # ΔΔG = mutant - WT
                ddG_scores.append(mutant_score - wt_score_repack)
                parent_dG_scores.append(wt_score_repack)
                mutant_dG_scores.append(mutant_score)


            mut_ddG_dict['_'.join(list(mut_combo))] = ddG_scores
            parent_dG_dict['_'.join(list(mut_combo))] = parent_dG_scores
            mutant_dG_dict['_'.join(list(mut_combo))] = mutant_dG_scores

    print(f"Calculated ΔΔG for {len(mut_ddG_dict)} mutation combinations")

    return(mut_ddG_dict, parent_dG_dict, mutant_dG_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='monomer or complex')
    args = parser.parse_args()

    structure_type=args.i

    n_repacks=1 # number of fastrelax iterations per mutant
    n_relaxs=1 # number of fastrelax iterations to find initial model
    outdir_prefix = '../data/rosetta_analysis/'
    if structure_type=='monomer':
        make_dir(f"{outdir_prefix}monomer_results/")
        mb1_af_model = f"{outdir_prefix}monomer_models/fold_2025_03_18_mb171_parent_only_model_0.cif"
        mb1_pdb_output = f"{outdir_prefix}monomer_models/mb171_parent_only_model_0.pdb"
        mb1_muts_list =  ['N7D', 'F10Y', 'A14V', 'P29S', 'K45E', 'K52E']

        mb1_ddG_scores_dict, mb1_parent_dG_scores_dict, mb1_mutant_dG_scores_dict = ddG_calc(path_to_af_model=mb1_af_model,
                              pdb_output_name=mb1_pdb_output,
                              muts_list=mb1_muts_list,
                              chain='A',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)

        with open(f"{outdir_prefix}monomer_results/mb1_ddG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb1_ddG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb1_parent_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb1_parent_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb1_mutant_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb1_mutant_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mb2_af_model = f"{outdir_prefix}monomer_models/fold_2025_03_18_mb317_parent_only_model_0.cif"
        mb2_pdb_output = f"{outdir_prefix}monomer_models/mb317_parent_only_model_0.pdb"
        mb2_muts_list =  ['L29S', 'V32I', 'E46K', 'E56A', 'A57T']
        mb2_ddG_scores_dict, mb2_parent_dG_scores_dict, mb2_mutant_dG_scores_dict = ddG_calc(path_to_af_model=mb2_af_model,
                              pdb_output_name=mb2_pdb_output,
                              muts_list=mb2_muts_list,
                              chain='A',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)
        with open(f"{outdir_prefix}monomer_results/mb2_ddG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb2_ddG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb2_parent_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb2_parent_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb2_mutant_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb2_mutant_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mb3_af_model = f"{outdir_prefix}monomer_models/fold_2025_03_18_mb340_parent_only_model_0.cif"
        mb3_pdb_output = f"{outdir_prefix}monomer_models/mb340_parent_only_model_0.pdb"
        mb3_muts_list =  ['M10I', 'D26G', 'K41E']
        mb3_ddG_scores_dict, mb3_parent_dG_scores_dict, mb3_mutant_dG_scores_dict = ddG_calc(path_to_af_model=mb3_af_model,
                              pdb_output_name=mb3_pdb_output,
                              muts_list=mb3_muts_list,
                              chain='A',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)
        with open(f"{outdir_prefix}monomer_results/mb3_ddG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb3_ddG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb3_parent_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb3_parent_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb3_mutant_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb3_mutant_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mb4_af_model = f"{outdir_prefix}monomer_models/fold_2025_03_18_mb376_parent_only_model_0.cif"
        mb4_pdb_output = f"{outdir_prefix}monomer_models/mb376_parent_only_model_0.pdb"
        mb4_muts_list =  ['A13T', 'A24V', 'F25L', 'G40D']
        mb4_ddG_scores_dict, mb4_parent_dG_scores_dict, mb4_mutant_dG_scores_dict = ddG_calc(path_to_af_model=mb4_af_model,
                              pdb_output_name=mb4_pdb_output,
                              muts_list=mb4_muts_list,
                              chain='A',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)
        with open(f"{outdir_prefix}monomer_results/mb4_ddG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb4_ddG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb4_parent_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb4_parent_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}monomer_results/mb4_mutant_dG_scores_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb4_mutant_dG_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif structure_type=='complex':
        make_dir(f"{outdir_prefix}complex_results/")
        mb1_complex_af_model = f"{outdir_prefix}complex_models/fold_2025_03_18_huil7ra_mb171_parent_model_0.cif"
        mb1_complex_pdb_output = f"{outdir_prefix}complex_models/huil7ra_mb171_parent_model_0.pdb"
        mb1_muts_list =  ['N7D', 'F10Y', 'A14V', 'P29S', 'K45E', 'K52E']
        mb1_ddG_scores_complex_dict, mb1_parent_dG_scores_complex_dict, mb1_mutant_dG_scores_complex_dict = ddG_calc(path_to_af_model=mb1_complex_af_model,
                              pdb_output_name=mb1_complex_pdb_output,
                              muts_list=mb1_muts_list,
                              chain='B',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)

        with open(f"{outdir_prefix}complex_results/mb1_ddG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb1_ddG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb1_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb1_parent_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb1_mutant_dG_scores_complex_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb1_mutant_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        mb2_complex_af_model =  f"{outdir_prefix}complex_models/fold_2025_03_18_huil7ra_mb317_parent_model_0.cif"
        mb2_complex_pdb_output =  f"{outdir_prefix}complex_models/huil7ra_mb317_parent_model_0.pdb"
        mb2_muts_list =  ['L29S', 'V32I', 'E46K', 'E56A', 'A57T']
        mb2_ddG_scores_complex_dict, mb2_parent_dG_scores_complex_dict, mb2_mutant_dG_scores_complex_dict = ddG_calc(path_to_af_model=mb2_complex_af_model,
                              pdb_output_name=mb2_complex_pdb_output,
                              muts_list=mb2_muts_list,
                              chain='B',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)

        with open(f"{outdir_prefix}complex_results/mb2_ddG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb2_ddG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb2_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb2_parent_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb2_mutant_dG_scores_complex_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb2_mutant_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        mb3_complex_af_model = f"{outdir_prefix}complex_models/fold_2025_03_18_huil7ra_mb340_parent_model_0.cif"
        mb3_complex_pdb_output = f"{outdir_prefix}complex_models/huil7ra_mb340_parent_model_0.pdb"
        mb3_muts_list =  ['M10I', 'D26G', 'K41E']
        mb3_ddG_scores_complex_dict, mb3_parent_dG_scores_complex_dict, mb3_mutant_dG_scores_complex_dict = ddG_calc(path_to_af_model=mb3_complex_af_model,
                              pdb_output_name=mb3_complex_pdb_output,
                              muts_list=mb3_muts_list,
                              chain='B',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)

        with open(f"{outdir_prefix}complex_results/mb3_ddG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb3_ddG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb3_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb3_parent_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb3_mutant_dG_scores_complex_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb3_mutant_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mb4_complex_af_model = f"{outdir_prefix}complex_models/fold_2025_03_18_huil7ra_mb376_parent_model_0.cif"
        mb4_complex_pdb_output = f"{outdir_prefix}complex_models/huil7ra_mb376_parent_model_0.pdb"
        mb4_muts_list =  ['A13T', 'A24V', 'F25L', 'G40D']
        mb4_ddG_scores_complex_dict, mb4_parent_dG_scores_complex_dict, mb4_mutant_dG_scores_complex_dict = ddG_calc(path_to_af_model=mb4_complex_af_model,
                              pdb_output_name=mb4_complex_pdb_output,
                              muts_list=mb4_muts_list,
                              chain='B',
                              n_repacks=n_repacks,
                              n_relaxs=n_relaxs,
                              score_eqn='ref2015',
                              constrain_bb=True,
                              fastrelax_max_iter=200,
                              include_relax=True)

        with open(f"{outdir_prefix}complex_results/mb4_ddG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb4_ddG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb4_parent_dG_scores_complex_dict_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb4_parent_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{outdir_prefix}complex_results/mb4_mutant_dG_scores_complex_with_fastrelax_v2.pickle", 'wb') as handle:
            pickle.dump(mb4_mutant_dG_scores_complex_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Please specify monomer or complex.')

if __name__ == "__main__":
    main()
