#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 16 08:21:33 2024
@author: alcantar
example run: python 00_demultiplex_files.py -i ../../minibinders_orthorep_data/ngs_raw/maa_002/fastq/*_R1_001.fastq.gz -m ../../minibinders_orthorep_data/ngs_raw/maa_002/fastq/maa_002_metadata.txt

we note that in our original naming convention:
mb-1==mb171
mb-4==mb376
"""
# activate virtual enviroment before running script
# conda activate minibinders

from Bio import SeqIO
import pandas as pd
import gzip
import glob
import sys
import os

from utils import *

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', help='List of fastq files to parse')
    parser.add_argument('-m', help='Path to metadata')

    args = parser.parse_args()

    fastq_R1_files = args.i
    metadata_path = args.m

    fastq_R2_files = [find_R2(fastq) for fastq in fastq_R1_files]
    metadata_csv = pd.read_csv(metadata_path, sep='\t', header=0)

    # this is a less sophisticated way to find R2 files
    # this method does not account for "_R1_" in the directory path
    # fastq_R2_files = [fastq.replace('_R1_', '_R2_') for fastq in fastq_R1_files]

    # initializing lists that will contain metadata
    sample_number = []
    experiment_name = []
    sample_id = []
    sample = []
    bc_r1 = []
    bc_r2 = []
    flank_f = []
    flank_r = []
    binder_name = []
    binder_size = []
    concentration = []
    bin = []
    replicate = []

    binder_name_to_size_dict = {'mb171': 186,
                                'mb376': 216}

    conc_name_to_conc_dict = {'0nM': 0,
                              '0pt01nM': 0.01,
                              '0pt1nM': 0.1,
                              '1nM': 1,
                              '10nM': 10,
                              '100nM': 100,
                              '500nM': 500}

    sample_number_cnt = 1 # initialize sample count

    for R1_reads_path, R2_reads_path in zip(fastq_R1_files, fastq_R2_files):

        R1_filename = R1_reads_path.split('/')[-1]
        R2_filename = R2_reads_path.split('/')[-1]

        print('---------------------------------------------------')
        print(f'Now running sample: {R1_filename}')
        print('---------------------------------------------------')

        metadata_tmp_df = metadata_csv.copy()[metadata_csv['filename']==R1_filename]

        sample_names = list(metadata_tmp_df['sample_name'])
        barcodes_f = list(metadata_tmp_df['barcode_f'])
        barcodes_r = list(metadata_tmp_df['barcode_r'])

        # append pseudo metadata to help with parsing unassigned reads:
        # i.e., reads without a valid barcode
        sample_names.append('unassigned')
        barcodes_f.append('nnnn')
        barcodes_r.append('nnnn')

        sample_to_bc_f_dict = dict(zip(sample_names,barcodes_f))
        sample_to_bc_r_dict = dict(zip(sample_names,barcodes_r))

        # concatenate forward and reverse barcode to help with read parsing
        # force barcodes to be lower-case so synchronize with read parsing below
        barcodes_full = [bc_f.lower() + bc_r.lower() \
                         for bc_f,bc_r in zip(barcodes_f, barcodes_r)]

        # initiate dictionaries to help with file parsing
        sample_name_to_bc_dict = dict() # link barcodes_full to sample name
        sample_entries_dict_f = dict() # links sample name / barcode with forward reads
        sample_entries_dict_r = dict() # links sample name / barcode with reverse reads
        for sample_name,barcode, barcode_f, barcode_r in zip(sample_names, barcodes_full, barcodes_f, barcodes_r):
            sample_name_to_bc_dict.update({barcode: sample_name})
            sample_entries_dict_f.update({sample_name: []})
            sample_entries_dict_r.update({sample_name: []})
        sample_entries_dict_f.update({'unassigned': []})
        sample_entries_dict_r.update({'unassigned': []})

        # loop through R1 and R2 simultaneously
        with gzip.open(R1_reads_path, "rt") as forward_reads_fastq:
            with gzip.open(R2_reads_path, "rt") as reverse_reads_fastq:
                forward_reads = SeqIO.parse(forward_reads_fastq, "fastq")
                reverse_reads = SeqIO.parse(reverse_reads_fastq, "fastq")

                print('Demultiplexing reads...')
                for forward_read, reverse_read in zip(forward_reads,reverse_reads):

                    # extract barcodes from R1 and R2
                    barcode_tmp = ''.join(forward_read.seq[0:4]) + ''.join(reverse_read.seq[0:4])
                    barcode_tmp = barcode_tmp.lower() # force lowercase with sync with barcodes_full

                    # find sample based on barcodes. if the barcode is not valid,
                    # mark as 'unassigned'
                    try:
                        sample_name_temp = sample_name_to_bc_dict[barcode_tmp]
                        sample_entries_dict_f[sample_name_temp].append(forward_read)
                        sample_entries_dict_r[sample_name_temp].append(reverse_read)
                    except:
                        sample_entries_dict_f['unassigned'].append(forward_read)
                        sample_entries_dict_r['unassigned'].append(reverse_read)

                # create output names and paths. this basically just appends
                # the sample name to the front of the original filenames
                for sample_name in sample_entries_dict_f:
                    new_sample_name_f = R1_filename.replace('-rep','-'+ sample_name+'-rep')#sample_name + '_' + R1_filename
                    new_sample_name_r = R2_filename.replace('-rep','-'+ sample_name+'-rep')#sample_name + '_' + R2_filename
                    if "unassigned" in new_sample_name_f:
                        out_dir = R1_reads_path.replace('fastq', 'demultiplex/unassigned')
                    else:
                        out_dir = R1_reads_path.replace('fastq', 'demultiplex')
                    out_dir = '/'.join(out_dir.split('/')[:-1])+ '/'

                    # the files can be written as uncompressed files if you
                    # remove the comment below and remove "gzip" a few lines below
                    out_path_f = f'{out_dir}{new_sample_name_f}'#.replace('fastq.gz','fastq')
                    out_path_r = f'{out_dir}{new_sample_name_r}'#.replace('fastq.gz','fastq')
                    make_dir(out_dir)

                    print(f'Creating fastq files for sample: {sample_name}')
                    with gzip.open(out_path_f, "wt") as handle:
                        SeqIO.write(sample_entries_dict_f[sample_name], handle, "fastq")
                    with gzip.open(out_path_r, "wt") as handle:
                        SeqIO.write(sample_entries_dict_r[sample_name], handle, "fastq")

                    if sample_name != 'unassigned':
                        experiment_name.append('')
                        sample_id.append(new_sample_name_f)
                        bc_r1.append(sample_to_bc_f_dict[sample_name])
                        bc_r2.append(sample_to_bc_r_dict[sample_name])
                        flank_f.append('')
                        flank_r.append('')
                        sample_number.append(sample_number_cnt)
                        sample_number_cnt += 1
                        sample_meta_from_name = new_sample_name_f.split('-')[0:4]
                        rep_info = sample_meta_from_name[-1].split('_')[0]
                        binder_name.append(sample_meta_from_name[0])
                        binder_size.append(int(binder_name_to_size_dict[sample_meta_from_name[0]]))
                        concentration.append(conc_name_to_conc_dict[sample_meta_from_name[1]])
                        bin.append(int(sample_meta_from_name[2][-1]))
                        replicate.append(int(rep_info[-1]))



    # create dataframe for new metadata
    # to add: bin | minibinder based on file name
    new_metadata_df = pd.DataFrame(
    {'sample_number': sample_number,
     'experiment_name': experiment_name,
     'sample_id': sample_id,
     'binder_name': binder_name,
     'antigen_concentration_nM': concentration,
     'bin': bin,
     'replicate': replicate,
     'barcode_f': bc_r1,
     'barcode_r': bc_r2,
     'flank_f': flank_f,
     'flank_r': flank_r,
     'reference': binder_name,
     'reference_size': binder_size
    })

    new_metadata_path = out_dir.split('demultiplex/unassigned')[0] + 'demultiplex/new_metadata.txt'
    new_metadata_df.to_csv(new_metadata_path, sep='\t', index=False)

if __name__ == "__main__":
    main()
