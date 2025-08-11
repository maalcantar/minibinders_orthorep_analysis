#!/bin/bash

######################################################

# This script performs the following:
# i) FastQC - extracts quality metrics from raw fastq files
# ii) Fastp - performs quality filtering and trimming on raw reads
# iii) PEAR - merges quality filtered / trimmed forward and reverse reads
# iv) Seqkit - extracts general summary statistics for sequencing files
# v) Bowtie2 - performs alignment to a reference sequence
# outputs are written to ../../minibinders_orthorep_data/minibinders_orthorep_outputs directory

### PARAMETERS
## $1: metadata in .txt format (tab separated)
# headers should include:
# sample_number (e.g. 1); experiment_name (e.g. maa-002);
# sample_id (e.g. mb171-0nM-bin1-rep1_R1_001.fastq.gz) - should match .fastq.gz file names;
# binder_name (e.g., mb171); antigen_conc (e.g., 0); bin (e.g., 1);
# replicate (e.g., 1); barcode_f (e.g., gtca); barcode_r (e.g., gtca);
# flank_f (e.g., ACAAGAGAGAAGCTGACGCA); flank_r (e.g., TTCAGCAGCTTCAACTGCTT);
# reference (e.g., mb171); reference_size (e.g., 186)
## $2: list of fastq or fastq.gz files to be run through
# Only specify one of each read pair; the other
# filename is assumed (R2 if R1, or R1 if R2.)
# Example 1: sh 01_filter_merge_reads.sh
# ../../minibinders_orthorep_data/ngs_raw/maa_002/demultiplex/maa_002_demux_metadata.txt
# ../../minibinders_orthorep_data/ngs_raw/maa_002/demultiplex/*_R1_001.fastq.gz
#
### OUTPUTS
## FastQC
## $fastqc_out_path.html : fastqc report -- web link
## $fastqc_out_path.zip : fastqc report -- detailed files
#
## Fastp
## ${fastp_out_path}_out_R1.fastq.gz : quality filtered forward read
## ${fastp_out_path}_out_R2.fastq.gz  : quality filtered reverse read

## PEAR
## ${pear_out_path}_out_merged.assembled.fastq : merged fastq file that passed
# qc/qa filters

## Seqkit
## ${seqkit_stats_out_path}_filtering_stats.txt: statistics on number of reads
# containing barcode

## Bowtie2
## ${alignment_out_path}.bam: alignment file

## NOTES
# this script assumes the file name ends in R[12]_001.fastq.gz
# any string is allowed before R[12], as long as the forward and reverse read
# prefix is the same
# the min and max in PEAR is arbirary -- you should change this if needed

######################################################

# activate virtual enviroment
source activate minibinders

METADATA=$1
shift

# check to make sure files have the expected formats
for fastq in $@
do
	if [ -z $(basename $fastq | grep -i .fastq) ]
	then
		echo $(basename $fastq) "does not have .fastq suffix - aborting"
		exit 1
	fi

done
echo "all files are the correct format. continuing..."

# loop through all fastq files and extract file names and metadata required for
# writing outputs to correct folder
for fastq in "$@"
do
	fname=$(basename $fastq)
	dname=$(dirname $fastq)
	fpath=$dname/${fname%_R[12]_001.fastq*}
	experiment_name=$(cat $METADATA | grep $fname | cut -f2)
	reference_name=$(cat $METADATA | grep $fname | cut -f12)
	flank_f=$(cat $METADATA | grep $fname | cut -f10)
	flank_r=$(cat $METADATA | grep $fname | cut -f11)

  echo Experiment name $experiment_name
	fastqc_out_path=../../minibinders_orthorep_data/minibinders_orthorep_outputs/$experiment_name/fastqc_output/
  fastp_out_path=../../minibinders_orthorep_data/minibinders_orthorep_outputs/$experiment_name/fastp_output/${fname%_R[12]_001.fastq*}
  pear_out_path=../../minibinders_orthorep_data/minibinders_orthorep_outputs/$experiment_name/merged_reads/${fname%_R[12]_001.fastq*}
	seqkit_stats_out_path=../../minibinders_orthorep_data/minibinders_orthorep_outputs/$experiment_name/seqkit_stats/${fname%_R[12]_001.fastq*}
	seqkit_filtered_read_variants_out_path=../../minibinders_orthorep_data/minibinders_orthorep_outputs/$experiment_name/merged_variants_filtered_reads/${fname%_R[12]_001.fastq*}
	alignment_out_path=../../minibinders_orthorep_data/minibinders_orthorep_outputs/$experiment_name/alignments/${fname%_R[12]_001.fastq*}
	reference_path=../../minibinders_orthorep_data/ngs_raw/$experiment_name/references/${reference_name%.fa}

	# define prefix for all output directories
	output_dir_prefix=../../minibinders_orthorep_data/minibinders_orthorep_outputs/$experiment_name
	echo $output_dir_prefix
	if [ ! -e ${pear_out_path}_out_merged.assembled.fastq ]
	then

	  # create output directories if they don't exist already
		if [[ "$fastq" == "$1" ]]
			then

				# create general output folder for current experiment
				if [ ! -d "$output_dir_prefix" ]
			  then
					echo Making directory $output_dir_prefix
			    mkdir $output_dir_prefix
				fi

				# output directory for fastqc results
				if [ ! -d "$output_dir_prefix/fastqc_output/" ]
			  then
					echo Making directory $output_dir_prefix/fastqc_output/
			    mkdir $output_dir_prefix/fastqc_output/
				fi

				# output directory for fastp results
	      if [ ! -d "$output_dir_prefix/fastp_output/" ]
			  then
					echo Making directory $output_dir_prefix/fastp_output/
			    mkdir $output_dir_prefix/fastp_output/
				fi
				# output directory for fastp results
	      if [ ! -d "$output_dir_prefix/alignments/" ]
			  then
					echo Making directory $output_dir_prefix/alignments/
			    mkdir $output_dir_prefix/alignments/
				fi

				# output directory for merged reads
	      if [ ! -d "$output_dir_prefix/merged_reads/" ]
			  then
					echo Making directory $output_dir_prefix/merged_reads/
			    mkdir $output_dir_prefix/merged_reads/
				fi

				# output directory for merged reads
	      if [ ! -d "$output_dir_prefix/merged_variants_filtered_reads/" ]
			  then
					echo Making directory $output_dir_prefix/merged_variants_filtered_reads/
			    mkdir $output_dir_prefix/merged_variants_filtered_reads/
				fi

				# output directory for seqkit stats
	      if [ ! -d "$output_dir_prefix/seqkit_stats/" ]
			  then
					echo Making directory $output_dir_prefix/seqkit_stats/
			    mkdir $output_dir_prefix/seqkit_stats/
				fi
			fi

		 if [[ "$reference_name" == "mb171" ]]
		 then
			 echo "Minibinder is: mb171."
			 #expected amplicon is 203bp -- allowing for +/- 3bp error
			 min_merge_len=238
			 max_merge_len=258

	 elif [[ "$reference_name" == "mb376" ]]
	 then
		 echo "Minibinder is: mb376."
		 #expected amplicon is 193bp -- allowing for +/- 3bp error
		 min_merge_len=268
		 max_merge_len=288

	 else
		 echo $reference_name "Minibinder must be: mb171 or mb376."
		 exit 1
	 fi

	  # run fastqc on forward and reverse reads
	  fastqc ${fpath}_R1_001.fastq.gz -o $fastqc_out_path
	  fastqc ${fpath}_R2_001.fastq.gz -o $fastqc_out_path

	  # run fastp (with default settings) on forward and reverse reads
		# save quality report
	  fastp \
	      -i ${fpath}_R1_001.fastq.gz \
	      -I ${fpath}_R2_001.fastq.gz \
	      -o ${fastp_out_path}_out_R1.fastq.gz \
	      -O ${fastp_out_path}_out_R2.fastq.gz \
	      -j ${fastp_out_path}.json \
	      -h ${fastp_out_path}.html \
				-l 150 \
				-q 30 \
				-u 20
	#
	  # run pear (i.e., merge) on quality filtered forward and reverse reads
	  pear \
	      -f ${fastp_out_path}_out_R1.fastq.gz \
	      -r ${fastp_out_path}_out_R2.fastq.gz \
	      -n $min_merge_len \
				-m $max_merge_len \
	      -o ${pear_out_path}_out_merged \
				-j 7

		# remove additional pear output files that will not be used
	  rm ${pear_out_path}_out_merged.unassembled.forward.fastq
	  rm ${pear_out_path}_out_merged.unassembled.reverse.fastq
	  rm ${pear_out_path}_out_merged.discarded.fastq
	fi

	### FILTER FASTQ FILES USING SEQKIT
	echo Beginning filtering for $fname

	# extract reads that contain the desired barcode configuration
	# (e.g., attcNNNNNNNNNcAattcNNNNNNNNNcAATT for doubles)
	cat ${pear_out_path}_out_merged.assembled.fastq | seqkit stats > ${seqkit_stats_out_path}_filtering_stats.txt
	cat ${pear_out_path}_out_merged.assembled.fastq | seqkit amplicon -m 0 -P -s -F $flank_f -R $flank_r -r 21:-21 --bed > ${seqkit_filtered_read_variants_out_path}_variants.txt
	# rm ${pear_out_path}_out_merged.assembled.fastq
	bowtie2 \
	      --sensitive-local \
	      -x $reference_path \
	      -U ${pear_out_path}_out_merged.assembled.fastq \
	      --no-unal \
	      -p 7 | samtools view -bS - > ${alignment_out_path}.bam

			echo _____________new sample____________________________

done
