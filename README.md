# Mapping the evolution of computationally designed protein binders

## Introduction 

This repository contains all code needed to reproduce DNA sequencing data processing and analyses described in:
> Miguel A. Alcantar, Alexandra M. Paulk, Shoeib Moradi, Debjani Bhar, Grant L. J. Keller, Tanmoy Sanyal, Hua Bai, Gamze Camdere, Seog Joon Han, Mani Jain, Brandon Jew, Sezen Vatansever Inak, Christopher J. Langmead, Christine E. Tinberg, Irwin Chen, Chang C. Liu. “Mapping the evolution of computationally designed protein binders”. Submitted. bioRxiv DOI: https://doi.org/10.1101/2025.10.04.680454.

Code author: Miguel A. Alcantar.
# Installation & requirements 

This repository, including all code needed to reproduce analyses, can be installed using:

~~~
git clone https://github.com/maalcantar/minibinders_orthorep_analysis.git
cd minibinders_orthorep_analysis
pip install -r requirements.txt
~~~

R library requirements:
* DNABarcodes v1.32.0 #(https://bioconductor.org/packages/release/bioc/html/DNABarcodes.html)

Additional requirements: 
* fastp v0.12.4 #(https://github.com/OpenGene/fastp)
* seqkit v2.3.1 #(https://bioinf.shenwei.me/seqkit/)
* PEAR v0.9.11 #(https://cme.h-its.org/exelixis/web/software/pear/doc.html)
* Bowtie2 v2.4.4 #(https://bowtie-bio.sourceforge.net/bowtie2/index.shtml)


# Directory structure

### source code

All code is in  <code>src/</code>, which contains a combination of bash, python, and R scripts. The numbering at the beginning of each file name indicates the order in which that script should be run.

### data
Due to the large size of output files related to sequencing data processing, these outputs are generally written to a separate directory: minibinders_orthorep_data/minibinders_orthorep_outputs/. This directory is outside of this repository. All other outputs are written to this repository in the  <code>data/</code> repository.


Raw sequencing data will be made publicly available under NCBI Bioproject PRJNA1304565.

Please feel free to reach out if you have any questions about implementation or reproducing analyses! (malcant4 [at] uci [dot] edu).
