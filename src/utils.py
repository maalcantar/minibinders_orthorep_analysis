import os
import sys

def reverse_complement(seq):

    '''
    creates the reverese complement on a sequence

    PARAMETERS
    -----------
    seq: str
        string consisting of the four nucleotides: a,t,c,g

    RETURNS
    ----------
    seq_rcomp: str
        reverse complement of the given string

    '''

    base_pair_dict = {'a':'t',
                     'c': 'g',
                     'g': 'c',
                     't': 'a',
                     'n':'n'}

    seq_rcomp = [base_pair_dict[nt] for nt in seq[::-1]]
    seq_rcomp = ''.join(seq_rcomp)
    return(seq_rcomp)

def make_dir(dir_path):

    '''
    check if a directory exists. if it does not, create the directory.

    PARAMETERS
    --------------------
    dir_path: string
        path to directory

    RETURNS
    --------------------
    none

    '''

    CHECK_FOLDER = os.path.isdir(dir_path)
    if not CHECK_FOLDER:
        os.makedirs(dir_path)
        print(f"created folder : {dir_path}")

def find_R1(R2_path):

    '''
    obtains the corresponding R1 file given a path to R2
    this functions performs the following two checks:
    i) checks if "_R1_" shows up in the file name
    ii) checks if the R2 file actually exists
    if either of these checks fail, the user in notified and the script ends

    PARAMETERS
    -----------
    R2_path: str
        path to an R2 file. "_R2_" should be present and only show up once

    RETURNS
    ----------
    R1_name: str
        path to corresponding R1 file

    this is a partner function to "find_R1"

    '''

    fname = R2_path.split('/')[-1]
    dir_name = '/'.join(R2_path.split('/')[:-1]) + '/'

    if '_R1_' in fname:
        print("Error: It seems like you provided R1 files. \n" \
               "Please provide _R2_ files and try again.")
        sys.exit()

    fname_R1 = fname.replace('_R2_', '_R1_')
    R1_name = dir_name + fname_R1

    if not os.path.isfile(R1_name):
        print(f"Error: The following R1 file was not found: \n" \
              "{R1_name}\n" \
               "Please check your inputs and R2 files, and try again")
        sys.exit()
    return(R1_name)


def find_R2(R1_path):

    '''
    obtains the corresponding R2 file given a path to R1
    this functions performs the following two checks:
    i) checks if "_R2_" shows up in the file name
    ii) checks if the R1 file actually exists
    if either of these checks fail, the user in notified and the script ends

    this is a partner function to "find_R1"

    PARAMETERS
    -----------
    R1_path: str
        path to an R1 file. "_R1_" should be present and only show up once

    RETURNS
    ----------
    R2_name: str
        path to corresponding R2 file

    '''

    fname = R1_path.split('/')[-1]
    dir_name = '/'.join(R1_path.split('/')[:-1]) + '/'

    if '_R2_' in fname:
        print("Error: It seems like you provided R2 files. \n" \
               "Please provide _R1_ files and try again.")
        sys.exit()

    fname_R2 = fname.replace('_R1_', '_R2_')
    R2_name = dir_name + fname_R2

    if not os.path.isfile(R2_name):
        print(f"Error: The following R2 file was not found: \n" \
              "{R2_name}\n" \
               "Please check your inputs and R2 files, and try again")
        sys.exit()
    return(R2_name)


def find_mutations(seq1, seq2):

    """

    identify mutations between two sequences. this function to agnostic to the
    kinds of sequences that are input. this function will only work if seq1 and
    seq2 are the same length. this should be validated outside of this function

    PARAMETERS
    -----------
    seq1: str
        first dna sequence
    seq2: str
        second dna sequence

    RETURNS
    -----------
    mutations: list of str
        list of mutations. each mutation is of the form "YnZ". Y is the original
        character, n is the location of the mutation, and Z is the new character

    """

    mutations = []
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            mutations.append((f"{seq1[i]}{i+1}{seq2[i]}"))
    return(mutations)

def translate_sequence(dna_seq):

    '''
    translates a given dna sequence. only works for dna sequences that do
    not contain non-canonical nucleotides.
    adapted from: https://www.geeksforgeeks.org/dna-protein-python-3/
    '''

    dna_to_aa = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    protein =""
    if len(dna_seq)%3 == 0:
        for i in range(0, len(dna_seq), 3):
            codon = dna_seq[i:i + 3].upper()
            protein+= dna_to_aa[codon]

    return(protein)

def hamming_distance(sequence_1,
                    sequence_2):

    '''
    calculate hamming distance between two strings: that is, what is the minimal
    number of substitutions to get from string 1 to string 2. this script assumes
    that len(sequence_1) == len(sequence_2). this function is adapted from:
    http://claresloggett.github.io/python_workshops/improved_hammingdist.html

    PARAMETERS
    --------------------
    sequence_1: str
        first string or nucleic acid sequence to compare
    sequence_2: str
        second string or nucleic acid sequence to compare

    RETURNS
    --------------------
    distance: int
        the hamming distance between sequence_1 and sequence_2
    '''

    # initialize distance to 0
    distance = 0
    len_sequence_1 = len(sequence_1)

    # loop over entire string / sequence and compare each position of
    # sequence 1 vs. 2
    for base in range(len_sequence_1):
        # Add 1 to the distance if these two characters are not equal
        if sequence_1[base] != sequence_2[base]:
            distance += 1
    # return hamming distance
    return(distance)

def get_yes_no_input(mb):

    '''
    Simple quality control check to make sure the user used the correct
    arguments when running a function. Specifically, this prompts the user
    to double check that the files and inputs for different minibinders were not
    mixed up.

    PARAMETERS
    --------------------
    mb: str
        name of the minibinder input by the user

    RETURNS
    --------------------
    None
    '''

    while True:
        user_input = input(f"Are you sure you want to analyze {mb} (yes/no): ").lower()
        if user_input in ("yes", "y"):
            return True
        elif user_input in ("no", "n"):
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def extract_mutations_from_string(mutations_str):

    """
    Takes mutations that are in string format -- i.e., from the counts and
    scores dataframes -- and converts them into a list

    PARAMETERS
    --------------------
    mutations_str: str
        string that contains a list of mutations

    RETURNS
    --------------------
    mutations_list: list of str
        list of strings with mutations
    """

    mutations_list = []
    mutations_list.extend(mutations_str.strip('[').strip(']').replace("'", "").split(','))
    return(mutations_list)

def flatten_mutations(xss):
    
    """
    Flattens a list of lists of mutations

    PARAMETERS
    --------------------
    mutations_str: xss
        list to flatten

    RETURNS
    --------------------
    flattened list of mutations
    """

    # The .replace(" ","") is used because some mutations contain a whitespace
    return [x.replace(" ","") for xs in xss for x in xs]
