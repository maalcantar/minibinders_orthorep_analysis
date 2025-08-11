# example usage: Rscript 00b_tite_seq_barcode_generation.R
# we did not set a seed for our run; each run will produce a different set
# of four DNA barcodes. The barcodes produced and ultimately used by us were:
# barcode 1: gtca
# barcode 2: tcgt
# barcode 3: caag
# barcode 4: agtc

library("DNABarcodes")

# create barcodes
mySetAshlock <- create.dnabarcodes(4, dist=4, heuristic="ashlock")

print(tolower(mySetAshlock)[[1]])
print(tolower(mySetAshlock)[[2]])
print(tolower(mySetAshlock)[[3]])
print(tolower(mySetAshlock)[[4]])
