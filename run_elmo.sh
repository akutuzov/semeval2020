#!/usr/bin/env bash

CORPUS1=${1}
FILENAME1=$(basename -- "$CORPUS1")
CORPUS2=${2}
FILENAME2=$(basename -- "$CORPUS2")
ELMO1=${3}
ELMO2=${4}
TARGET=${5}

echo 'Inferring embeddings for the corpus 1...'
python3 code/elmo/extract_elmo.py --input ${CORPUS1} --elmo ${ELMO1} --outfile ${FILENAME1}.npz  --vocab ${TARGET}

echo 'Inferring embeddings for the corpus 2...'
python3 code/elmo/extract_elmo.py --input CORPUS --elmo ELMO_MODEL --outfile ${FILENAME2}.npz --vocab TARGET_WORDS

echo 'Calculating diversity coefficients...'
python3 code/elmo/get_coeffs.py -i0 ${FILENAME1}.npz -i1 ${FILENAME2}.npz > ${FILENAME1}_${FILENAME1}.tsv
