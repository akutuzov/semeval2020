#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile

CORPUS1=${1}
FILENAME1=$(basename -- "$CORPUS1" .txt.gz)
CORPUS2=${2}
FILENAME2=$(basename -- "$CORPUS2" .txt.gz)
ELMO1=${3}
ELMO2=${4}
TARGET=${5}
LANG=${6}

echo 'Inferring embeddings for the corpus 1...'
python3 code/elmo/extract_elmo.py --input ${CORPUS1} --elmo ${ELMO1} --outfile ${FILENAME1}.npz  --vocab ${TARGET}

echo 'Inferring embeddings for the corpus 2...'
python3 code/elmo/extract_elmo.py --input ${CORPUS2} --elmo ${ELMO2} --outfile ${FILENAME2}.npz --vocab ${TARGET}

echo 'Calculating diversity coefficients...'
python3 code/elmo/get_coeffs.py -i0 ${FILENAME1}.npz -i1 ${FILENAME2}.npz > ${FILENAME1}_${FILENAME2}.tsv

echo 'Preparing a submission...'
mkdir -p elmo_embeddings/answer/task2/ && mv ${FILENAME1}_${FILENAME2}.tsv elmo_embeddings/answer/task2/$LANG.txt
# zip -r elmo_embeddings/answer.zip elmo_embeddings/answer/ && rm -r elmo_embeddings/answer/
