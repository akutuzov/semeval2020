#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=results_real

method=bigone
our_answer=elmo_${method}_top_ap_cosine_half

gold=test_data_public

# Make folders
mkdir -p $data/${our_answer}/answer/task1
mkdir -p $data/${our_answer}/answer/task2


python3 code/pred_from_senses.py -t 0.3 -i0 $data/${our_answer}/english/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/english/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/english.txt
python3 code/pred_from_senses.py -t 0.4 -i0 $data/${our_answer}/german/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/german/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/german.txt
python3 code/pred_from_senses.py -t 0.8 -i0 $data/${our_answer}/latin/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/latin/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/latin.txt
python3 code/pred_from_senses.py -t 0.9 -i0 $data/${our_answer}/swedish/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/swedish/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/swedish.txt

python3 code/pred_from_senses.py -t ${1} -i0 $data/${our_answer}/english/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/english/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/english.txt
python3 code/pred_from_senses.py -t ${1} -i0 $data/${our_answer}/german/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/german/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/german.txt

python3 code/pred_from_senses.py -t 0.2 -i0 $data/${our_answer}/english/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/english/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/english.txt
python3 code/pred_from_senses.py -t 0.4 -i0 $data/${our_answer}/german/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/german/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/german.txt
python3 code/pred_from_senses.py -t 0.7 -i0 $data/${our_answer}/latin/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/latin/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/latin.txt
python3 code/pred_from_senses.py -t 0.8 -i0 $data/${our_answer}/swedish/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/swedish/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/swedish.txt
