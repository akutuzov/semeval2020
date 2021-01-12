#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ft-en
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc

# Load all necessary modules.
echo "Loading modules..."
module load 2019

echo "Loading virtual environment..."
source ${HOME}/projects/erp/venv/bin/activate

cd ${HOME}/projects/semeval2020 || exit

language=english
lang=en
preproc=token  # or lemma
model=bert-base-uncased  # bert-base-german-cased, bert-base-multilingual-cased, af-ai-center/bert-base-swedish-uncased
batch=32
context=200
nsubs_start=150
nsubs_end=100
temperature=1
freq_type=log #zipf
nClusters=7

echo "substitutes.py: T1"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/substitutes.py \
	--model_name ${model} \
	--corpus_path finetuning_corpora/${language}/${preproc}/1.txt \
	--targets_path finetuning_corpora/${language}/targets/target_forms.csv \
	--output_path subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
	--n_subs ${nsubs_start} \
	--batch_size ${batch} \
	--seq_len ${context} \
	--ignore_decoder_bias \
	&> out/substitutes_${language}_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}

echo "substitutes.py: T2"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/substitutes.py \
	--model_name ${model} \
	--corpus_path finetuning_corpora/${language}/${preproc}/2.txt \
	--targets_path finetuning_corpora/${language}/targets/target_forms.csv \
	--output_path subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
	--n_subs ${nsubs_start} \
	--batch_size ${batch} \
	--context ${context} \
	--ignore_decoder_bias \
	&> out/substitutes_${language}_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}


echo "inject_lexical_similarity.py: T1"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/inject_lexical_similarity.py \
  --model_name ${model} \
  --targets_path finetuning_corpora/${language}/targets/target_forms.csv \
	--subs_path subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
  --output_path subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  --batch_size ${batch} \
  --normalise_embeddings \
	--ignore_decoder_bias \
  &> out/inject_${language}_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}

echo "inject_lexical_similarity.py: T2"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/inject_lexical_similarity.py \
  --model_name ${model} \
  --targets_path finetuning_corpora/${language}/targets/target_forms.csv \
	--subs_path subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
  --output_path subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  --batch_size ${batch} \
  --normalise_embeddings \
	--ignore_decoder_bias \
  &> out/inject_${language}_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}


echo "postprocessing.py: T1"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/postprocessing.py \
  --subs_path subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  --output_path subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  --n_subs ${nsubs_end} \
  --temperature ${temperature} \
  --language ${lang} \
  --lemmatise \
  --frequency_type ${freq_type} \
  &> out/postproc_${language}_${preproc}_T1_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}

echo "postprocessing.py: T2"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/postprocessing.py \
  --subs_path subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  --output_path subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  --n_subs ${nsubs_end} \
  --temperature ${temperature} \
  --language ${lang} \
  --lemmatise \
  --frequency_type ${freq_type} \
  &> out/postproc_${language}_${preproc}_T2_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}


echo "wsi_jsd.py"
python3 code/wsi_jsd.py \
  --subs_path_t1 subs_results/${language}/corpus1_${preproc}_T1_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  --subs_path_t2 subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  --output_path subs_results/${language}/jsd_${preproc}_T2_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.csv \
  --apply_tfidf \
  --n_clusters=${nClusters} \
  &> out/jsd_${language}_${preproc}_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}
