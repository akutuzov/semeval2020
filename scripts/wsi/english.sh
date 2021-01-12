#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ft-en
#SBATCH --time=24:00:00
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
	${model} \
	finetuning_corpora/${language}/${preproc}/corpus1.txt.gz \
	finetuning_corpora/targets/${language}/target_forms.csv \
	subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
	--nSubs ${nsubs_start} \
	--batch ${batch} \
	--context ${context} \
	--ignoreBias \
	&> out/substitutes_${language}_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}

echo "substitutes.py: T2"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/substitutes.py \
	${model} \
	finetuning_corpora/${language}/${preproc}/corpus2.txt.gz \
	finetuning_corpora/targets/${language}/target_forms.csv \
	subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
	--nSubs ${nsubs_start} \
	--batch ${batch} \
	--context ${context} \
	--ignoreBias \
	&> out/substitutes_${language}_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}


echo "inject_lexical_similarity.py: T1"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/inject_lexical_similarity.py \
  ${model} \
  finetuning_corpora/targets/${language}/target_forms.csv \
	subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
  subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  --batch ${batch} \
  --normalise \
  --ignoreBias \
  &> out/inject_${language}_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}

echo "inject_lexical_similarity.py: T2"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/inject_lexical_similarity.py \
  ${model} \
  finetuning_corpora/targets/${language}/target_forms.csv \
	subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.pkl \
  subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  --batch ${batch} \
  --normalise \
  --ignoreBias \
  &> out/inject_${language}_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}


echo "postprocessing.py: T1"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/postprocessing.py \
  subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  subs_results/${language}/corpus_${preproc}_T1_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  --nSubs ${nsubs_end} \
  --temperature ${temperature} \
  --language ${lang}\
  --lemmatise \
  --frequency ${freq_type} \
  &> out/postproc_${language}_${preproc}_T1_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}

echo "postprocessing.py: T2"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/postprocessing.py \
  subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}_ctx${context}_bsz${batch}.lexsim.pkl \
  subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  --nSubs ${nsubs_end} \
  --temperature ${temperature} \
  --language ${lang}\
  --lemmatise \
  --frequency ${freq_type} \
  &> out/postproc_${language}_${preproc}_T2_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}


echo "wsi_jsd.py"
python3 wsi.py \
  subs_results/${language}/corpus1_${preproc}_T1_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  subs_results/${language}/corpus_${preproc}_T2_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}.post.pkl \
  --tfidf \
  --nClusters=${nClusters} \
  &> out/jsd_${language}_${preproc}_nsubs${nsubs_start}-${nsubs_end}_ctx${context}_bsz${batch}_temp${temperature}_fr_${freq_type}
