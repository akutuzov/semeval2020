# Provide project path as argument to this script!
project_path=$1
cd $project_path || exit

unzip -n test_data_public.zip
data=test_data_public


# English
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
unzip -n semeval2020_ulscd_eng.zip
rm -rf $data/english
mv semeval2020_ulscd_eng  $data/english
rm semeval2020_ulscd_eng.zip

# German
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
unzip -n semeval2020_ulscd_ger.zip
rm -rf $data/german
mv semeval2020_ulscd_ger  $data/german
rm semeval2020_ulscd_ger.zip

# Latin
wget https://zenodo.org/record/3674988/files/semeval2020_ulscd_lat.zip
unzip -n semeval2020_ulscd_lat.zip
rm -rf $data/latin
mv semeval2020_ulscd_lat  $data/latin
rm semeval2020_ulscd_lat.zip

# Swedish
wget https://zenodo.org/record/3672950/files/semeval2020_ulscd_swe.zip
unzip -n semeval2020_ulscd_swe.zip
rm -rf $data/swedish
mv semeval2020_ulscd_swe  $data/swedish
rm semeval2020_ulscd_swe.zip


declare -a languages=(english latin german swedish)
declare -a methods=(avg last last4 mid4)

for language in "${languages[@]}"
do
  for method in "${methods[@]}"
  do

    targets=$data/$language/targets.txt
    npz1=matrices_eval/$language/corpus1_${method}.npz
    npz2=matrices_eval/$language/corpus2_${method}.npz
    out=results_eval/$language

	  # TASK 2

    # Average pairwise distance
    python3 code/bert/distance.py $targets $npz1 $npz2 $out/bert_apd_${method}.csv

    # Absolute difference in mean relatedness
    python3 code/bert/relatedness.py -a $targets $npz1 $npz2 $out/bert_amrd_${method}.csv

    # Diversity (pairwise and wrt centroid)
    python3 code/elmo/get_coeffs.py -t $targets -i0 $npz1 -i1 $npz2 --mode centroid --output $out/bert_cdiv_${method}.csv
    python3 code/elmo/get_coeffs.py -t $targets -i0 $npz1 -i1 $npz2 --mode pairwise --output $out/bert_pdiv_${method}.csv

    # JSD between usage distributions
    python3 code/jsd.py $targets $npz1 $npz2 $out/bert_jsd_${method}.csv

	  # Cosine distance between static embeddings
    python3 code/cosine_baseline.py -t $targets -i0 $npz1 -i1 $npz2 --mode mean --output $out/bert-mean_cosine_${method}.csv

	done
done


