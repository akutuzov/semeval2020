# For compatibility with server
project_path=$1


cd $project_path || exit

# Get data
unzip -n test_data_public.zip
data=test_data_public

# English
wget -q https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
unzip -n semeval2020_ulscd_eng.zip
rm -rf $data/english
mv semeval2020_ulscd_eng  $data/english
rm semeval2020_ulscd_eng.zip

# German
wget -q https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
unzip -n semeval2020_ulscd_ger.zip
rm -rf $data/german
mv semeval2020_ulscd_ger  $data/german
rm semeval2020_ulscd_ger.zip

# Latin
wget -q https://zenodo.org/record/3674099/files/semeval2020_ulscd_lat.zip
unzip -n semeval2020_ulscd_lat.zip
rm -rf $data/latin
mv semeval2020_ulscd_lat  $data/latin
rm semeval2020_ulscd_lat.zip

# Swedish
wget -q https://zenodo.org/record/3672950/files/semeval2020_ulscd_swe.zip
unzip -n semeval2020_ulscd_swe.zip
rm -rf $data/swedish
mv semeval2020_ulscd_swe  $data/swedish
rm semeval2020_ulscd_swe.zip




# Make folders
mkdir -p matrices_eval/
mkdir -p results_eval/

# Iterate over languages
declare -a languages=(english german latin swedish)
for language in "${languages[@]}"
do

    # Make folders
    mkdir -p matrices_eval/$language/
    mkdir -p results_eval/$language/

    # Collect contextualised representations
    python3 code/bert/collect.py models/bert/$language/config.txt $data/$language/corpus1/lemma $data/$language/targets.txt matrices_eval/$language/bert_corpus1.dict
    python3 code/bert/collect.py models/bert/$language/config.txt $data/$language/corpus2/lemma $data/$language/targets.txt matrices_eval/$language/bert_corpus2.dict

    # Compute diachronic average pairwise distance
    python3 code/bert/distance.py $data/$language/targets.txt matrices_eval/$language/bert_corpus1.dict matrices_eval/$language/bert_corpus2.dict results_eval/$language/bert_avg_pw_dist.csv

    # Compute diachronic (absolute) difference in mean relatedness
    python3 code/bert/relatedness.py -a $data/$language/targets.txt matrices_eval/$language/bert_corpus1.dict matrices_eval/$language/bert_corpus2.dict results_eval/$language/bert_mean_rel_diff.csv

    # Classify results into two classes according to threshold
    python3 code/class.py results_eval/$language/bert_avg_pw_dist.csv results_eval/$language/bert_avg_pw_dist_classes.csv 0.5
    python3 code/class.py results_eval/$language/bert_mean_rel_diff.csv results_eval/$language/bert_mean_rel_diff_classes.csv 0.04


    ### Make answer files for submission ###

    # average pairwise distance
    mkdir -p results_eval/answer/task2/ && cp results_eval/$language/bert_avg_pw_dist.csv results_eval/answer/task2/$language.txt
    mkdir -p results_eval/answer/task1/ && cp results_eval/$language/bert_avg_pw_dist_classes.csv results_eval/answer/task1/$language.txt
    zip -r results_eval/answer_bert_avg_pw_dist.zip results_eval/answer/ && rm -r results_eval/answer/

    # difference in mean relatedness
    mkdir -p results_eval/answer/task2/ && cp results_eval/$language/bert_mean_rel_diff.csv results_eval/answer/task2/$language.txt
    mkdir -p results_eval/answer/task1/ && cp results_eval/$language/bert_mean_rel_diff_classes.csv results_eval/answer/task1/$language.txt
    zip -r results_eval/answer_bert_mean_rel_diff.zip results_eval/answer/ && rm -r results_eval/answer/

done
