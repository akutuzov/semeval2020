# For compatibility with server
project_path=$1


# Get data

unzip -n $project_path/test_data_public.zip
data=$project_path/test_data_public

cd $project_path || exit

# English
wget -q https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
unzip -n semeval2020_ulscd_eng.zip
rm -rf $data/english
mv semeval2020_ulscd_eng  ${data:1}/english
rm semeval2020_ulscd_eng.zip

# German
wget -q https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
unzip -n semeval2020_ulscd_ger.zip
rm -rf $data/german
mv semeval2020_ulscd_ger  ${data:1}/german
rm semeval2020_ulscd_ger.zip

# Latin
wget -q https://zenodo.org/record/3674099/files/semeval2020_ulscd_lat.zip
unzip -n semeval2020_ulscd_lat.zip
rm -rf $data/latin
mv semeval2020_ulscd_lat  ${data:1}/latin
rm semeval2020_ulscd_lat.zip

# Swedish
wget -q https://zenodo.org/record/3672950/files/semeval2020_ulscd_swe.zip
unzip -n semeval2020_ulscd_swe.zip
rm -rf $data/swedish
mv semeval2020_ulscd_swe  ${data:1}/swedish
rm semeval2020_ulscd_swe.zip


cd || exit


# Make folders
mkdir -p $project_path/matrices_eval/
mkdir -p $project_path/results_eval/

# Iterate over languages
declare -a languages=(english german latin swedish)
for language in "${languages[@]}"
do

    # Make folders
    mkdir -p $project_path/matrices_eval/$language/
    mkdir -p $project_path/results_eval/$language/

    # Collect contextualised representations
    python3 $project_path/code/bert/collect.py $project_path/models/bert/$language/config.txt $data/corpora/$language/corpus1 $data/targets/$language.txt $project_path/matrices_eval/$language/bert_corpus1.dict
    python3 $project_path/code/bert/collect.py $project_path/models/bert/$language/config.txt $data/corpora/$language/corpus2 $data/targets/$language.txt $project_path/matrices_eval/$language/bert_corpus2.dict

    # Compute diachronic average pairwise distance
    python3 $project_path/code/bert/distance.py $data/targets/$language.txt $project_path/matrices_eval/$language/bert_corpus1.dict $project_path/matrices_eval/$language/bert_corpus2.dict $project_path/results_eval/$language/bert_avg_pw_dist.csv

    # Compute diachronic (absolute) difference in mean relatedness
    python3 $project_path/code/bert/relatedness.py -a $data/targets/$language.txt $project_path/matrices_eval/$language/bert_corpus1.dict $project_path/matrices_eval/$language/bert_corpus2.dict $project_path/results_eval/$language/bert_mean_rel_diff.csv

    # Classify results into two classes according to threshold
    python3 $project_path/code/class.py $project_path/results_eval/$language/bert_avg_pw_dist.csv $project_path/results_eval/$language/bert_avg_pw_dist_classes.csv 0.5
    python3 $project_path/code/class.py $project_path/results_eval/$language/bert_mean_rel_diff.csv $project_path/results_eval/$language/bert_mean_rel_diff_classes.csv 0.04


    ### Make answer files for submission ###

    # average pairwise distance
    mkdir -p $project_path/results_eval/answer/task2/ && cp $project_path/results_eval/$language/bert_avg_pw_dist.csv $project_path/results_eval/answer/task2/$language.txt
    mkdir -p $project_path/results_eval/answer/task1/ && cp $project_path/results_eval/$language/bert_avg_pw_dist_classes.csv $project_path/results_eval/answer/task1/$language.txt
    zip -r $project_path/results_eval/answer_bert_avg_pw_dist.zip $project_path/results_eval/answer/ && rm -r $project_path/results_eval/answer/

    # difference in mean relatedness
    mkdir -p $project_path/results_eval/answer/task2/ && cp $project_path/results_eval/$language/bert_mean_rel_diff.csv $project_path/results_eval/answer/task2/$language.txt
    mkdir -p $project_path/results_eval/answer/task1/ && cp $project_path/results_eval/$language/bert_mean_rel_diff_classes.csv $project_path/results_eval/answer/task1/$language.txt
    zip -r $project_path/results_eval/answer_bert_mean_rel_diff.zip $project_path/results_eval/answer/ && rm -r $project_path/results_eval/answer/

done
