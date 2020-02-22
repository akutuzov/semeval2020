# For compatibility with server
project_path=$1

# Unzip trial data
unzip -o $project_path/trial_data_public.zip

# Make folders
mkdir -p $project_path/matrices/
mkdir -p $project_path/results/

# Iterate over languages
declare -a languages=(english german latin swedish)
for language in "${languages[@]}"
do

    # Make folders
    mkdir -p $project_path/matrices/$language/
    mkdir -p $project_path/results/$language/

    # Collect contextualised representations
    python3 $project_path/code/bert/collect.py $project_path/models/bert/$language/config.txt $project_path/trial_data_public/corpora/$language/corpus1 $project_path/trial_data_public/targets/$language.txt $project_path/matrices/$language/bert_corpus1.dict
    python3 $project_path/code/bert/collect.py $project_path/models/bert/$language/config.txt $project_path/trial_data_public/corpora/$language/corpus2 $project_path/trial_data_public/targets/$language.txt $project_path/matrices/$language/bert_corpus2.dict

    # Compute diachronic average pairwise distance
    python3 $project_path/code/bert/distance.py $project_path/trial_data_public/targets/$language.txt $project_path/matrices/$language/bert_corpus1.dict $project_path/matrices/$language/bert_corpus2.dict $project_path/results/$language/bert_avg_pw_dist.csv

    # Compute diachronic (absolute) difference in mean relatedness
    python3 $project_path/code/bert/relatedness.py -a $project_path/trial_data_public/targets/$language.txt $project_path/matrices/$language/bert_corpus1.dict $project_path/matrices/$language/bert_corpus2.dict $project_path/results/$language/bert_mean_rel_diff.csv

    # Classify results into two classes according to threshold
    python3 $project_path/code/class.py $project_path/results/$language/bert_avg_pw_dist.csv $project_path/results/$language/bert_avg_pw_dist_classes.csv 0.5
    python3 $project_path/code/class.py $project_path/results/$language/bert_mean_rel_diff.csv $project_path/results/$language/bert_mean_rel_diff_classes.csv 0.04


    ### Make answer files for submission ###

    # average pairwise distance
    mkdir -p $project_path/results/answer/task2/ && cp $project_path/results/$language/bert_avg_pw_dist.csv $project_path/results/answer/task2/$language.txt
    mkdir -p $project_path/results/answer/task1/ && cp $project_path/results/$language/bert_avg_pw_dist_classes.csv $project_path/results/answer/task1/$language.txt
    zip -r $project_path/results/answer_bert_avg_pw_dist.zip $project_path/results/answer/ && rm -r $project_path/results/answer/

    # difference in mean relatedness
    mkdir -p $project_path/results/answer/task2/ && cp $project_path/results/$language/bert_mean_rel_diff.csv $project_path/results/answer/task2/$language.txt
    mkdir -p $project_path/results/answer/task1/ && cp $project_path/results/$language/bert_mean_rel_diff_classes.csv $project_path/results/answer/task1/$language.txt
    zip -r $project_path/results/answer_bert_mean_rel_diff.zip $project_path/results/answer/ && rm -r $project_path/results/answer/

done
