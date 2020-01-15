# Unzip trial data
unzip -o trial_data_public.zip

# Make folders
mkdir -p matrices/
mkdir -p results/

# Iterate over languages
declare -a languages=(english german latin swedish)
for language in "${languages[@]}"
do

    # Make folders
    mkdir -p matrices/$language/
    mkdir -p results/$language/

    # Collect contextualised representations
    python3 code/bert/collect.py models/bert/$language/config.txt trial_data_public/corpora/$language/corpus1 trial_data_public/targets/$language.txt matrices/$language/bert_corpus1.dict
    python3 code/bert/collect.py models/bert/$language/config.txt trial_data_public/corpora/$language/corpus2 trial_data_public/targets/$language.txt matrices/$language/bert_corpus2.dict

    # Compute diachronic average pairwise distance
    python3 code/bert/distance.py trial_data_public/targets/$language.txt matrices/$language/bert_corpus1.dict matrices/$language/bert_corpus2.dict results/$language/bert_avg_pw_dist.csv

    # Compute diachronic difference in mean relatedness
    python3 code/bert/relatedness.py trial_data_public/targets/$language.txt matrices/$language/bert_corpus1.dict matrices/$language/bert_corpus2.dict results/$language/bert_mean_rel_diff.csv

    # Classify results into two classes according to threshold
    python3 code/class.py results/$language/bert_avg_pw_dist.csv results/$language/bert_avg_pw_dist_classes.csv 0.1
    python3 code/class.py results/$language/bert_mean_rel_diff.csv results/$language/bert_mean_rel_diff_classes.csv 0.0


    ### Make answer files for submission ###

    # average pairwise distance
    mkdir -p results/answer/task2/ && cp results/$language/bert_avg_pw_dist.csv results/answer/task2/$language.txt
    mkdir -p results/answer/task1/ && cp results/$language/bert_avg_pw_dist_classes.csv results/answer/task1/$language.txt
    cd results/ && zip -r answer_bert_avg_pw_dist.zip answer/ && rm -r answer/ && cd ..

    # difference in mean relatedness
    mkdir -p results/answer/task2/ && cp results/$language/bert_mean_rel_diff.csv results/answer/task2/$language.txt
    mkdir -p results/answer/task1/ && cp results/$language/bert_mean_rel_diff_classes.csv results/answer/task1/$language.txt
    cd results/ && zip -r answer_bert_mean_rel_diff.zip answer/ && rm -r answer/ && cd ..

done
