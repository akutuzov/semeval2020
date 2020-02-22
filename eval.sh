# For compatibility with server
project_path=$1

# Unzip trial data
unzip -oq $project_path/trial_data_public.zip

# Iterate over languages
declare -a languages=(english german latin swedish)
for language in "${languages[@]}"
do
    echo $language

    true_answers1_path="$project_path/trial_data_public/answer/task1/$language.txt"
    true_answers2_path="$project_path/trial_data_public/answer/task2/$language.txt"

#    echo -e "\nFrequency Difference"
#    unzip -oq $project_path/results/answer_fd.zip
#    python3 $project_path/code/eval.py $project_path/results/answer/task1/$language.txt $project_path/results/answer/task2/$language.txt $true_answers1_path $true_answers2_path
#    rm -r $project_path/answer/
#
#    echo -e "\nColumn Intersection + Cosine Distance"
#    unzip -oq $project_path/results/answer_cnt_ci_cd.zip
#    python3 $project_path/code/eval.py $project_path/results/answer/task1/$language.txt $project_path/results/answer/task2/$language.txt $true_answers1_path $true_answers2_path
#    rm -r $project_path/answer/

    echo -e "\nAverage Pairwise Distance"
    unzip -oq $project_path/results/answer_bert_avg_pw_dist.zip
    python3 $project_path/code/eval.py $project_path/results/answer/task1/$language.txt $project_path/results/answer/task2/$language.txt $true_answers1_path $true_answers2_path
    rm -r $project_path/answer/

    echo -e "\nMean Relatedness Difference"
    unzip -oq $project_path/results/answer_bert_mean_rel_diff.zip
    python3 $project_path/code/eval.py $project_path/results/answer/task1/$language.txt $project_path/results/answer/task2/$language.txt $true_answers1_path $true_answers2_path
    rm -r $project_path/answer/

    echo -e "\n"

done
