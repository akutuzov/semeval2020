# Unzip trial data
unzip -o trial_data_public.zip

# Iterate over languages
declare -a languages=(english german latin swedish)
for language in "${languages[@]}"
do
    echo $language

    true_answers1_path="trial_data_public/answer/task1/$language.txt"
    true_answers2_path="trial_data_public/answer/task2/$language.txt"

    echo "Frequency Difference"
    unzip -oq results/answer_fd.zip
    python3 code/eval.py answer/task1/$language.txt answer/task2/$language.txt $true_answers1_path $true_answers2_path
    rm -r answer/

    echo "Column Intersection + Cosine Distance"
    unzip -oq results/answer_cnt_ci_cd.zip
    python3 code/eval.py answer/task1/$language.txt answer/task2/$language.txt $true_answers1_path $true_answers2_path
    rm -r answer/

done
