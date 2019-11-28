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
    
    
    ### Baseline 1: Normalized Frequency Difference (FD) ###
    
    # Get normalized (-n) frequencies for both corpora
    python3 code/freq.py -n trial_data_public/corpora/$language/corpus1 trial_data_public/targets/$language.txt results/$language/freq_corpus1.csv
    python3 code/freq.py -n trial_data_public/corpora/$language/corpus2 trial_data_public/targets/$language.txt results/$language/freq_corpus2.csv

    # Subtract frequencies and store absolute (-a) difference
    python3 code/diff.py -a trial_data_public/targets/$language.txt results/$language/freq_corpus1.csv results/$language/freq_corpus2.csv results/$language/fd.csv

    # Classify results into two classes according to threshold
    python3 code/class.py results/$language/fd.csv results/$language/fd_classes.csv 0.0003

    
    ### Baseline 2: Count Vectors with Column Intersection and Cosine Distance (CNT+CI+CD) ###
    
    # Get co-occurrence matrices for both corpora
    python3 code/cnt.py trial_data_public/corpora/$language/corpus1 matrices/$language/cnt_matrix1 1
    python3 code/cnt.py trial_data_public/corpora/$language/corpus2 matrices/$language/cnt_matrix2 1

    # Align matrices with Column Intersection
    python3 code/ci.py matrices/$language/cnt_matrix1 matrices/$language/cnt_matrix2 matrices/$language/cnt_matrix1_aligned matrices/$language/cnt_matrix2_aligned

    # Load matrices and calculate Cosine Distance
    python3 code/cd.py trial_data_public/targets/$language.txt matrices/$language/cnt_matrix1_aligned matrices/$language/cnt_matrix2_aligned results/$language/cnt_ci_cd.csv
    
    # Classify results into two classes according to threshold
    python3 code/class.py results/$language/cnt_ci_cd.csv results/$language/cnt_ci_cd_classes.csv 0.4

    
    ### Make answer files for submission ###
    
    # Baseline 1
    mkdir -p results/answer/task2/ && cp results/$language/fd.csv results/answer/task2/$language.txt
    mkdir -p results/answer/task1/ && cp results/$language/fd_classes.csv results/answer/task1/$language.txt
    cd results/ && zip -r answer_fd.zip answer/ && rm -r answer/ && cd ..
    
    # Baseline 2
    mkdir -p results/answer/task2/ && cp results/$language/cnt_ci_cd.csv results/answer/task2/$language.txt
    mkdir -p results/answer/task1/ && cp results/$language/cnt_ci_cd_classes.csv results/answer/task1/$language.txt
    cd results/ && zip -r answer_cnt_ci_cd.zip answer/ && rm -r answer/ && cd ..
    
done
