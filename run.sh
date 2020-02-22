# Get data

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
wget https://zenodo.org/record/3674099/files/semeval2020_ulscd_lat.zip
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
    /opt/conda/envs/python3.6/bin/python3 code/freq.py -n $data/$language/corpus1/lemma $data/$language/targets.txt results/$language/freq_corpus1.csv
    /opt/conda/envs/python3.6/bin/python3 code/freq.py -n $data/$language/corpus2/lemma $data/$language/targets.txt results/$language/freq_corpus2.csv

    # Subtract frequencies and store absolute (-a) difference
    /opt/conda/envs/python3.6/bin/python3 code/diff.py -a $data/$language/targets.txt results/$language/freq_corpus1.csv results/$language/freq_corpus2.csv results/$language/fd.csv
    
    # Classify results into two classes according to threshold
    /opt/conda/envs/python3.6/bin/python3 code/class.py results/$language/fd.csv results/$language/fd_classes.csv 0.0000005
    
    
    ### Baseline 2: Count Vectors with Column Intersection and Cosine Distance (CNT+CI+CD) ###
    
    # Get co-occurrence matrices for both corpora
    /opt/conda/envs/python3.6/bin/python3 code/cnt.py $data/$language/corpus1/lemma matrices/$language/cnt_matrix1 1
    /opt/conda/envs/python3.6/bin/python3 code/cnt.py $data/$language/corpus2/lemma matrices/$language/cnt_matrix2 1

    # Align matrices with Column Intersection
    /opt/conda/envs/python3.6/bin/python3 code/ci.py matrices/$language/cnt_matrix1 matrices/$language/cnt_matrix2 matrices/$language/cnt_matrix1_aligned matrices/$language/cnt_matrix2_aligned

    # Load matrices and calculate Cosine Distance
    /opt/conda/envs/python3.6/bin/python3 code/cd.py $data/$language/targets.txt matrices/$language/cnt_matrix1_aligned matrices/$language/cnt_matrix2_aligned results/$language/cnt_ci_cd.csv
    
    # Classify results into two classes according to threshold
    /opt/conda/envs/python3.6/bin/python3 code/class.py results/$language/cnt_ci_cd.csv results/$language/cnt_ci_cd_classes.csv 0.4

    
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
