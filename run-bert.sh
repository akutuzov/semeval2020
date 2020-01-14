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
    python3 code/bert/collect.py models/bert/$language/config.txt trial_data_public/corpora/$language/corpus1 trial_data_public/targets/$language.txt matrices/$language/bert_corpus1.np
    python3 code/bert/collect.py models/bert/$language/config.txt trial_data_public/corpora/$language/corpus2 trial_data_public/targets/$language.txt matrices/$language/bert_corpus2.np

done
