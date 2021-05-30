#! /bin/bash

for lang in english german latin swedish
do
    for feat in morph synt
	do
	    echo ${lang} ${feat} `python3 compare_ling.py -i1 data/features/${lang}/corpus1_${feat}.json -i2 data/features/${lang}/corpus2_${feat}.json`
	done
done