#! /bin/bash

for LANG in english german italian latin norwegian1 norwegian2 russian1 russian2 russian3 swedish
do
    echo "==============="
    echo ${LANG}
    mkdir -p ../results2022/${LANG}

    echo "Computing COS scores..."
    python3 ../code/new_scores/cosine_distance.py -i0 embeddings/${LANG}/corpus1.npz -i1 embeddings/${LANG}/corpus2.npz -t targets/${LANG}/targets.txt > ../results2022/${LANG}/cos.tsv
    echo "Computing APD scores..."
    python3 ../code/new_scores/average_pairwise_distance.py -i0 embeddings/${LANG}/corpus1.npz -i1 embeddings/${LANG}/corpus2.npz -t targets/${LANG}/targets.txt > ../results2022/${LANG}/apd.tsv
    echo "Computing JSD scores..."
    python3 ../code/jsd.py targets/${LANG}/targets.txt embeddings/${LANG}/corpus1.npz embeddings/${LANG}/corpus2.npz ../results2022/${LANG}/jsd.tsv

    echo "Computing combined scores..."
    python3 ../code/new_scores/combine_scores.py -i0 ../results2022/${LANG}/apd.tsv -i1 ../results2022/${LANG}/cos.tsv -o ../results2022/${LANG}/apd_cos_geom.tsv
    echo "Computing binary scores..."
    python3 ../code/new_scores/binary_scores.py -i ../results2022/${LANG}/apd_cos_geom.tsv -o ../results2022/${LANG}/apd_cos_geom_binary.tsv
    python3 ../code/new_scores/binary_scores.py -i ../results2022/${LANG}/cos.tsv -o ../results2022/${LANG}/cos_binary.tsv
    python3 ../code/new_scores/binary_scores.py -i ../results2022/${LANG}/apd.tsv -o ../results2022/${LANG}/apd_binary.tsv
    python3 ../code/new_scores/binary_scores.py -i ../results2022/${LANG}/jsd.tsv -o ../results2022/${LANG}/jsd_binary.tsv
done