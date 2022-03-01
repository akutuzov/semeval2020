#!/bin/bash

#SBATCH --job-name=XLMR_predict
#SBATCH --account=ec30
#SBATCH --time=20:00:00      # Max walltime is 150 hours.
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=10

# Definining resource we want to allocate.
#SBATCH --nodes=1

module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module purge   # Recommended for reproducibility
module load nlpl-python-candy/2021.01-foss-2019b-Python-3.7.4
module load nlpl-scikit-bundle/0.22.2.post1-foss-2019b-Python-3.7.4

bash run_xlmr_pred.sh
