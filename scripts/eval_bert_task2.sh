# Provide project path as argument to this script!
project_path=$1
cd $project_path || exit

declare -a languages=(english latin german) # swedish)
declare -a methods=(avg last last4 mid4)
declare -a metrics=(_apd _amrd _jsd _pdiv _cdiv -mean_cosine)

for language in "${languages[@]}"
do
  echo ">>>" $language "<<<"
  gold1=test_data_truth/task1/${language}.txt
  gold2=test_data_truth/task2/${language}.txt
  out=results_eval/$language

  for method in "${methods[@]}"
  do
    echo ">>" $method "<<"

    for metric in "${metrics[@]}"
    do
	    echo '>' $metric '<'
      python3 code/eval.py none ${out}/bert${metric}_${method}.csv $gold1 $gold2
    done

	done
done
