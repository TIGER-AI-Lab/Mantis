dataset_path="TIGER-Lab/MIQA-Eval"
model_name=$1
for subset in "mantis_eval" "nlvr2"
do
    python eval.py --dataset_path $dataset_path --dataset_name $subset --model_name $model_name --results_dir "results" --overwrite False > ./logs/eval_${model_name}_${subset}.log 2>&1
done

dataset_path="../../data/qbench2/data/q-bench2-a1-pair-dev.json"
subset="q-bench2-a1-pair-dev"
python eval.py --dataset_path $dataset_path --dataset_name $subset --model_name $model_name --results_dir "results" --overwrite True > ./logs/eval_${model_name}_${subset}.log 2>&1
