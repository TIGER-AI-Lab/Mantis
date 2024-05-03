mkdir -p logs
model_name=$1
python eval.py --dataset_path "TIGER-Lab/Mantis-Eval" --dataset_name "mantis_eval" --model_name $model_name --results_dir "results" --overwrite False > ./logs/eval_${model_name}_mantis_eval.log 2>&1
python eval.py --dataset_path "TIGER-Lab/NLVR2" --dataset_name "nlvr2" --model_name $model_name --results_dir "results" --overwrite False > ./logs/eval_${model_name}_nlvr2.log 2>&1

dataset_path="../../data/qbench2/data/q-bench2-a1-pair-dev.json"
subset="q-bench2-a1-pair-dev"
python eval.py --dataset_path $dataset_path --dataset_name $subset --model_name $model_name --results_dir "results" --overwrite True > ./logs/eval_${model_name}_${subset}.log 2>&1
