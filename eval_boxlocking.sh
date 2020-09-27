#!/bin/sh
env="box_locking"
scenario_name="quadrant"
num_agents=2
num_boxes=6
task_type='all-return'
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=1 python eval_hns.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --task_type ${task_type} --model_dir "/home/xuyf/mappo-hns/results/BoxLocking/quadrant/boxlocking_2agent_6box_sparse_default_allreturn/" --eval --seed 1
done
