#!/bin/sh
env="hidenseek"
scenario_name="quadrant"
num_agents=2
num_boxes=4
floor_size=6.0
task_type='all'
algo="Boxlocking_2agent_6box_success_all_floor6_fixed_door"
#algo=""
seed_max=1

ulimit -n 4096
export OPENBLAS_NUM_THREADS=1

echo "env is ${env}, scenario is ${scenario_name}, num_agents is ${num_agents}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train_hns_transfertask.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --task_type ${task_type} --seed ${seed} --floor_size ${floor_size}  --n_rollout_threads 260 --num_mini_batch 1 --episode_length 120 --num_env_steps 100000000 --ppo_epoch 15 --attn --save_interval 20
    echo "training is done!"
done
