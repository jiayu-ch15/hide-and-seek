#!/bin/sh
env="hidenseek"
scenario_name="quadrant"
num_hiders=1
num_seekers=1
num_boxes=1
num_ramps=4
floor_size=6.0
task_type='all-return'
algo="Hidenseek_2agent_1box_4ramp_floor6_curriculum_env520_step160"
# algo='check'
seed_max=1

ulimit -n 4096
export OPENBLAS_NUM_THREADS=1

echo "env is ${env}, scenario is ${scenario_name}, num_agents is ${num_agents}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train_hidenseek_curriculum.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_hiders ${num_hiders} --num_seekers ${num_seekers} --num_boxes ${num_boxes} --num_ramps ${num_ramps}  --task_type ${task_type} --seed ${seed} --floor_size ${floor_size}  --n_rollout_threads 520 --num_mini_batch 1 --episode_length 160 --num_env_steps 100000000 --ppo_epoch 15 --attn --save_interval 1 --eval
    echo "training is done!"
done
