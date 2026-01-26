#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
DIST=uniform
Learner=FairDICE
Divergence=SOFT_CHI
for ENV in MO-Hopper-v2 MO-Walker2d-v2 MO-Swimmer-v2 MO-HalfCheetah-v2 MO-Ant-v2 MO-Hopper-v3; do
    for quality in expert amateur; do
        for beta in 10 1 0.1 0.01 0.001 0.0001 0.00001; do
            for seed in 1 2 3 4 5 6 7 8 9 10; do
                echo "Running $Learner $ENV $quality $beta $seed"
                if [ "$ENV" == "MO-Hopper-v3" ]; then
                    num_layers=4
                else 
                    num_layers=3
                fi 
                if [ "$ENV" == "MO-Ant-v2" ]; then
                    hidden_dim=512
                else
                    hidden_dim=768
                fi
                uv run train.py \
                    --learner $Learner \
                    --divergence $Divergence \
                    --env_name $ENV \
                    --quality $quality \
                    --beta $beta \
                    --seed $seed \
                    --preference_dist $DIST \
                    --eval_episodes 10 \
                    --batch_size 256 \
                    --hidden_dim $hidden_dim \
                    --num_layers $num_layers \
                    --total_train_steps 100_000 \
                    --log_interval 10_000 \
                    --normalize_reward True \
                    --save_path ./pt-fix-grad0
            done
        done
    done
done
