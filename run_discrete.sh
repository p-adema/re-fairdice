#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DIST=uniform
Learner=FairDICE
Divergence=SOFT_CHI
ENV=MO-FourRooms
QUALITY=expert 

for beta in 1.0 0.1 0.01 0.001; do
    for seed in 1 2 3 4 5; do
        echo "Running $Learner $ENV $QUALITY $beta $seed"
        
        hidden_dim=256
        num_layers=2

        python main.py \
            --learner $Learner \
            --divergence $Divergence \
            --env_name $ENV \
            --quality $QUALITY \
            --beta $beta \
            --seed $seed \
            --preference_dist $DIST \
            --eval_episodes 50 \
            --batch_size 256 \
            --hidden_dim $hidden_dim \
            --num_layers $num_layers \
            --total_train_steps 50000 \
            --log_interval 1000 \
            --normalize_reward False
    done
done