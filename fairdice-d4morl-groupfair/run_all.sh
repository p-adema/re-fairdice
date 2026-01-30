#!/bin/bash
# This file will run all experiments needed to reproduce FairDICE baselines, and at the
# end also gathers results for GroupPolicy. Results are placed into subdirectories of
# the `structured-results` directory; to draw graphs, the BASE_DIR in boxplots.ipynb
# should be updated to point to your `structured-results` directory.
set -e
export CUDA_VISIBLE_DEVICES=0
DIST=uniform
Learner=FairDICE
Divergence=SOFT_CHI
for LOSS in wrong-broadcast behaviour-cloning; do
    for ENV in MO-Hopper-v2 MO-Walker2d-v2 MO-Swimmer-v2 MO-HalfCheetah-v2 MO-Ant-v2 MO-Hopper-v3; do
        for quality in expert amateur; do
            for beta in 10 1 0.1 0.01 0.001 0.0001 0.00001; do
                for seed in 1 2 3 4 5 6 7 8 9 10; do
                    echo "Running $LOSS $ENV $quality $beta $seed"
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
                    python main.py \
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
                        --log_interval 25_000 \
                        --normalize_reward True \
                        --loss_kind $LOSS \
                        --save_path "./structured-results/results-$LOSS"
                done
            done
        done
    done
done

for GRAD in 0 0.0001 0; do
    for ENV in MO-Hopper-v2 MO-Walker2d-v2 MO-Swimmer-v2 MO-HalfCheetah-v2 MO-Ant-v2 MO-Hopper-v3; do
        for quality in expert amateur; do
            for beta in 10 1 0.1 0.01 0.001 0.0001 0.00001; do
                for seed in 1 2 3 4 5 6 7 8 9 10; do
                    echo "Running fixed-fairdice with grad $LOSS $ENV $quality $beta $seed"
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
                    python main.py \
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
                        --log_interval 25_000 \
                        --normalize_reward True \
                        --loss_kind fixed-fairdice \
                        --gradient_penalty_coeff $GRAD \
                        --save_path "./structured-results/results-fixed-$GRAD"
                done
            done
        done
    done
done

num_layers=3
hidden_dim=768
quality=expert
beta=1
for ENV in MO-Hopper-v2 MO-Swimmer-v2; do
    for DIST in narrow wide; do
        for seed in 1 2 3 4 5; do
            echo "Running  $ENV $DIST $seed"
            python main.py \
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
                --log_interval 25_000 \
                --normalize_reward True \
                --gradient_penalty_coeff 0 \
                --loss_kind fixed-fairdice \
                --save_path "./structured-results/results-filter-preference"
        done
    done
done


for NONLIN in piecewise-log-quadratic log; do
    for ENV in MO-Hopper-v2 MO-Walker2d-v2 MO-Swimmer-v2 MO-HalfCheetah-v2 MO-Ant-v2 MO-Hopper-v3; do
        for quality in expert amateur; do
            for beta in 10 1 0.1 0.01 0.001; do
                for seed in 1 2 3 4 5 6 7 8 9 10; do
                    echo "Running $NONLIN $ENV $quality $beta $seed"
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
                    python main.py \
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
                        --log_interval 25_000 \
                        --gradient_penalty_coeff 0 \
                        --loss_kind fixed-fairdice \
                        --u_nonlinearity $NONLIN \
                        --save_path "./structured-results/results-$NONLIN"
                done
            done
        done
    done
done


ENV=MO-GroupPolicy-v1
num_layers=3
hidden_dim=768
for penalty in 0 0.0001 0.1; do
    for quality in amateur expert; do
        for beta in 10 1 0.1 0.01 0.001 0.0001 0.00001; do
            for seed in 1 2 3 4 5 6 7 8 9 10; do
                echo "Running $penalty $quality  $beta $seed"
                python main.py \
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
                    --total_train_steps 100000 \
                    --log_interval 25000 \
                    --loss_kind fixed-fairdice \
                    --gradient_penalty_coeff $penalty \
                    --save_path "./structured-results/results-discrete-$penalty" \
                    --discrete 1
            done
        done
    done
done