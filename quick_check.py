import environments
import gymnasium
import numpy as np
import torch

gymnasium.register_envs(environments)


def main():
    print("Hello from fact-2026!")
    print("GPU:", torch.cuda.is_available())
    # env = gymnasium.make("MO-Walker2d-v2")
    env = gymnasium.make_vec("MO-Walker2d-v2", 3, max_episode_steps=3)
    state, info = env.reset(seed=0)
    print("Meta:", env.metadata)
    print("Ant info", info)
    print("Obs:", state.shape)
    for step in range(4):
        state, _, term, trunc, info = env.step(np.zeros((3, 6), dtype=float))
        print(f"{step}: Step info", info)
        print(f"{step}: Obs:", state.shape)
        print(f"{step}: term:", term)
        print(f"{step}: trunc:", trunc)



if __name__ == "__main__":
    main()
