import environments
import gymnasium
import numpy as np
import torch

gymnasium.register_envs(environments)


def main():
    print("Hello from fact-2026!")
    print("GPU:", torch.cuda.is_available())
    # env = gymnasium.make("MO-Walker2d-v2")
    env = gymnasium.make_vec("MO-Walker2d-v2", 10)
    state, info = env.reset(seed=0)
    print("Ant info", info)
    print("Obs:", state.shape)
    state, *_, info = env.step(np.zeros((10, 6), dtype=float))
    print("Step info", info)
    print("Obs:", state.shape)
    print("Meta:", env.metadata)


if __name__ == "__main__":
    main()
