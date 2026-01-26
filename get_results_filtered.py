import os
import numpy as np

results = []

envs = ["MO-Hopper-v2", "MO-Walker2d-v2", "MO-Swimmer-v2", "MO-HalfCheetah-v2", "MO-Ant-v2"]
qualities = ["expert", "amateur"]
betas = [1, 0.1, 0.01, 0.001, 0.0001]
seeds = [1, 2, 3, 4, 5]

path = "rerun_results/old-results/"
files = os.listdir(path)

for file in files:
    folder_path = os.path.join(path, file)

    file = file.split("_")
    env = file[3]
    quality = file[4]
    beta = file[8][4:]
    seed = file[9][4:]

    npy_path = os.path.join(folder_path, "eval", "normalized_returns_step_100000.npy")
    normalized_returns = np.load(npy_path)
    nsw_score = np.mean(np.sum(np.log(normalized_returns), axis=1))

    results.append({
        "env": env,
        "quality": quality,
        "beta": beta,
        "seed": seed,
        "nsw_score": nsw_score
    })

for env in envs:
    for quality in qualities:
        for beta in betas:
            nsw_scores = []

            for seed in seeds:
                for r in results:
                    if (
                        r["env"] == env
                        and r["quality"] == quality
                        and float(r["beta"]) == beta
                        and int(r["seed"]) == seed
                    ):
                        nsw_scores.append(r["nsw_score"])
            
            if len(nsw_scores) == 0:
                continue

            nsw_scores = np.array(nsw_scores)
            mean_nsw = nsw_scores.mean()
            std_nsw = nsw_scores.std()

            print(f"Environment: {env}, Quality: {quality}, Beta: {beta}, NSW: {mean_nsw:.4f}, STD: {std_nsw:.4f}")