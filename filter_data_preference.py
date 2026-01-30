import pickle
import os

path = "data/"
new_path = "filtered_data"

configs = ["MO-Swimmer-v2", "MO-Walker2d-v2", "MO-Ant-v2", "MO-HalfCheetah-v2", "MO-Hopper-v2"]

for config in configs:
    for quality in ["expert", "amateur"]:
        for preference_lb, preference_hb in zip([0.2, 0.3, 0.4], [0.8, 0.7, 0.6]):
            content_path = f"{path}{config}/{config}_50000_{quality}_uniform.pkl"
            content = pickle.load(open(content_path, "rb"))

            retain = []
            for item in content:
                preference_weight = item["preference"][0][0]
                if preference_weight >= preference_lb and preference_weight <= preference_hb:
                    continue
                else:
                    retain.append(item)
            
            new_dir = f"{new_path}{preference_lb}-{preference_hb}/{config}/"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            new_content_path = f"{new_dir}{config}_50000_{quality}_uniform.pkl"

            print(f"\n{config} {quality} with removed trajectories between {preference_lb} and {preference_hb}:")
            print(f"Original size: {len(content)}")
            print(f"New size: {len(retain)}")
            print(f"Trajectory removed: {(len(content) - len(retain))/len(content)}")
            
            with open(new_content_path, "wb") as f:
                pickle.dump(retain, f)