
# nohup python SB3Train.py > logs/22.log 2>&1 &
# tensorboard --logdir=tensorboard --port=6005

import gymnasium as gym
import os
import time
import numpy as np
# import my_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/22/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

gym.envs.register(
    id=f"active-search-v0",
    entry_point=f"base:Xia1",
)

config = {"saveModel": "ppo",
          "learning_rate": 0.0005,
          "seed": 88,
          "n_steps": 2048,
          "batch_size": 128,
          "tensorboard_log": "./tensorboard/",
          "total_timesteps": 10000000}

print("当前进程：", os.getpid(), " 父进程：", os.getppid())
print("Time =", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# Parallel environments
env = gym.make("active-search-v0")

model = PPO("MlpPolicy", env=env,
            learning_rate=config["learning_rate"],
            seed=config["seed"],
            # The number of timesteps to run for each environment per update
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            tensorboard_log=config["tensorboard_log"],
            verbose=1,
            device='cuda:6',)

model.learn(
    total_timesteps=config["total_timesteps"], callback=checkpoint_callback)

model.save("save_model/" + config["saveModel"] +
           '_' + str(config["seed"]) +
           '_' + str(config["learning_rate"]))

print("Model Saved")
