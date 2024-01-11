# nohup python continue_training.py > logs/SB3Train_2.log 2>&1 &
import gymnasium as gym
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

save_path = "/home/quentinhsia/code/Xv5/models/uav_2_ppl_15_action_27_10_1/"
print(save_path)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=save_path,
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

gym.envs.register(
    id=f"active-search-v0",
    entry_point=f"base:Xia1",
)

config = {"loadModel": "models/uav_2_ppl_15_action_27_10_1/rl_model_1300000_steps",
          "total_timesteps": 10000000}

# Parallel environments
env = gym.make("active-search-v0")
model = PPO.load(config["loadModel"], device="cuda:6")
model.set_env(env)
model.learn(total_timesteps=config["total_timesteps"],
            log_interval=1, reset_num_timesteps=False)
model.save(config["loadModel"] + '_continue')
print("Model saved to " + config["loadModel"] + "_continue.zip")
