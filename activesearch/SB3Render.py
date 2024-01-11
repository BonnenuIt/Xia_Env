'''
nohup python TrainContinue.py --overwrite --output_dir logs/debug --gpu --gpu_id 7 > logs/TrainContinue.log 2>&1 &
ps -ef然后可以查看后台运行的东西, kill -9 xxxxx 进程号可以终止，后台的输出可以在 train_our_policy.log里面看
'''

import os
import time
import gymnasium as gym
import numpy as np
# from MyCallback import TensorboardCallback
# import my_env
from stable_baselines3 import PPO

from my_animation import MyAnimation

gym.envs.register(
    id=f"active-search-v0",
    entry_point=f"base:Xia1",
)
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,4,3,5,2"
# config = {"loadModel": "save_model/ppo2_continue_continue", "total_timesteps": 2000000}

# Parallel environments
env = gym.make("active-search-v0")
# model = PPO.load(config["loadModel"], device="cuda:6")
# model.set_env(env)
# model.learn(total_timesteps=config["total_timesteps"], log_interval=1,
#             callback=TensorboardCallback(), reset_num_timesteps=False)
# model.save(config["loadModel"] + '_continue')
# print("Model saved to " + config["loadModel"] + "_continue.zip")


model = PPO.load("models/21/rl_model_800000_steps.zip")

mode = "model"  # model 测试模型 random 随机

obs, info = env.reset()
for i in range(200):
    t1 = time.time()

    if mode == "model":
        action, _states = model.predict(obs, deterministic=False)
        t2 = time.time()
        obs, rewards, dones, truncated, info = env.step(action)

    elif mode == "random":

        dummy_action = env.action_space.sample()    # Box(0.0, 1.0, (4,), float32)
        dummy_action = [13, 12]
        # print(dummy_action)
        obs, rewards, dones, truncated, info = env.step(dummy_action)
    
    t3 = time.time()
    env.render()
    t4 = time.time()
    # print("time =", t4-t3, t3-t2, t2-t1)
    print("total_rewards = ", env.rew_render_total,
          "rewards = ", rewards, "\ttime =", env.time)
    if dones:
        # obs, info = env.reset()
        break

# MyAnimation(num_frame=99, root_path='imgs/', save_path='c2.gif')
