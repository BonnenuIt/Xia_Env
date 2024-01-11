import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from rule import rule

model = PPO.load(
    "models/20/rl_model_450000_steps.zip")


# import activesearch
gymnasium.envs.register(
    id=f"active-search-v0",
    entry_point=f"base:Xia1",
)

env = gymnasium.make("active-search-v0")
obs, info = env.reset()
timesteps = 200*41
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

for _ in range(timesteps):
    dummy_action = env.action_space.sample()    # Box(0.0, 1.0, (4,), float32)
    # dummy_action = [1, 1]
    print(dummy_action)
    observation, rewards, terminated, truncated, info = env.step(dummy_action)

    # env.render()
    # if env.time > 150:
    # print("total_rewards = ", env.rew_render_total,
    #       "rewards = ", rewards, "\ttime =", env.time)
    if terminated:
        if _ < 200+2:
            FoundPeopleInd = np.array(env.FoundPeopleIndPast)[np.newaxis, :]
        else:
            FoundPeopleInd = np.concatenate((np.array(env.FoundPeopleIndPast)[np.newaxis, :], FoundPeopleInd))
        obs, info = env.reset()
        # break


print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
obs, info = env.reset()
for _ in range(timesteps):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, terminated, truncated, info = env.step(action)
    # env.render()
    if terminated:
        if _ < 200+2:
            FoundPeopleInd_ours = np.array(env.FoundPeopleIndPast)[np.newaxis, :]
        else:
            FoundPeopleInd_ours = np.concatenate((np.array(env.FoundPeopleIndPast)[np.newaxis, :], FoundPeopleInd_ours))
        obs, info = env.reset()

for _ in range(41):
    FoundPeopleInd_rule = rule()
    if _ == 0:
        FoundPeopleInd_rules = np.array(FoundPeopleInd_rule)[np.newaxis, :]
    else:
        FoundPeopleInd_rules = np.concatenate((np.array(FoundPeopleInd_rule)[np.newaxis, :], FoundPeopleInd_rules))
    obs, info = env.reset()


import seaborn as sns # 导入模块
sns.set() # 设置美化参数，一般默认就好
print()

rewards = FoundPeopleInd.flatten(order='C')
print(rewards[:300])
episode1 = np.array(range(FoundPeopleInd.shape[1]))
for _ in range(FoundPeopleInd.shape[0]):
    if _ == 0:
        episode = episode1.copy()
        continue
    episode = np.concatenate((episode, episode1))


rewards_ours = FoundPeopleInd_ours.flatten(order='C')
rewards_rules = FoundPeopleInd_rules.flatten(order='C')


plt.figure()
sns.lineplot(x=episode, y=rewards, linestyle="--",)
sns.lineplot(x=episode, y=rewards_ours)
sns.lineplot(x=episode, y=rewards_rules, linestyle="-.",)
plt.xlabel("Timesteps")
plt.ylabel("Fully recovery rate")
plt.savefig("res.png")