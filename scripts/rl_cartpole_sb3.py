import gymnasium as gym
import warp_envs.warp_gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import os

if not os.path.exists("ppo_cartpole.zip"):
    print("Training policy")

    env = gym.make("WarpSimCartPole-v1", render_mode=None)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)

    model.save("ppo_cartpole")

model = PPO.load("ppo_cartpole", print_system_info=True)
env = gym.make("WarpSimCartPole-v1", render_mode="human")
vec_env = DummyVecEnv([lambda: env])
obs = vec_env.reset()
print("rollout")
#for i in range(1000):
while 1:
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
