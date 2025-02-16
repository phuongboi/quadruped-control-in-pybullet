import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from rl_controller.ppo import PPO
from envs.rl_env import RL_Env

def test():

    ################## hyperparameters ##################

    max_ep_len = 1000           # max timesteps in one episode
    env = RL_Env(show_gui=True, world="plane")

    state_dim = 27
    action_dim = 8
    action_std = 0.1
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    K_epochs =80
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed

    #checkpoint_path = "log_dir/5/3577_ppo_drone.pth"
    #checkpoint_path = "log_dir/6/4436_ppo_drone.pth"
    checkpoint_path = "results/try1/1895_ppo_drone.pth"
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    obs = env.reset()
    ep_reward = 0
    for i in range((env.EPISODE_LEN_SEC)*env.SIM_FREQ):
        if i % int((env.SIM_FREQ/env.CTRL_FREQ)) == 0:
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
        obs, reward, terminated, info = env.step(action)
        ep_reward += reward
        if terminated:
            break
    print('Episode: {} \t\t Reward: {}'.format(0, round(ep_reward, 2)))

if __name__ == '__main__':
    test()
