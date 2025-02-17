import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from rl_controller.ppo import PPO
from envs.rl_env import RL_Env
import imageio_ffmpeg
import pybullet as p
def test():

    ################## hyperparameters ##################

    max_ep_len = 1000           # max timesteps in one episode
    env = RL_Env(show_gui=True, world="plane")

    state_dim = 27
    action_dim = 8
    action_std = 0.01
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    K_epochs =80
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed

    checkpoint_path = "results/try1/7154_ppo_drone.pth"
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    # camera parameter
    cam_target_pos = [0,0,0]
    cam_distance = 2.0
    cam_yaw, cam_pitch, cam_roll = 50, -30.0, 0
    cam_width, cam_height = 480, 360

    cam_up, cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = [0, 0, 1], 2, 0.01, 100, 60
    vid = imageio_ffmpeg.write_frames('results/vid.mp4', (cam_width, cam_height), fps=30)
    vid.send(None) # seed the video writer with a blank frame
    record =False
    obs = env.reset()
    ep_reward = 0
    for i in range((env.EPISODE_LEN_SEC)*env.SIM_FREQ):
        if i % int((env.SIM_FREQ/env.CTRL_FREQ)) == 0:
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            # record video
            if record:
                cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx)
                cam_projection_matrix = p.computeProjectionMatrixFOV(cam_fov, cam_width*1./cam_height, cam_near_plane, cam_far_plane)
                image = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix)[2][:, :, :3]
                #video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                vid.send(np.ascontiguousarray(image))
        obs, reward, terminated, info = env.step(action)
        ep_reward += reward
        if terminated:
            break
    vid.close()
    print('Episode: {} \t\t Reward: {}'.format(0, round(ep_reward, 2)))

if __name__ == '__main__':
    test()
