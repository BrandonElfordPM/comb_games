import os, sys
sys.path.append( os.path.dirname(os.path.dirname((os.path.realpath(__file__)))) )

import time

import numpy as np

import gym

from Games.envs import DiskonnectPlayerEnv

import wandb
#from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env

SEED=20200530


def main():
    board = np.array([-1, 1, 0])
    board_len = len(board)

    env_left = gym.make("DiskonnectPlayerEnv-v0", player_piece=-1, length=board_len, board=board)
    env_left = DummyVecEnv([lambda: env_left])

    env_right = gym.make("DiskonnectPlayerEnv-v0", player_piece=1, length=board_len, board=board)
    env_right = DummyVecEnv([lambda: env_right])
    
    device = "cuda"
    
    policy = 'MlpPolicy'
    policy_fcn = [int(board_len*2), int(board_len*2)]
    value_fcn  = [int(board_len*2), int(board_len*2)]
    
    policy_kwargs = dict( net_arch=[ dict( pi=policy_fcn, 
                                           vf=value_fcn  ) ] )

    model = PPO(policy            = policy,
                env               = env_left,
                batch_size        = 32,
                policy_kwargs     = policy_kwargs,
                verbose           = 0,
                seed              = SEED,
                device            = device)
    
    timesteps = 1e4

    model.learn(total_timesteps     = timesteps,
                tb_log_name         = 'ppo',
                reset_num_timesteps = False)


if __name__=="__main__":
    main()
