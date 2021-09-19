import os, sys
sys.path.append( os.path.dirname(os.path.dirname((os.path.realpath(__file__)))) )

import time

import gym

from Diskonnect1D.envs import DiskonnectPlayerEnv
from Diskonnect1DEnv import Diskonnect1D

import wandb
#from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env

SEED=20200530


def main():
    board_len = 30
    board = None

    env = gym.make("DiskonnectPlayerEnv-v0", player_piece=-1, length=board_len, board=None)
    env = DummyVecEnv([lambda: env])
    env = VecCheckNan(env, raise_exception=True)
    
    device = 'cpu'
    
    policy = 'MlpPolicy'
    policy_fcn = [int(board_len*2), int(board_len*2)]
    value_fcn  = [int(board_len*2), int(board_len*2)]
    
    policy_kwargs = dict( net_arch=[ dict( pi=policy_fcn, 
                                           vf=value_fcn  ) ] )

    model = PPO(policy            = policy,
                env               = env,
                batch_size        = 128,
                tensorboard_log   = None,
                policy_kwargs     = policy_kwargs,
                verbose           = 0,
                seed              = SEED,
                device            = device)
    
    timesteps = 1e6

    model.learn(total_timesteps     = timesteps,
                tb_log_name         = 'ppo',
                reset_num_timesteps = False)

if __name__=="__main__":
    main()
