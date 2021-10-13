import os, sys
sys.path.append( os.path.dirname(os.path.dirname((os.path.realpath(__file__)))) )

import time

import torch

import numpy as np

import gym

from Games.envs import DiskonnectPlayerEnv

import wandb

from stable_baselines3 import PPO, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env

SEED=20200530


def init_model(env, lr=3e-4, batch_size=32):
    # default device
    device = "cpu"
    # default policy
    policy = 'MlpPolicy'
    # using length of board for proportional hidden layer shapes
    length = env.envs[0].len
    policy_fcn = [int(length*2), int(length*2)]
    value_fcn  = [int(length*2), int(length*2)]
    policy_kwargs = dict( net_arch=[ dict( pi=policy_fcn, 
                                           vf=value_fcn  ) ] )
    # using PPO
    return PPO(policy         = policy,
               env            = env,
               learning_rate  = lr,
               batch_size     = batch_size,
               policy_kwargs  = policy_kwargs,
               verbose        = 0,
               seed           = SEED,
               device         = device)


def train_level(player_piece, logging, timesteps, board_len, boards=[], level_num=0, state_dict=None):
    # build environment 
    env = gym.make("DiskonnectPlayerEnv-v0", player_piece=player_piece, length=board_len, boards=boards, logging=logging)
    env = DummyVecEnv([lambda: env])
    print("=== Training model {} =====".format(level_num))
    # building policy model
    model = init_model(env)
    # if continuing training, need to configure wandb to continue logging
    if level_num > 0:
        model.env.envs[0].global_step += int( ( timesteps + 720 ) * level_num)  # not sure why I need to add 720
    # if continuing, load in state dict
    if state_dict is not None:
        model.policy.load_state_dict(           state_dict['model_state_dict']     )
        model.policy.optimizer.load_state_dict( state_dict['optimizer_state_dict'] )
    # train model
    model.learn(total_timesteps     = timesteps,
                tb_log_name         = 'ppo',
                reset_num_timesteps = False)
    # save state dict for model and optimizer
    state_dict = {
                    'model_state_dict':      model.policy.state_dict(),
                    'optimizer_state_dict':  model.policy.optimizer.state_dict(),
                   }
    print("--- Done -----")
    return state_dict


def main():
    # train method
    train_method = 'fixed' # can be 'fixed' or 'random'
    # using Wandb to visualize results
    logging = True
    ### default env/hyper params
    board_len = 9
    timesteps = 3e4
    level_num = 0
    ###
    if train_method == 'fixed':
        if logging:
            wandb.init(name         = "multi-level-training-1",
                       project      = '1D-Diskonnect',
                       monitor_gym  = True,
                       reinit       = True)
        boards = [ np.array([0, 1, 0, 1, -1, 1, 0, 1, 0]) ]
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 boards,
                                 level_num=level_num
                                 )
        level_num+=1
        ###
        boards.append( np.array([0, 1, -1, 1, 0, 1, -1, 1, 0]) )
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 boards,
                                 level_num=level_num,
                                 state_dict=state_dict
                                 )
        level_num+=1
        ###
        boards.append( np.array([-1, 1, 0, 1, 0, 1, 0, 1, -1]) )
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 boards,
                                 level_num=level_num,
                                 state_dict=state_dict
                                 )
        ### 
        level_num+=1
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]),
                                 level_num=level_num,
                                 state_dict=state_dict
                                 )
        ### 
        level_num+=1
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 np.array([-1, 1, -1, 1, -1, 1, -1, 1, 0]),
                                 level_num=level_num,
                                 state_dict=state_dict
                                 )
        ### 
        level_num+=1
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 np.array([-1, 0, -1, 1, -1, 1, -1, 0, -1]),
                                 level_num=level_num,
                                 state_dict=state_dict
                                 )
        ### 
        level_num+=1
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 np.array([0, 0, -1, 1, -1, 1, -1, -1, 0]),
                                 level_num=level_num,
                                 state_dict=state_dict
                                 )
        ### 
        level_num+=1
        state_dict = train_level(-1, 
                                 logging,
                                 timesteps,
                                 board_len,
                                 np.array([-1, -1, 0, 1, -1, -1, 1, 0, -1]),
                                 level_num=level_num,
                                 state_dict=state_dict
                                 )
    ###
    else:
        timesteps *= 1e3
        if logging:
            wandb.init(name         = "random-training-1",
                       project      = '1D-Diskonnect',
                       monitor_gym  = True,
                       reinit       = True)
        state_dict = train_level(-1,
                                 logging, 
                                 timesteps,
                                 board_len)
    ###    
    print("=== Saving model =====")
    
    torch.save( state_dict, "diskonnect_model-1" )

if __name__=="__main__":
    main()
