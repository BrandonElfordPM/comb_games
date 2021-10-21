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
PREV_TIMESTEPS=0

def init_model(env, policy_kwargs={}):
    lr         = policy_kwargs['lr']         if 'lr'         in policy_kwargs else 3e-4
    batch_size = policy_kwargs['batch_size'] if 'batch_size' in policy_kwargs else 32
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


def test(model, env, board_len):

    i = 0
    obs, _, _ = env.reset()
    while not env.board._is_done_():
        env.board.render(mode='human')
        if i % 2 == 0: # player 1
            action = model.predict(obs)
            obs, _, _ = env.step(action)
        elif i % 2 == 1: # player 2
            move = [-1,-1]
            moving = input("Choose a piece to move:\n")
            try:
                moving = int(moving)
                if (moving > board_len) or (moving <= 0):
                    raise ValueError("Piece position must be on the board! Within [1,...,{}]".format(board_len))
                else:
                    if env.board.board[moving] == 1:
                        raise ValueError("Can't use the opponent's piece")
                    else:
                        move[0] = moving
                        dir = input("Which direction are you moving? (L or R)\n")
                        if (dir == 'L') or (dir == 'l'):
                            landing = moving-2
                        elif (dir == 'R') or (dir == 'r'):
                            landing = moving+2    
                        else:
                            raise ValueError("Invalid direction {} chosen".format(dir))
                    
                        if (landing > board_len) or (landing <= 0):
                            raise ValueError("Move is invalid!")
                        else:
                            middle = (moving+landing)/2
                            if env.board.board[middle] != 1:
                                raise ValueError("Invalid move! Must jump over an opponent's piece.")
                            else:
                                env.board.board[moving]  = 0
                                env.board.board[middle]  = 0
                                env.board.board[landing] = -1

            except:
                raise ValueError("Piece must be an integer!")



def main():
    # train method
    test_method = 'fixed' # can be 'fixed' or 'random'
    # using Wandb to visualize results
    logging = True
    ### default env/hyper params
    board_len = 9

    player_piece = -1

    boards = [ np.array([0, 1, 0, 1, -1, 1, 0, 1, 0]) ]
    
    # build environment 
    env = gym.make("DiskonnectPlayerEnv-v0", player_piece=player_piece, length=board_len, boards=boards, logging=logging)
    env = DummyVecEnv([lambda: env])
    # building policy model
    model = init_model(env)

    f = "Saved_models/diskonnect-model-2"
    state_dict = torch.load(f)

    model.policy.load_state_dict(           state_dict['model_state_dict']     )
    model.policy.optimizer.load_state_dict( state_dict['optimizer_state_dict'] )

    if logging:
        wandb.init(name         = "random-training-1",
                   project      = '1D-Diskonnect',
                   monitor_gym  = True,
                   reinit       = True)
                   
    state_dict = test(model,
                        env,
                        board_len
                        )

if __name__=="__main__":
    main()


file_name = "Saved_models/diskonnect-model-2"