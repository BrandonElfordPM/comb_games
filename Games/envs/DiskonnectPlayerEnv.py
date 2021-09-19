import numpy as np

import random as rand

import gym
from gym import spaces

from Diskonnect1DEnv import Diskonnect1D

import wandb

wandb.init(name = "first_training",
           project      = '1D-Diskonnect',
           monitor_gym  = True,
           reinit       = True)


class DiskonnectPlayerEnv(gym.Env):
    
    def __init__(self, player_piece, length, board=None):
        self.action_space = spaces.MultiDiscrete( [ length, 2 ] ) # player piece to move, direction
        self.observation_space = spaces.Box( -np.ones(length), np.ones(length), dtype=np.float32 )
        
        self.player = player_piece
        self.len = length
        
        self.global_step = 0

        if board == None:
            self.board = Diskonnect1D(self.len)
        else:
            self.board = board

        self.reset()
        
        
    def step(self, action):
        piece_to_move = action[0]
        direction     = action[1]
        move = (piece_to_move, piece_to_move-2 if direction == 0 else piece_to_move+2)
        
        reward = self.board._update_board_(self.player, move)
        
        done = self.board._is_done_()
        
        obs = self.board.board
        
        info = {'states': self.board,
                'reward': reward,
                'move':   move
                }
        self.__log__(info)

        self.curr_step += 1
        self.global_step += 1
        
        return obs, reward, done, None
    

    def reset(self):
        self.curr_step = 0
        self.board.reset()
        info = {}
        self.__log__(info, commit=True)
        #self.board.render()
        return self.board.board
    

    def render(self, mode=None):
        self.board.render(mode)
        

    def __log__(self, info, commit=False):
        wandb.log(info, commit=commit, step=self.global_step)

