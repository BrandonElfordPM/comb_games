import numpy as np
import pandas as pd
import random as rand
from copy import deepcopy

import gym
from gym import spaces

#from Diskonnect1DEnv import Diskonnect1D

import wandb


#############################


class DiskonnectPlayerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-float(1.0), float(1.0))

    def __init__(self, player_piece, length, boards=[], logging=False):
        self.action_space = spaces.MultiDiscrete( [ length, 2 ] ) # player piece to move, direction
        self.observation_space = spaces.Box( -np.ones(length), np.ones(length), dtype=np.float32 )

        self.player = player_piece
        self.len = length
        self.boards = boards
        self.board = Diskonnect1D(length=length)
        
        self.global_step=0
        self.curr_step=0
        self.episode_len=length*5
        self.log=logging
        self.reset()
###
    def step(self, action):
        piece_to_move = action[0]
        direction     = action[1]
        move = (piece_to_move, piece_to_move-2 if direction == 0 else piece_to_move+2)
        
        reward = self.board._update_board_(self.player, move)
        reward -= 0.1 # time cost, want to finish the game as soon as possible
        
        self.episode_reward += reward
        
        done = self.board._is_done_()
        if self.board.legal_moves[self.player] == []:
            done = True
            reward += 1.0
        if self.curr_step > self.episode_len:
            done = True
            reward -= 1.0
        obs = self.board.board

        self.curr_step += 1
        self.global_step += 1

        if self.log:
            wandb_board = { str(i): self.board.board[i] for i in range(self.len) }
            self.__log__(wandb_board)
            info = {'states':  self.board.board,
                    'reward':  reward,
                    'move_0':  move[0],
                    'move_1':  move[1],
                    'global_step': self.global_step
                    }
            if done:
                info.update({'episode_reward': self.episode_reward})
            self.__log__(info, commit=True)
        
        return obs, reward, done, {}
###
    def reset(self):
        self.curr_step = 0
        if len(self.boards) == 0:
            self.board._generate_board_()
        elif len(self.boards) == 1:
            self.board.board = deepcopy( self.boards[0] )
        else:
            self.board.board = deepcopy( rand.choice(self.boards) ) 
        self.board.reset()
        self.episode_reward = 0
        return self.board.board
###
    def render(self, mode=None):
        self.board.render(mode)
###
    def __log__(self, info, commit=False):
        wandb.log(info, commit=commit, step=self.global_step)
        

#############################


class Diskonnect1D():
    def __init__(self, length=None, board=None):
        if length != None:
            self.len = length
            self._generate_board_()
        if type(board) != type(None):
            self.board = board
            self.len = len(board)
        self.reset()
###
    def reset(self):
        self.vis_board = None
        self.legal_moves = {-1:[],1:[]}
        self._gen_legal_moves_()
###
    def _generate_board_(self):
        while True:
            # fix the number of players to be in [2?, length-10?]
            num_pieces = int(max((self.len-5)/2, 2))
            
            # always an equal number of players
            self.board = np.zeros(self.len)
            self.open_positions = set(range(self.len))
            
            # player positions
            self.player1_positions = rand.sample(self.open_positions, num_pieces)
            for piece in self.player1_positions:
                self.open_positions.remove(piece)
                self.board[piece] = 1
                
            self.player2_positions = rand.sample(self.open_positions, num_pieces)
            for piece in self.player2_positions:
                self.open_positions.remove(piece)
                self.board[piece] = -1

            self._gen_legal_moves_()
            if not self._is_done_():
                break
###
    def _update_board_(self, player, move):
        is_legal_move = self._check_legal_move_(player, move)
        if is_legal_move == 1:
            self.board[move[0]]          = 0
            self.board[int(sum(move)/2)] = 0
            self.board[move[1]]          = player
            self._gen_legal_moves_()
        return is_legal_move
###
    def _check_legal_move_(self, player, move):
        if move in self.legal_moves[player]:
            return 1
        else:
            return 0
###
    def _gen_legal_moves_(self):
        self.legal_moves = {-1:[],1:[]}
        for idx, ele in enumerate(self.board):
            right_jump_over, right_land, left_jump_over, left_land = None,None,None,None
            if ele == 0:
                continue
            if idx <= 1:
                right_jump_over = self.board[idx+1:idx+2]
                right_land = self.board[idx+2:idx+3]
            elif idx >= self.len-2:
                left_jump_over = self.board[idx-1:idx]
                left_land = self.board[idx-2:idx-1]
            else:
                right_jump_over = self.board[idx+1:idx+2]
                right_land = self.board[idx+2:idx+3]
                left_jump_over = self.board[idx-1:idx]
                left_land = self.board[idx-2:idx-1]
            if (right_land == 0) and (right_jump_over != 0) and (ele != right_jump_over):
                self.legal_moves[ele].append((idx, idx+2))
            if (left_land == 0) and (left_jump_over != 0) and (ele != left_jump_over):
                self.legal_moves[ele].append((idx, idx-2))             
###
    def _is_done_(self):
        if self.legal_moves == {-1:[],1:[]}:
            return True
        return False
###
    def render(self, mode='human'):
        if mode == 'human':
            if self.vis_board == None:
                self.vis_board = [None] * self.len
            for idx in range(self.len):
                if self.board[idx]==-1:
                    self.vis_board[idx] = 'L'
                elif self.board[idx]==1:
                    self.vis_board[idx] = 'R'
                elif self.board[idx]==0:
                    self.vis_board[idx] = '_'
            print(self.vis_board)
        else:
            pass

