import numpy as np
import random as rand

import gym
from gym import spaces

#from Diskonnect1DEnv import Diskonnect1D

import wandb

wandb.init(name         = "first_training",
           project      = '1D-Diskonnect',
           monitor_gym  = True,
           reinit       = True)


###########


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
        
        info = {'states':  self.board.board,
                'reward':  reward,
                'move_0':  move[0],
                'move_1':  move[1]
                }
                
        self.__log__(info)

        self.curr_step += 1
        self.global_step += 1
        
        return obs, reward, done, {}
    

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
        

###########


class Diskonnect1D():
    
    def __init__(self, length, board=None):
        self.len = length
        self.reset()
        if board != None:
            self.board = board
        
    def reset(self):
        self.board = None
        self.vis_board = None
        self.legal_moves = {-1:[],1:[]}
        self._generate_board_()
        self._gen_legal_moves_()
    
    def _generate_board_(self):
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
            
    def _update_board_(self, player, move):
        is_legal_move = self._check_legal_move_(player, move)
        if is_legal_move == 1:
            self.board[move[0]]          = 0
            self.board[int(sum(move)/2)] = 0
            self.board[move[1]]          = player
            self._gen_legal_moves_()
        return is_legal_move
                    
    def _check_legal_move_(self, player, move):
        if move in self.legal_moves[player]:
            return 1
        else:
            return -1
    
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
                
    def _is_done_(self):
        if self.legal_moves == {-1:[],1:[]}:
            return True
        return False

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

