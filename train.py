import os, sys
sys.path.append( os.path.dirname(os.path.dirname((os.path.realpath(__file__)))) )

import time

import numpy as np

import gym

from Games.envs import DiskonnectPlayerEnv

import wandb

wandb.init(name         = "multi-level-training-1",
           project      = '1D-Diskonnect',
           monitor_gym  = True,
           reinit       = True)

from stable_baselines3 import PPO, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env

SEED=20200530


def init_model(env, lr=3e-4):
    device = "cpu"
    
    policy = 'MlpPolicy'
    
    length = env.envs[0].len
    policy_fcn = [int(length*2), int(length*2)]
    value_fcn  = [int(length*2), int(length*2)]
    
    policy_kwargs = dict( net_arch=[ dict( pi=policy_fcn, 
                                           vf=value_fcn  ) ] )
    return PPO(policy         = policy,
               env            = env,
               learning_rate  = lr,
               batch_size     = 32,
               policy_kwargs  = policy_kwargs,
               verbose        = 0,
               seed           = SEED,
               device         = device)


def main():
    board_len = 9
    
    timesteps = 2e5
    
    ###
                                           
    board_1 = np.array([-1, 1, 0, 1, 0, 1, 0, 1, -1])

    env_1 = gym.make("DiskonnectPlayerEnv-v0", player_piece=-1, length=board_len, board=board_1)
    env_1 = DummyVecEnv([lambda: env_1])

    print("=== Training first model =====")

    model = init_model(env_1)

    model.learn(total_timesteps     = timesteps,
                tb_log_name         = 'ppo',
                reset_num_timesteps = False)
                  
    state_dict_1 = {
                    'model_state_dict':      model.policy.state_dict(),
                    'optimizer_state_dict':  model.policy.optimizer.state_dict(),
                   }
    
    print("--- Done -----")
    
    ###
    
    board_2 = np.array([0, 1, -1, 1, 0, 1, -1, 1, 0])

    env_2 = gym.make("DiskonnectPlayerEnv-v0", player_piece=-1, length=board_len, board=board_2)
    env_2 = DummyVecEnv([lambda: env_2])

    print("=== Training second model =====")

    model = init_model(env_2)
                
    model.env.envs[0].global_step += timesteps + 704 # not sure why I need to add 704
                  
    model.policy.load_state_dict(           state_dict_1['model_state_dict']     )
    model.policy.optimizer.load_state_dict( state_dict_1['optimizer_state_dict'] )

    model.learn(total_timesteps     = timesteps,
                tb_log_name         = 'ppo',
                reset_num_timesteps = False)
                  
    state_dict_2 = {
                    'model_state_dict':      model.policy.state_dict(),
                    'optimizer_state_dict':  model.policy.optimizer.state_dict(),
                   }
    
    ###
    
    board_3 = np.array([0, 1, 0, 1, -1, 1, 0, 1, 0])
    
    env_3 = gym.make("DiskonnectPlayerEnv-v0", player_piece=-1, length=board_len, board=board_3)
    env_3 = DummyVecEnv([lambda: env_3])
    
    print("=== Training third model =====")
    
    model = init_model(env_3)
                  
    model.env.envs[0].global_step += timesteps * 2 + 704 # not sure why I need to add 704
                  
    model.policy.load_state_dict(           state_dict_2['model_state_dict']     )
    model.policy.optimizer.load_state_dict( state_dict_2['optimizer_state_dict'] )
    
    model.learn(total_timesteps     = timesteps,
                tb_log_name         = 'ppo',
                reset_num_timesteps = False)
                
    state_dict_3 = {
                    'model_state_dict':      model.policy.state_dict(),
                    'optimizer_state_dict':  model.policy.optimizer.state_dict(),
                   }

    print("--- Done -----")
    
    ### 
    
    print("=== Saving model =====")
    
    torch.save( state_dict_3, "diskonnect_model-1" )

if __name__=="__main__":
    main()
