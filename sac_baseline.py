import gym
import numpy as np
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from multi_env import MultiEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
import torch
import gym_softreacher
import os
from model_env import ModelEnv




def soft_reward(s, a):
    return -(np.inner(s[:3],s[:3]) + 0.001*np.inner(a,a))

model = torch.load('models/theta_test_u.pt')
env = gym.make('SoftReacher-v0') 
#env = ModelEnv(model, soft_reward, env_, H=100)
log_dir='./sac/cheetah_wexpert_iclr_1_3/'
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env,log_dir)
model = SAC(MlpPolicy, env, verbose=10)
model.learn(total_timesteps=125000, log_interval=10)
model.save('sac_cheetah_wexpert_iclr_1_3')

#del model

#model = SAC.load('sac_cheetah_wexpert_iclr_1_1')
#print(ts2xy(load_results(log_dir),'timesteps'))
#obs = env.reset()
#r_t = 0
#for i in range(250):
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
   #env.render()
#    r_t += rewards
#    print(r_t)
#env.close()
