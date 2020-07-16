import argparse
import gym
#import gym_lqr
import os
import sys
import pickle
import time
from utils import *
from trpo_models.mlp_policy import Policy
from trpo_models.mlp_critic import Value
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent
from model_env import ModelEnv
import torch

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="Pendulum-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.98)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--linear', action='store_true', default=False,
                    help='use linear policy parameterization')
parser.add_argument('--save_name_ext', metavar='G', default='',
                    help='save_name_extension')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env_ = gym.make(args.env_name)
model = torch.load('models/res_mbtrpo.pt')
env = ModelEnv(env_, model)
state_dim = env.observation_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.linear:
    hidden_size=()
else:
    hidden_size=(64,)

if args.model_path is None:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std, hidden_size=hidden_size)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)


def main_loop():
    avg_reward_n = np.zeros(args.max_iter_num, dtype=np.float32)
    avg_reward_n_eval = np.zeros(args.max_iter_num, dtype=np.float32)
    total_steps = np.zeros(args.max_iter_num, dtype=np.float32)
    log_dict = {'reward':avg_reward_n, 'reward_eval':avg_reward_n_eval, 'total_steps':total_steps}
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, stochastic=True)
        _, log_eval = agent.collect_samples(args.min_batch_size, stochastic=False)
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))
        log_dict['reward'][i_iter] = log['avg_reward']
        log_dict['reward_eval'][i_iter] = log_eval['avg_reward']
        log_dict['total_steps'][i_iter] = (i_iter * args.min_batch_size)
        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(learned_dir(), 'trpo/'+args.save_name_ext + '-{}.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)
        with open('logs/trpo/'+args.save_name_ext+'-{}'.format(args.env_name)+'.pkl','wb')  as f:
            pickle.dump(log_dict, f)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
