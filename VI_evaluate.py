import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from evaluate_utils import evaluate
import argparse
import gym
import gym_custom
#rc('text', usetex=True)


def main():
    parser = argparse.ArgumentParser(description='FVIN Evaluate')
    parser.add_argument('--model1','-m1', type=str)
    parser.add_argument('--model2','-m2', default=None, type=str)
    parser.add_argument('--env', default='PendulumMod-v0',type=str)
    parser.add_argument('--render', action='store_true',default=False)
    parser.add_argument('--H', default=100, type=int)
    parser.add_argument('--policy', default=None, type=str)
    parser.add_argument('--seed', default=5, type=int)
    args = parser.parse_args()

    envname = args.env
    temp_env = gym.make(envname)
    x_dim = len(temp_env.observation_space.low)
    seed = args.seed
    H = args.H
    pi_H = 15
    pi = args.policy


    fname1 = 'models/' + args.model1 + '.pt'
    label1 = args.model1
    xs1, xs_hat1, us1, cost1 = evaluate(fname1, envname, seed=seed,
                                        pi=pi, pi_H=pi_H, H=H,
                                        render=args.render)
    if args.model2 is not None:
        fname2 = 'models/' + args.model2 + '.pt'
        xs2, xs_hat2, us2, cost2 = evaluate(fname2, envname, seed=seed,
                                        pi=pi, pi_H=pi_H, H=H,
                                        render=args.render)
        label2 = args.model2
    plt.style.use('ggplot')
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(x_dim, 2)

    ax1 = fig.add_subplot(gs[:,1])
    e1 = xs_hat1 - xs1
    plt.plot([], c='black', label=r'Ground Truth')
    plt.plot(np.linalg.norm(e1,axis=1), c='r',label=r'{}'.format(label1))
    if args.model2 is not None:
        e2 = xs_hat2 - xs2
        plt.plot(np.linalg.norm(e2,axis=1), c='c',label=r'{}'.format(label2))
    
    plt.xlabel(r'time step')
    plt.ylabel(r'$l_2$ error')
    plt.legend()
    for i in range(x_dim):
        ax = fig.add_subplot(gs[i,0])
        plt.plot(xs1[:,i], c='black',label=r'Ground Truth')
        plt.plot(xs_hat1[:,i], c='r',label=r'{}'.format(label1))
        if args.model2 is not None:
            plt.plot(xs_hat2[:,i], c='c',label=r'{}'.format(label2))
        plt.ylabel(r'$x_{}$'.format(i))

    
    plt.xlabel(r'time step')
    plt.show()


if __name__ == '__main__':
    main()

