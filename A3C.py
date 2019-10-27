import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
from torch.distributions import Normal
from torch.multiprocessing import *
import gym
import numpy as np


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x *= Sigmoid(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.state_feature = nn.Sequential()
        self.state_feature.add_module("DenseLayer1", Linear(3, 32))
        self.state_feature.add_module("Activation1", ReLU())
        self.state_feature.add_module("DenseLayer2", Linear(32, 16))
        self.state_feature.add_module("Activation2", ReLU())

        self.mu = nn.Sequential()
        self.mu.add_module("mu_Linear", Linear(16, 16))
        self.mu.add_module("mu_Activation", ReLU())
        self.mu.add_module("mu_output", Linear(16, 1))
        self.mu.add_module("mu_output_Activation", Tanh())

        self.sigma = nn.Sequential()
        self.sigma.add_module("sigma_Linear", Linear(16, 16))
        self.sigma.add_module("sigma_Activation", ReLU())
        self.sigma.add_module("sigma_output", Linear(16, 1))
        self.sigma.add_module("sigma_output_Activation", ReLU())

        self.v = nn.Sequential()
        self.v.add_module("v_Linear", Linear(16, 16))
        self.v.add_module("v_Activation", ReLU())
        self.v.add_module("v_output", Linear(16, 1))

    def forward(self, x):
        feature = self.state_feature(x)
        mu = self.mu(feature)
        sigma = self.sigma(feature)
        v = self.v(feature)
        return mu, sigma, v

    def choose_action(self, s):
        mu, sigma, _ = self.forward(s)
        action = Normal(mu.view(1, ).data, sigma.view(1, ).data).sample().numpy()
        action = -2+1e5 if action <= -2 else action
        action = 2-1e5 if action >= 2 else action
        return action

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, v = self.forward(s)
        td_error = v_t - v
        """MSE损失函数"""
        critic_loss = td_error.pow(2)
        actor_loss = -Normal(mu, sigma).log_prob(a) * td_error
        return actor_loss.mean(), critic_loss.mean()


class Worker(Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'work%i' % name
        self.g_ep, self.g_ep_r = global_ep, global_ep_r
        self.gnet = gnet
        self.opt = opt
        self.res_queue = res_queue
        self.lnet = Net()
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        
