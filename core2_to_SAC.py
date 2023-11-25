# -*-coding:utf-8-*-
from builtins import print

import numpy as np
import random
import scipy.signal
from gym.spaces import Box, Discrete, Tuple, Dict
import copy
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import setting
import torch.nn.functional as F
import collections

# from util import Encoder, Decoder, padding_mask
NETCHAGE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs, rbgMap, invFlag):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, rbgMap, invFlag, act=None):
        pi = self._distribution(obs, rbgMap, invFlag)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
            # print('logp_a',logp_a)
        return pi, logp_a


class MultiCategoricalActor(Actor):

    def __init__(self, observation, act_dim, hidden_sizes, activation):
        super().__init__()
        self.max_req = observation["Requests"][0]
        self.enb_cnt = observation["RbgMap"][0]
        self.rbg_cnt = observation["RbgMap"][1]
        self.actdim = act_dim
        # print(self.rbg_cnt, '----------')
        # print('actdim', act_dim)
        obs_dim = np.sum([np.prod(v) for k, v in observation.items()])
        # print('obs', observation)
        # print('act_dim', act_dim, 'act_dim.shape', len(act_dim))
        # assert max(act_dim) == min(act_dim)  # 每个元素都相等

        # if idea2:
        #     self.out_dim = act_dim // self.enb_cnt
        #     self.stateshare = nn.Linear(obs_dim, 512)
        #     self.logits_net = mlp([512] + list(hidden_sizes) + [self.out_dim], activation)
        #     self.logits_net1 = mlp([512 + self.out_dim] + list((256, 512, 256)) + [self.out_dim], activation)
        #     self.logits_net2 = mlp([512 + self.out_dim * 2] + list((256, 512, 256)) + [self.out_dim], activation)
        # elif idea2=='FL':
        #     self.out_dim = act_dim
        #     self.logits_net = mlp([self.obs_dim]+[512] + list(hidden_sizes) + list((256, 512, 256))+list((256, 512, 256))+[self.out_dim], activation)
        if NETCHAGE:
            # self.stateshare = nn.Linear(obs_dim, 256)
            self.out_dim = act_dim // self.enb_cnt
            self.logits_net = mlp([obs_dim + self.rbg_cnt*self.enb_cnt] +list(hidden_sizes) + [self.out_dim], activation)
            # print('循环网络结构')
            # print(self.stateshare)
            # print(self.logits_net)
        else:
            self.out_dim = act_dim
            # self.logits_net = mlp([obs_dim] + [1024]+[512]+ list(hidden_sizes) + [self.out_dim], activation)
            self.logits_net = mlp([obs_dim] +list(hidden_sizes) + [self.out_dim], activation)
            # print('非循环网络结构')
            # print(self.logits_net)


    def _distribution(self, obs, rbgMap, invFlag):
        # assert len(obs.shape) < 3
        batch_size = 1 if len(obs.shape) == 1 else obs.shape[0]
        # batch_size = 1
        # 根据rbgMap构造mask
        rm1 = rbgMap.int().reshape(batch_size, -1).unsqueeze(2).expand(-1, -1, self.max_req)
        rm2 = torch.zeros((*rm1.shape[:-1], 1), dtype=torch.int, device=rm1.device)
        rmask = torch.cat((rm2, rm1), 2).bool()

        temp = invFlag.int().reshape(batch_size, self.enb_cnt, -1)
        am1 = temp.unsqueeze(2).expand(-1, -1, self.rbg_cnt, -1)
        am1 = am1.reshape(batch_size, -1, self.max_req)
        am2 = torch.zeros((*am1.shape[:-1], 1), dtype=torch.int, device=am1.device)  # 不分的一列
        amask = torch.cat((am2, am1), 2).bool()
        #########333
        # power_mask1 = torch.zeros(size=(batch_size, self.enb_cnt, 4), dtype=torch.int)
        # power_mask2 = torch.ones(size=(batch_size, self.enb_cnt, self.max_req + 1 - 4), dtype=torch.int)
        # power_mask = torch.cat((power_mask1, power_mask2), dim=2).bool()
        # rmask = torch.cat((power_mask, rmask), dim=1)
        # amask = torch.cat((power_mask, amask), dim=1)
        #########3
        # print('amask',amask)
        inp = torch.cat((obs, rbgMap.float(), invFlag.float()), 0 if len(obs.shape) == 1 else 1)

        if NETCHAGE:
            # s_share = self.stateshare(inp)
            logit_list = []
            # s_share=s_share.clone().detach().requires_grad_(True)
            for i in range(self.enb_cnt):
                act_zero_matrix = torch.zeros(self.rbg_cnt * (self.enb_cnt - i),device=device) if len(obs.shape) == 1 else torch.zeros((obs.shape[0], self.rbg_cnt * (self.enb_cnt - i)),device=device)
                # print('actdim', self.actdim)
                # print('act_zero_matrix', act_zero_matrix.shape)
                s_input = torch.cat((inp, act_zero_matrix), 0 if len(obs.shape) == 1 else 1)
                logit = self.logits_net(s_input)
                act_logit=logit.reshape(-1,self.rbg_cnt,self.max_req+1)
                act_mask=amask[:,i*self.rbg_cnt:(i+1)*self.rbg_cnt,:]
                act_mask2=rmask[:,i*self.rbg_cnt:(i+1)*self.rbg_cnt,:]
                act_logit=act_logit.masked_fill_(act_mask|act_mask2,-np.inf)
                act=Categorical(logits=act_logit).sample().squeeze(0 if len(obs.shape) == 1 else 1)
                # print(act_logit.shape)
                # print(act)
                # input()
                logit_list.append(logit)
                inp = torch.cat((inp, act.float()), 0 if len(obs.shape) == 1 else 1)
            logits = torch.cat(tuple(logit_list), 0 if len(obs.shape) == 1 else 1)
        else:
            logits = self.logits_net(inp)
        logits = logits.reshape(rmask.shape)
        logits = logits.masked_fill_(rmask | amask, -np.inf)
        # print("logits",logits)
        # print(logits.shape)
        # input()
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        # if len(act.shape) == 3:
            # lp = pi.log_prob(act)
            # print("lp",lp)
            # print("lp.shape",lp.shape)
            # input()
        if len(act.shape) == 2:  # 两个维度，第一个维度为batch_size，第二个维度为每个动作的维数
            print("pi:   ",pi)
            lp = pi.log_prob(act)
            # print("lp:  ",lp)
            # print(lp.shape)
            # print(torch.sum(lp,1))
            # input()
            # print('lp_forward',lp)
            # print('lp_forward.sum(lp, 1)',torch.sum(lp, 1))
            return torch.sum(lp, 1)  # 按照行为单位相加
        else:
            return torch.sum(pi.log_prob(act))


class MLPCriticv(nn.Module):

    def __init__(self, observation_space, hidden_sizes, activation, act_dim):
        super().__init__()
        obs_dim = np.sum([np.prod(v) for k, v in observation_space.items()])
        self.v_net = mlp([obs_dim+105300] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs.float()), -1)  # Critical to ensure v has right shape.



# #修改的Q函数  原来输出为V(s)标量 现在要更改为输出所有动作的Q值
# class MLPCritic(nn.Module):

#     def __init__(self, observation_space, hidden_sizes, activation, act_dim):
#         super().__init__()
#         obs_dim = np.sum([np.prod(v) for k, v in observation_space.items()])
#         self.v_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

#     def forward(self, obs):
#         # return self.v_net(obs.float())
#         return torch.squeeze(self.v_net(obs.float()), -1)  # Critical to ensure v has right shape.


# class RA_ActorCritic(nn.Module):

#     def __init__(self, observation_space, action_space,
#                  hidden_sizes=(256, 512, 1024, 512, 256), activation=nn.Tanh, use_cuda=True):
#         super().__init__()
#         # obs=np.prod(observation_space["Requests"].shape)
#         action_dim = np.prod(action_space)
#         self.pi = MultiCategoricalActor(observation_space, action_dim, hidden_sizes, activation)

#         # build value function
#         self.v = MLPCritic(observation_space, (256, 512, 1024, 512, 256), activation)
#         self.use_cuda = use_cuda
#         if use_cuda:
#             self.pi = self.pi.to(device)
#             self.v = self.v.to(device)

#     def step(self, obs, rbg, flag):
#         # print('obs', obs)
#         # print('rbg', rbg)
#         # print('flag', flag)
#         # print('obs shape', obs.shape, 'rbg.shape', rbg.shape, 'flag shape', flag.shape)
#         if self.use_cuda:
#             obs = obs.to(device)
#             rbg = rbg.to(device)
#             flag = flag.to(device)

#         with torch.no_grad():
#             pi = self.pi._distribution(obs, rbg, flag)
#             a = pi.sample()
#             # print('a', a)
#             # print('logitshape', a.shape)
#             logp_a = self.pi._log_prob_from_distribution(pi, a)
#             # print('logp_a'*5, logp_a)
#             # input()
#             inp = torch.cat((obs, rbg.float(), flag.float()), 0 if len(obs.shape) == 1 else 1)
#             # print('inp', inp)
#             # print('inpshape', inp.shape)
#             v = self.v(inp)

#         if self.use_cuda:
#             return a.cpu().flatten().numpy(), v.cpu().numpy(), logp_a.cpu().flatten().numpy()
#         else:
#             return a.flatten().numpy(), v.numpy(), logp_a.flatten().numpy()

#     def act(self, obs):
#         return self.step(obs)[0]



class RA_ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 512, 1024, 512, 256), activation=nn.Tanh, use_cuda=True):
        super().__init__()
        # obs=np.prod(observation_space["Requests"].shape)
        action_dim = np.prod(action_space)
        
        #build policy function
        self.pi = MultiCategoricalActor(observation_space, action_dim, hidden_sizes, activation)

        # build two value function
        self.v1 = MLPCriticv(observation_space, (256, 512, 1024, 512, 256), activation, action_dim)
        self.v2 = MLPCriticv(observation_space, (256, 512, 1024, 512, 256), activation, action_dim)

        #build two traget value function
        self.v1_target = MLPCriticv(observation_space, (256, 512, 1024, 512, 256), activation, action_dim)
        self.v2_target = MLPCriticv(observation_space, (256, 512, 1024, 512, 256), activation, action_dim)
        
        self.use_cuda = use_cuda
        if use_cuda:
            self.pi = self.pi.to(device)
            self.v1 = self.v1.to(device)
            self.v2 = self.v2.to(device)
            self.v1_target = self.v1_target.to(device)
            self.v2_target = self.v2_target.to(device)
    
    def action_reshape_toOnehot(self,actions_dimensional):
        actions_dimensional = actions_dimensional.cpu().numpy().flatten()
        action_one_hot = np.zeros((len(actions_dimensional),325),dtype=np.int64)
        for i in range(len(actions_dimensional)):
            index = actions_dimensional[i]
            action_one_hot[i][index] = 1
        return action_one_hot
    def action_reshape_toDimensional(self,action_one_hot):
        action_one_hot = action_one_hot.squeeze().cpu().detach().numpy()
        len = action_one_hot.shape[0]
        action_dimensional = np.zeros(len, dtype=int)
        for i in range(len):
            row = action_one_hot[i]
            index = np.where(row == 1)[0]
            action_dimensional[i] = index
        return action_dimensional
                
    def step(self, obs, rbg, flag, index = False):
        # print('obs', obs)
        # print('rbg', rbg)
        # print('flag', flag)
        # print('obs shape', obs.shape, 'rbg.shape', rbg.shape, 'flag shape', flag.shape)
        if self.use_cuda:
            obs = obs.to(device)
            rbg = rbg.to(device)
            flag = flag.to(device)
    
        # with torch.no_grad():
        pi = self.pi._distribution(obs, rbg, flag)
        temperature = 0.1
        if index:     
            a = F.gumbel_softmax(pi.logits,temperature,hard=True)
            logp_a = torch.nn.functional.log_softmax(pi.logits, dim=-1)[a.nonzero(as_tuple=True)].sum()
            if self.use_cuda:
                return a, logp_a, pi
            else:
                return a.flatten().numpy(), logp_a.flatten().numpy(), pi.flatten().numpy()
        else:
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            if self.use_cuda:
                return a.cpu().flatten().numpy(), logp_a, pi
            else:
                return a.flatten().numpy(), logp_a.flatten().numpy(), pi.flatten().numpy()
                # print('logp_a'*5, logp_a)
                # input()
            # inp = torch.cat((obs, rbg.float(), flag.float(),a_v), 0 if len(obs.shape) == 1 else 1)
                # print('inp', inp)
                # print('inpshape', inp.shape)
            # v = self.v1(inp)
    
    def act(self, obs):
        return self.step(obs)[0]
    
    def action_reshape(self,allocationmap1,request_List):
        acT = allocationmap1[:, 1:]
        # print("act:",acT)
        # print(len(acT))
        request = request_List
        # print("request",request)
        reshape_act = np.zeros(324,)
        for i in range(len(acT)):
            k = int(request[i] // 12)
            for j in range(setting.rbg):
                if acT[i][j] == 1:
                    reshape_act[(k*12)+j] = request[i]+1
        # print(reshape_act)
        # input()
        reshape_act = reshape_act.astype(int)
        return reshape_act
# def collection_info(logit):
#     file = open('logit.txt', 'a')
#     file.write(str(logit) + '\n')
#     return 0
