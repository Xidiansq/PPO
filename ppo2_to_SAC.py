# -*-coding:utf-8-*-
from copy import deepcopy
import itertools
import random
from tkinter import Variable
import numpy as np
from pandas import Categorical
import torch
from torch.optim import Adam
from tqdm import tqdm
import time
import core2_to_SAC
import json
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
#################实时动态图################
import matplotlib.pyplot as plt
import calculateSinr
import satellite_run
import setting
from sklearn import preprocessing

import torch.nn.functional as F
import collections
import torch.distributions as dist

ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()  # 开启一个画图的窗口


####多幅子图
# plt.figure()
# plt.subplot(2,1,1)

class ReplayBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, rbg_dim, inv_dim, max_req, act_dim, size, gamma=0.99, lam=0.95):
        self.max_req = max_req
        self.obs_buf = np.zeros(core2_to_SAC.combined_shape(size, obs_dim), dtype=np.float32)
        self.rbg_buf = np.zeros(core2_to_SAC.combined_shape(size, rbg_dim), dtype=np.int) #rbg占用情况
        self.inv_buf = np.zeros(core2_to_SAC.combined_shape(size, inv_dim), dtype=np.int) #无效请求标志位（掩码）
        self.obs2_buf = np.zeros(core2_to_SAC.combined_shape(size, obs_dim), dtype=np.float32)
        self.rbg2_buf = np.zeros(core2_to_SAC.combined_shape(size, rbg_dim), dtype=np.int) #rbg占用情况
        self.inv2_buf = np.zeros(core2_to_SAC.combined_shape(size, inv_dim), dtype=np.int) #无效请求标志位（掩码）
        self.act_buf = np.zeros(core2_to_SAC.combined_shape(size, act_dim*325), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.tti2ptr = {}
        self.gamma, self.lam = gamma, lam
        self.buffer = collections.deque(maxlen = size)

    def store_pending(self, tti, obs, act, obs2):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.tti2ptr[tti] = (obs, act, obs2)

    def reward_throughput(self, capacity, txBytes, rbg_usable, rbg_used):
        # x1 = txBytes*8*1000/1e6  #Mbps
        if capacity == 0:
            return 0
        else:
            x1 = txBytes / capacity
            # if x1 == 1:
            #     print(txBytes)
            #     print(capacity)
        # x2 = rbg_used / rbg_usable
        return x1

    def reward_fairness(self, unassign, req_ue):

        # 如果是从资源块的角度来计算的话，侧重公平性用log10处理,侧重吞吐量用loge或者log2处理
        # 如果是从请求用户的角度来计算的话，直接用比值
        if unassign == 0:
            return 1
        else:
            # x1 = np.log(sat_ue/req_ue)
            # return -1/x1req_ue - unassign
            if req_ue - unassign == 0:
                return 0
            else:
                return (req_ue - unassign) / req_ue

    def get_reward(self, capacity, txBytes, rbg_usable, rbg_used, req, unassign, rbg_needed):

        r3 = self.reward_throughput(capacity, txBytes, rbg_usable, rbg_used)
        # r2 = 1
        # r1=txBytes/rbg_used/capacity
        r1 = self.reward_fairness(unassign, req)
        # r=per_rbg_utility
        r2 = min(rbg_used, rbg_usable) / rbg_usable
        # r1=peruitily
        # r = self.reward_fairness(unassign, req)
        # r = r1
        # r=r1
        # r_bler_rbg_used = (1 - bler)
        # r = r_bler_rbg_used * r2 *0.3+0.7*r3
        r = r3
        # r=r2*r1
        return r, r1, r2
    
    def store(self, obs, act, rew, next_obs, done):
        # buf.store(o, a, tti_reward, next_o, d)

        # obs1 = obs['Requests']
        # obs2 = next_obs['Requests']
        # # self.obs_buf[self.ptr] = obs['Requests']
        # rbg1 = obs['RbgMap']
        # rbg2 = next_obs['RbgMap']
        # inv1 = obs['InvFlag']
        # inv2 = next_obs['InvFlag']
        
        # act = act
        # rew = rew
        # done = done
        # self.buffer.append((obs1, rbg1, inv1, act, rew, obs2, rbg2, inv2, done))
        # self.ptr += 1
        
        if self.ptr < self.max_size:  # buffer has to have room so you can store
            self.obs_buf[self.ptr] = obs['Requests']  
            self.rbg_buf[self.ptr] = obs['RbgMap']
            self.inv_buf[self.ptr] = obs['InvFlag']  

            self.obs2_buf[self.ptr] = next_obs['Requests']
            self.rbg2_buf[self.ptr] = next_obs['RbgMap']
            self.inv2_buf[self.ptr] = next_obs['InvFlag']
            
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.ptr += 1
            # self.size = min(self.size+1, self.max_size)
            return True
        else:
            raise IndexError()
            return False

    def __len__(self):
        return self.ptr
    
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, rbg=self.rbg_buf, inv=self.inv_buf,
                    obs2=self.obs2_buf, rbg2=self.rbg2_buf, inv2=self.inv2_buf,
                    act=self.act_buf, ret=self.rew_buf, d=self.done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()} #转为tenosr
    
    def sample_batch(self, batch_size):
        # transitions = random.sample(self.buffer,batch_size)
        # obs1, rbg1, inv1, act, rew, obs2, rbg2, inv2, done = zip(*transitions)
        # print(self.max_size)
        # print(batch_size)
        # input()
        idxs = np.random.randint(0, self.ptr, size=batch_size)
        batch = dict(obs = self.obs_buf[idxs],
                     rbg = self.rbg_buf[idxs],
                     inv = self.inv_buf[idxs],
                     obs2 = self.obs2_buf[idxs],
                     rbg2 = self.rbg2_buf[idxs],
                     inv2 = self.inv2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
   


def sac(env_fn, actor_critic=core2_to_SAC.RA_ActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=50, epochs=2000, replay_size=int(1e6), gamma=0.98, 
        actor_lr=1e-3, critic_lr=1e-2, alpha=0.01, alpha_Ir=1e-2, target_entropy=-1, tau=0.005,
        batch_size=128, update_after=500,
        max_ep_len=50, logger_kwargs=dict(), save_freq=1, use_cuda=True):
    #replay_size 缓冲区的大小
    #gamma 参数
    #polyak 目标网络的平滑系数 提高算法稳定性和收敛速度
    #alpha 是sac算法里的温度系数（熵系数）
    #batch_size 从缓冲区中取出的样本数量
    # update_after  在sac算法那更新之前  必须保证buffer中有的元素数
    #update_every  每隔多少步执行一次策略和值函数的更新
    #num_test_episodes每一个测试迭代中执行的测试轨迹数量
    #max_ep_len 每个轨迹最大长度（步数）
    
# def ppo(env_fn, actor_critic=core2_to_SAC.RA_ActorCritic, ac_kwargs=dict(), seed=0,
#         steps_per_epoch=200, epochs=1000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
#         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=200,
#         target_kl=0.01, logger_kwargs=dict(), save_freq=10, use_cuda=True):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    XUNHUAN=core2_to_SAC.NETCHAGE
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs) #初始化日志
    print("sadsadasdsa",logger)
    logger.save_config(locals())

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn.Env()
    use_cuda = use_cuda and torch.cuda.is_available()
    print('use_cuda', use_cuda)

    # assert isinstance(env.observation_space, gym.spaces.Dict)
    max_req = env.observation_space["Requests"][0]
    obs_dim = np.prod(env.observation_space["Requests"]) #324 15
    rbg_dim = np.prod(env.observation_space["RbgMap"]) #27 12
    inv_dim = np.prod(env.observation_space["InvFlag"]) # 27
    act_dim = env.action_space[0]
    
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    #获取环境动作空间的限制范围,并假设所有维度共享同一个限制范围。
    # act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac_kwargs['use_cuda'] = use_cuda
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_target = deepcopy(ac)
    # Sync params across processes
    sync_params(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    #目标网络的参数需要定期与主网络的参数同步
    #但不是通过常规的梯度下降来更新，而是通过一种叫做"Polyak平均"的方法。
    for p in ac_target.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    #将主Q网络（ac.v1）和目标Q网络（ac.v2）的所有可训练参数合并到一个参数列表中（q_params）
    # 以便在深度强化学习算法中使用这些参数进行操作，例如参数更新或其他优化相关的操作。
    q_params = itertools.chain(ac.v1.parameters(), ac.v2.parameters())    

    # 使用alpha的log值,可以使训练结果比较稳定
    log_alpha = torch.tensor(np.log(alpha), dtype=torch.float)
    log_alpha.requires_grad = True  # 可以对alpha求梯度
    log_alpha_optimizer = torch.optim.Adam([log_alpha],lr=alpha_Ir)
    # target_entropy = target_entropy  # 目标熵的大小
    target_entropy = 0.98*(-np.log(1/act_dim))


    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())  # 即steps_per_epoch
    print('num_procs()', num_procs())
    # Experience buffer
    buf = ReplayBuffer(obs_dim, rbg_dim, inv_dim, max_req, act_dim, replay_size)

    # Count variables
    var_counts = tuple(core2_to_SAC.count_vars(module) for module in [ac.pi, ac.v1, ac.v2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    # input()
    
    # buf = PPOBuffer(obs_dim, rbg_dim, inv_dim, max_req, act_dim, local_steps_per_epoch, gamma, lam)

    #计算概率分布的熵的函数，衡量不确定性或随机性的指标，评估一个概率分布的混乱程度。
    def entropy(dist):
        min_real = torch.finfo(dist.logits.dtype).min
        logits = torch.clamp(dist.logits, min=min_real)
        p_log_p = logits * dist.probs
        return -p_log_p.sum(-1)

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(rewards, obs2,rbg2,inv2, dones):
        a2,logp_a2, pi = ac.step(obs2, rbg2, inv2, index=True)
        a2 = a2.view(batch_size,-1)
        entropy = -logp_a2
        inp2 = torch.cat((obs2, rbg2, inv2, a2), dim=1)
        q1_value = ac.v1_target(inp2)
        q2_value = ac.v2_target(inp2)
        # min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
        #                        dim=1,
        #                        keepdim=True)
        min_qvalue = torch.min(q1_value, q2_value)
        min_qvalue = min_qvalue.reshape((batch_size,1))
        rewards = rewards.reshape((batch_size,1))
        dones = dones.reshape((batch_size,1))
        # print("min_qvalue:",min_qvalue)
        # print(min_qvalue.shape)
        # print("log_alpha:",log_alpha)
        # print(log_alpha.shape)
        # print("entropy",entropy)
        # print(entropy.shape)
        next_value = min_qvalue + log_alpha.exp() * entropy
        # print("next_value:",next_value)
        # print(next_value.shape)
        # print("rewards: ",rewards)
        # print(rewards.shape)
        # print(gamma)
        # next_value = next_value.reshape(-1)
        td_target = rewards + gamma * next_value * (1 - dones)
        return td_target
    
    def soft_update(net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    pi_optimizer = Adam(ac.pi.parameters(), lr=actor_lr)
    v1_optimizer = Adam(ac.v1.parameters(), lr=critic_lr)
    v2_optimizer = Adam(ac.v2.parameters(), lr=critic_lr)
    def update():
        data = buf.sample_batch(batch_size)
        obs, rbg, inv = data['obs'].float(), data['rbg'].float(), data['inv'].float()
        next_obs, next_rbg, next_inv = data['obs2'].float(), data['rbg2'].float(), data['inv2'].float()
        actions, dones, ret = data['act'].int(), data['done'], data['rew'].float()
        # print(actions)
        # print(actions.shape)
        # print(type(actions))
        # input()
        # action = torch.zeros((32,324,325),dtype=np.int)
        # for batch in range(32):
        #     for i in range(324):
        #         value = actions[batch,i]
        #         if(value !=0):
        #             action[batch,value-1,i+1] =1
        # actions = action
        if use_cuda:
            obs, rbg, inv, ret = obs.to(device), rbg.to(device), inv.to(device), ret.to(device)
            next_obs, next_rbg, next_inv = next_obs.to(device), next_rbg.to(device), next_inv.to(device)
            actions, dones = actions.to(device,dtype=torch.int64), dones.to(device)
        inp = torch.cat((obs, rbg, inv, actions), dim=1)

        # 更新两个Q网络
        td_target = calc_target(ret, next_obs,next_rbg,next_inv, dones)

        critic_1_q_values = ac.v1(inp)
        # print(actions.shape)
        # print(critic_1_q_values)
        # print(critic_1_q_values.shape)
        # print("============")
        # print(td_target)
        # # print(td_target.detach())
        # print(td_target.shape)
        # input()
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        # critic_2_q_values = ac.v2(inp).gather(1, actions)
        critic_2_q_values = ac.v2(inp)
        
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        v1_optimizer.zero_grad()
        critic_1_loss.backward()
        v1_optimizer.step()
        for name, param in ac.v2_target.named_parameters():  
            if param.requires_grad:
                print("aaaa",name)
                print("bbbbb",param.data) 
        v2_optimizer.zero_grad()
        critic_2_loss.backward()
        v2_optimizer.step()

        # 更新策略网络
        a_gs, logpa_gs,_ = ac.step(obs, rbg, inv,index=True)
        a_gs = a_gs.view(batch_size,-1)
        entropy = -logpa_gs
        inp_gs = torch.cat((obs, rbg, inv, a_gs), dim=1)
        # # 直接根据概率计算熵
        # entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = ac.v1(inp_gs)
        q2_value = ac.v2(inp_gs)
        # min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
        #                        dim=1,
        #                        keepdim=True)  # 直接根据概率计算期望
        min_qvalue = torch.min(q1_value, q2_value)
        min_qvalue = min_qvalue.reshape((batch_size,1))
        actor_loss = torch.mean(-log_alpha.exp() * entropy - min_qvalue)
        pi_optimizer.zero_grad()
        actor_loss.backward()
        pi_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - target_entropy).detach() * log_alpha.exp())
        pi_optimizer.zero_grad()
        alpha_loss.backward()
        pi_optimizer.step()

        soft_update(ac.v1, ac.v1_target)
        soft_update(ac.v2, ac.v2_target)


    def action_reshape_toOnehot(actions_dimensional):
        # actions_dimensional = actions_dimensional.cpu().numpy().flatten()
        action_one_hot = np.zeros((len(actions_dimensional),325),dtype=np.int64)
        for i in range(len(actions_dimensional)):
            index = actions_dimensional[i]
            action_one_hot[i][index] = 1
        action_one_hot = action_one_hot.flatten()
        return action_one_hot
    # Prepare for interaction with environment
    ontime = 4
    offtime = 6
    _, o = env.reset(ontime, offtime)
    # reqBytes, txBytes, ackBytes, nakBytes = 0, 0, 0, 0
    # env.simArgs["--epochId"] += 1
    fig, ax = plt.subplots()
    x = []
    y = []
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        pbar = tqdm(total=local_steps_per_epoch)
        # upper_per_rbg = setting.rbgcapa
        ac.eval()
        ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler, ep_rbg_used = 0, 0, 0, 0, 0, 0, 0, 0
        epoch_tx, epoch_capacity, epoch_waiting, epoch_reward, epoch_newbytes, epoch_bler, epoch_rbg_used = 0, 0, 0, 0, 0, 0, 0
        ep_r1, ep_r2 = 0, 0
        sum_tx = 0
        error = 0
        final_waiting = 0
        time_delay=0
        for i_episode in range(local_steps_per_epoch):
        # while len(buf) < local_steps_per_epoch:
            # print(o["InvFlag"])
            obs = torch.as_tensor(o["Requests"], dtype=torch.float32)
            rbg = torch.as_tensor(o["RbgMap"], dtype=torch.int32)
            fla = torch.as_tensor(o["InvFlag"], dtype=torch.int32)
            print('-'*50)
            a,logp, pi = ac.step(obs, rbg, fla)
            print(a)
            info, next_o, extra, d = env.step(epoch, a)
            a_reshape = a.reshape(27, -1)
            print('user_action', a_reshape)
            print(extra)
            # #####################当前的一个obs_tti##################################
            tti_reward = 0
            tti_r1, tti_r2 = 0, 0
            # 每个tti可以传输的字节上限    当前tti待传字节数 与 可传字节数的最小值
            cell_reward_list = []
            waiting_bytes, tx_bytes, capacity_bytes, new_bytes, bler, rbg_used = 0, 0, 0, 0, 0, 0
            for _, cell in extra.items():
                tx_bytes += cell['last_time_txdata']
                # capacity_bytes += rbg_capacity * int(cell['rbg_usable'])
                new_bytes += cell['newdata']
                rbg_used += cell['rbg_used']
                capacity = min(int(cell['waitingdata']),1365197.874680996)
                # capacity=upper_per_rbg * int(cell['rbg_usable'])
                rrr, r_bler_rbg, r_req_unalloca = buf.get_reward(capacity, int(cell['last_time_txdata']),
                                                                 int(cell['rbg_usable']),
                                                                 int(cell['rbg_used']), int(cell['enb_req_total']),
                                                                 int(cell['unassigned_total']),
                                                                 int(cell['number_of_rbg_nedded']))
                # rrr = cell['per_rbg_utility']
                print('reward', rrr)
                cell_reward_list.append(rrr)
                tti_reward += rrr
                tti_r1 += r_bler_rbg
                tti_r2 += r_req_unalloca
            tti_reward = tti_reward / len(cell_reward_list) if len(cell_reward_list) != 0 else 0
            tti_r1 = tti_r1 / len(cell_reward_list) if len(cell_reward_list) != 0 else 0
            tti_r2 = tti_r2 / len(cell_reward_list) if len(cell_reward_list) != 0 else 0
            ep_r1 += tti_r1
            ep_r2 += tti_r2
            ep_tx += tx_bytes
            # ep_waiting += waiting_bytes
            ep_capacity += capacity_bytes
            ep_newbytes += new_bytes
            ep_rbg_used += rbg_used
            ep_bler += bler
            print('tti_reward', tti_reward)
            print('cell_reward_list', cell_reward_list)
            # print(cell_reward_list)
            # print('tti_reward',tti_reward)
            ep_ret += tti_reward
            ep_len += 1
            # buf.execute_pop(reward_tti, tti_reward)
            # buf.store(o, a, v, logp, tti_reward)
            a = action_reshape_toOnehot(a)
            buf.store(o, a, tti_reward, next_o, d)
            pbar.update(1)
            o = next_o


            timeout = ep_len == max_ep_len  # 一个episode
            terminal = timeout or d
            epoch_ended = (len(buf) % 50) == 0
                # print('ep_len', ep_len, 'len(buf)', len(buf))
                # if epoch_ended and not (terminal):
                #     print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                # if timeout or epoch_ended:

                #     # if isinstance(o, gym.spaces.Box):
                #     #     _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                #     # else:
                #     obs = torch.as_tensor(o["Requests"], dtype=torch.float32)
                #     rbg = torch.as_tensor(o["RbgMap"], dtype=torch.int32)
                #     fla = torch.as_tensor(o["InvFlag"], dtype=torch.int32)
                #     _, _, _ = ac.step(obs, rbg, fla)
                # else:
                #     v = 0

                # buf.finish_path(v)  # 一个episode
            epoch_reward += ep_ret
            epoch_capacity += ep_capacity
            epoch_tx += ep_tx
            # epoch_waiting += ep_waiting
            epoch_newbytes += ep_newbytes
            epoch_bler += ep_bler
            epoch_rbg_used += ep_rbg_used
            ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler = 0, 0, 0, 0, 0, 0, 0
            if epoch_ended:
                # 缓存满了，触发更新
                if len(buf) > update_after:#minimal_size
                    update()
                for _, cell in extra.items(): time_delay += cell['time_delay']
                sum_tx = info['total_txdata'].sum()
                final_waiting = info['waitingdata'].sum()
                ep_tx = epoch_tx / int(steps_per_epoch / max_ep_len)
                ep_capacity = epoch_capacity / int(steps_per_epoch / max_ep_len)
                # print("epoch_reward",epoch_reward)
                # print(steps_per_epoch)
                # print(max_ep_len)
                # input()
                ep_ret = epoch_reward / int(steps_per_epoch / max_ep_len)
                ep_waiting = epoch_waiting / int(steps_per_epoch / max_ep_len)
                ep_newbytes = epoch_newbytes / int(steps_per_epoch / max_ep_len)
                ep_bler = epoch_bler / int(steps_per_epoch / max_ep_len)
                ep_rbg_used = epoch_rbg_used / int(steps_per_epoch / max_ep_len)
                if len(y) >= 2:
                    if y[-1] - y[-2] >= 15:
                        error = 11111
                logger.store(Ep_ret=ep_ret, ep_bler=ep_r1, ep_fairness=ep_r2, Ep_tx=ep_tx, EP_new=ep_newbytes,
                                EP_finalwiat=final_waiting,
                                sumtx=sum_tx, Ep_capacity=ep_capacity, Ep_rbgused=ep_rbg_used, Time_delay=time_delay)

                # if True:  # if terminal:
                #     # only save EpRet / EpLen if trajectory finished
                #     logger.store(EpRet=ep_ret, EpLen=ep_len, ReqBytes=reqBytes, TxBytes=txBytes, AckBytes=ackBytes,
                #                  NakBytes=nakBytes, Throughput=ackBytes / capacity_episode)
                _, o = env.reset(ontime, offtime)
                # reqBytes, txBytes, ackBytes, nakBytes, capacity_episode = 0, 0, 0, 0, 0
                # env.simArgs["--epochId"] += 1
                
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)
        x.append(epoch + 1)
        y.append(ep_ret)
        ax.cla()  # clear plot
        ax.plot(x, y, 'r', lw=1)  # draw line chart
        plt.pause(0.1)
        plt.savefig('./liangceng.jpg')
        start_time = time.time()
        # update()
        end_time = time.time()
        print(end_time - start_time)
        logger.log_tabular('epoch       ', epoch)
        logger.log_tabular("ep_ret      ", ep_ret)
        # logger.log_tabular("ep_bler      ", ep_r1)
        # logger.log_tabular("ep_fair      ", ep_r2)
        logger.log_tabular("ep_tx       ", ep_tx)
        logger.log_tabular("newbytes    ", ep_newbytes)
        logger.log_tabular("final_wiating   ", final_waiting)
        # logger.log_tabular('sum_tx      ', sum_tx)
        # logger.log_tabular("ep_capa     ", ep_capacity)
        logger.log_tabular('ep_rbg', ep_rbg_used)
        logger.log_tabular('time_dealy ', time_delay)
        # logger.log_tabular('ep_bler     ', ep_bler)
        # logger.log_tabular("ERROR       ", error)
        logger.dump_tabular()


if __name__ == '__main__':

    from spinup.utils.run_utils import setup_logger_kwargs
    import os
    #gumbel_softmax
    # logits = torch.Tensor([[0.7, 0, 0], [0, 0, 0.1], [0.8, 0.1, 0.1]])
    # print("概率分布形式：",logits)
    # sample = F.gumbel_softmax(logits,tau=0.2,hard=True)
    # print("Gumbel",sample)
    # input()

    # #sample
    # pi = dist.Categorical(logits)
    # # 从多项分布中采样
    # sample = pi.sample()
    # print("sample采样结果形式：",sample)
    # input()
    # q = {0:0,1:0,2:0}
    # for i in range(10000): # 进行一万次采样
    #     t = torch.nn.functional.gumbel_softmax(logits, tau=0.2)
    #     print(t)
    #     input()
    #     q[t] += 1
    # print(q)
    # input()
    trace_dir = os.getcwd() + "/result"
    logger_kwargs = setup_logger_kwargs("sac-ra", data_dir=trace_dir, datestamp=True)
    # ppo(satellite_run,
    #     actor_critic=core2_to_SAC.RA_ActorCritic, ac_kwargs={"hidden_sizes": (128,256,128)},#"hidden_sizes": (128,256, 512, 256)}
    #     steps_per_epoch=50, epochs=1500, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
    #     vf_lr=1e-4, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=50,
    #     logger_kwargs=logger_kwargs, use_cuda=True)
    sac(satellite_run, actor_critic=core2_to_SAC.RA_ActorCritic, ac_kwargs={"hidden_sizes": (128,256,128)}, seed=0, 
        steps_per_epoch=50, epochs=2000, replay_size=int(1e5), gamma=0.98, 
        actor_lr=1e-3, critic_lr = 1e-2, alpha=0.01, alpha_Ir=1e-2, target_entropy=-1, tau=0.005,
        batch_size=128, update_after=200, 
        max_ep_len=50, logger_kwargs=logger_kwargs, save_freq=1, use_cuda=True)