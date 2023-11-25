from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time

from tqdm import tqdm

import core2
import json
import spinup.algos.pytorch.sac.core as core

from spinup.utils.logx import EpochLogger
#################实时动态图################
import matplotlib.pyplot as plt
import calculateSinr
import satellite_run
import setting
from sklearn import preprocessing

ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()  # 开启一个画图的窗口

####多幅子图
# plt.figure()
# plt.subplot(2,1,1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size,max_req,rbg_dim,inv_dim):
        self.max_req = max_req
        self.rbg_buf = np.zeros(core2.combined_shape(size, rbg_dim), dtype=np.int)  # rbg占用情况
        self.inv_buf = np.zeros(core2.combined_shape(size, inv_dim), dtype=np.int)  # 无效请求标志位
        print("inv_dim=",inv_dim)
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.tti2ptr = {}
        self.ptr, self.size, self.max_size = 0, 0, size

    def store_pending(self, tti, obs, act, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.tti2ptr[tti] = (obs, act, val, logp)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs['Requests']
        self.obs2_buf[self.ptr] = next_obs['Requests']

        # self.obs_buf[self.ptr] = obs['Requests']
        self.rbg_buf[self.ptr] = obs['RbgMap']
        self.inv_buf[self.ptr] = obs['InvFlag']

        
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def reward_throughput(self, capacity, txBytes, rbg_usable, rbg_used):
        # x1 = txBytes*8*1000/1e6  #Mbps
        if capacity == 0:
            return 0
        else:
            x1 = txBytes / capacity
            if x1 == 1:
                print(txBytes)
                print(capacity)
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
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def sac(env_fn, actor_critic=core2.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, use_cuda=True):
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

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # env, test_env = env_fn(), env_fn()
    env, test_env =env_fn.Env(),env_fn.Env()
    use_cuda = use_cuda and torch.cuda.is_available()
    print('use_cuda', use_cuda)
    # obs_dim = env.observation_space.shape
    # act_dim = env.action_space.shape[0]

    max_req = env.observation_space["Requests"][0]
    obs_dim = np.prod(env.observation_space["Requests"])
    rbg_dim = np.prod(env.observation_space["RbgMap"])
    inv_dim = np.prod(env.observation_space["InvFlag"])
    act_dim = env.action_space[0]


    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # ac_targ = deepcopy(ac)
    ac_kwargs['use_cuda'] = use_cuda
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size,max_req,rbg_dim,inv_dim)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    var_counts = tuple(core2.count_vars(module) for module in [ac.pi, ac.q1, ac.q2]) #这块不确定
    # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)


    def entropy(dist):
        min_real = torch.finfo(dist.logits.dtype).min  # 浮点数信息
        logits = torch.clamp(dist.logits, min=min_real)
        p_log_p = logits * dist.probs
        return -p_log_p.sum(-1)
    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d , rbg, inv= data['obs'], data['act'], data['rew'], data['obs2'], data['done'],data['rbg'], data['inv']
        if use_cuda:
            o, a, r, o2 , d , rbg, inv= o.to(device), a.to(device), r.to(device), o2.to(device), d.to(
                device), rbg.to(device), inv.to(device)
            
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o ,rbg , inv= data['obs'],data['rbg'],data['inv']
        if use_cuda:
            o, rbg, inv= o.to(device), rbg.to(device), inv.to(device)
        pi, logp_pi = ac.pi(o,rbg,inv)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(obs,deterministic=False):
        
        return ac.act(obs,deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            ontime = 1
            offtime = 9
            d, ep_ret, ep_len = False, 0, 0
            _,o=test_env.reset(ontime,offtime)
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                info, next_o, extra, d = test_env.step(epoch, a)
                obs = torch.as_tensor(o["Requests"], dtype=torch.float32)

                o, r, d, _ = test_env.step(0,get_action(obs,True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    # o, ep_ret, ep_len = env.reset(), 0, 0

    ontime = 1
    offtime = 9
    _, o = env.reset(ontime, offtime)#这里有问题
    fig, ax = plt.subplots()
    x = []
    y = []
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        

        ac.eval()
        ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler, ep_rbg_used = 0, 0, 0, 0, 0, 0, 0, 0
        epoch_tx, epoch_capacity, epoch_waiting, epoch_reward, epoch_newbytes, epoch_bler, epoch_rbg_used = 0, 0, 0, 0, 0, 0, 0
        ep_r1, ep_r2 = 0, 0
        sum_tx = 0
        error = 0
        final_waiting = 0
        obs = torch.as_tensor(o["Requests"], dtype=torch.float32)
        rbg = torch.as_tensor(o["RbgMap"], dtype=torch.int32)
        fla = torch.as_tensor(o["InvFlag"], dtype=torch.int32)
        print('-' * 50)
        # a, v, logp = ac.step(obs, rbg, fla)
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            # a = env.action_space.sample()
            a= ac.step(obs, rbg, fla)
            print("******************************************")
            # print(a)
            # input()
        # Step the env
        info, o2, extra, d = env.step(t, a)
        a_reshape = a.reshape(27, -1)
        print('user_action', a_reshape)
        print(extra)

        
        # o2, r, d, _ = env.step(a) # observation, reward, terminated, False, info
        # ep_ret += r
        # ep_len += 1

#
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
            capacity = min(int(cell['waitingdata']), 1365197.874680996)
            # capacity=upper_per_rbg * int(cell['rbg_usable'])
            rrr, r_bler_rbg, r_req_unalloca = replay_buffer.get_reward(capacity, int(cell['last_time_txdata']),
                                                                int(cell['rbg_usable']),
                                                                int(cell['rbg_used']), int(cell['enb_req_total']),
                                                                int(cell['unassigned_total']),
                                                                int(cell['number_of_rbg_nedded']), )
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
        print('---------------------tti_reward---------------------------', tti_reward)
        print('---------------------cell_reward_list---------------------', cell_reward_list)
        ep_ret += tti_reward
        ep_len += 1
        #(self, obs, act, rew, next_obs, done)
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, tti_reward, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        epoch_reward += ep_ret
        epoch_capacity += ep_capacity
        epoch_tx += ep_tx
        # epoch_waiting += ep_waiting
        epoch_newbytes += ep_newbytes
        epoch_bler += ep_bler
        epoch_rbg_used += ep_rbg_used
        ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler = 0, 0, 0, 0, 0, 0, 0
        
        if d or (ep_len == max_ep_len):
            sum_tx = info['total_txdata'].sum()
            final_waiting = info['waitingdata'].sum()
            ep_tx = epoch_tx / int(steps_per_epoch / max_ep_len)
            ep_capacity = epoch_capacity / int(steps_per_epoch / max_ep_len)
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
                            sumtx=sum_tx, Ep_capacity=ep_capacity, Ep_rbgused=ep_rbg_used, Ep_bler=ep_bler,
                            Error=error)
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            # o, ep_ret, ep_len = env.reset(), 0, 0
            _, o = env.reset(ontime, offtime)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                

                update(data=batch)



        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            
            x.append(epoch + 1)
            y.append(ep_ret)
            ax.cla()  # clear plot
            ax.plot(x, y, 'r', lw=1)  # draw line chart
            plt.pause(0.1)
            plt.savefig('./liangceng.jpg')
            start_time = time.time()
            end_time = time.time()
            print(end_time - start_time)
            # Test the performance of the deterministic version of the agent.
            # test_agent()

            # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LogPi', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('Time', time.time()-start_time)
            # logger.dump_tabular()
            # logger.log_tabular('epoch       ', epoch)
            
            logger.log_tabular("ep_ret      ", ep_ret)
            logger.log_tabular("ep_bler      ", ep_r1)
            logger.log_tabular("ep_fair      ", ep_r2)
            logger.log_tabular("ep_tx       ", ep_tx)
            logger.log_tabular("newbytes    ", ep_newbytes)
            logger.log_tabular("final_wiating   ", final_waiting)
            # logger.log_tabular('sum_tx      ', sum_tx)
            # logger.log_tabular("ep_capa     ", ep_capacity)
            logger.log_tabular('ep_rbg', ep_rbg_used)
            logger.log_tabular('ep_bler     ', ep_bler)
            # logger.log_tabular("ERROR       ", error)
            logger.dump_tabular()

       

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--hid', type=int, default=256)
    # parser.add_argument('--l', type=int, default=2)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--exp_name', type=str, default='sac')
    # args = parser.parse_args()

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    from spinup.utils.run_utils import setup_logger_kwargs
    import os

    trace_dir = os.getcwd() + "/result"
    logger_kwargs = setup_logger_kwargs("sac-ra", data_dir=trace_dir, datestamp=True)

    torch.set_num_threads(torch.get_num_threads())

   
    sac(satellite_run, actor_critic=core2.MLPActorCritic, ac_kwargs={"hidden_sizes": (128,256,128)}, seed=0, 
        steps_per_epoch=5, epochs=100, replay_size=int(1e3), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, use_cuda=True)
