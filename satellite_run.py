import numpy as np
import pandas as pd
from user import *
from beam_init import *
# from LEOSatellite import *
import matplotlib.pyplot as plt  # 约定俗成的写法plt
# from MaxCI import *
# from core import *
from calculateSinr import *

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
np.set_printoptions(linewidth=400)


class Env:
    def __init__(self):
        self.beam, self.lat_log = setInitBeamCenterPos(0, [0, 0, 0], type='IRIDIUM')
        self.maxdistance = setting.user_maxdistance
        self.user_per_beam=setting.user_per_beam
        self.power_discrete=np.array([10,15,20,25])
        self.user_number = self.user_per_beam*len(self.beam)
        self.beam_list = list(range(0, len(self.beam), 1))
        self.userlist = 0
        self.request_list = 0
        self.tti = 1
        self.rbgnumber = setting.rbg
        self.cqi = np.random.randint(15, 16, size=self.user_number)
        self.sbcqi = np.random.randint(15, 16, size=(self.user_number, self.rbgnumber))
        self.RbgMap = np.zeros((len(self.beam_list), self.rbgnumber))
        self.InvFlag = np.random.randint(1, 2, size=(len(self.beam_list), self.user_number))
        self.bler = np.zeros(self.user_number)
        self.current_cqi_reqest = 0
        self.current_bler_request = 0
        self.request_position_xyz_info = 0
        self.cellid = np.random.randint(0, 1, size=(self.user_number))
        self.observation_space = {'Requests': (self.user_number, 15), 'RbgMap': (len(self.beam_list), self.rbgnumber),
                                  'InvFlag': (len(self.beam_list), self.user_number)}
        self.action_space = (self.rbgnumber * len(self.beam_list), self.user_number + 1)
        self.extra_infor = {}
        self.last_tti_state = 0

    def reset(self, on, off):
        self.extra_infor = {}
        self.tti = 1
        self.bler = np.zeros(self.user_number)
        self.cqi = np.random.randint(15, 16, size=self.user_number)
        self.sbcqi = np.random.randint(15, 16, size=(self.user_number, self.rbgnumber))
        self.userlist = initial_all_user(self.maxdistance, self.user_per_beam,self.lat_log, ontime=on, offtime=off)
        ########3
        for i in range(len(self.userlist)):
            self.userlist[i].model2_update(tb=0,capacity=0)
        position_xyz0, position_log_lat0 = get_user_position(self.userlist)
        S0, self.request_list = get_user_traffic_info(self.userlist)
        cat_reqandposition_xyz, beam_number = userconnectsate(position_xyz0, self.beam, self.request_list,
                                                              self.user_number)
        self.request_position_xyz_info = cat_reqandposition_xyz
        # print(self.request_list)
        # S0['cqi'] = self.cqi
        # S0['bler'] = self.bler
        S0['beam_number'] = beam_number
        # S0['sbcqi'] = self.sbcqi.tolist()
        self.last_tti_state = S0
        self.InvFlag = self.generate_InvFlag(S0['beam_number'].to_numpy())
        # print('self.InvFlag', self.InvFlag)
        S_PPO_0 = {'Requests': S0.iloc[:, 0:15].to_numpy().flatten(), 'RbgMap': self.RbgMap.flatten(),
                   'InvFlag': self.InvFlag.flatten()}
        print(S0)
        return S0, S_PPO_0

    def step(self, epoch, action=0):
        self.extra_infor = {}
        last_time_request = self.request_list
        # print(type(last_time_request))
        # last_bler = self.bler[last_time_request]
        # last_cqi = self.cqi[last_time_request]
        action = self.reshape_act_tensor(action, last_time_request)
        #############根据上一时刻采取动作更新下一时刻的bler和cqi#########################
        tb_list, rbg_list, sinr ,capa= get_tx(action, self.request_position_xyz_info)
        # next_bler, next_cqi, sbcqi = calculate_cqi_bler(self.request_position_xyz_info, action)
        # tb_list, rbg_list = get_request_user_tb(next_cqi, action)
        ####################################
        position_xyz, position_log_lat, next_state, self.request_list = updata(self.userlist, tb_list,
                                                                               last_time_request,capa)
        # satuedict, Beam, UeLinkSate = userconnectsate(position_xyz, epoch, self.tti)
        # cat_reqandposition_xyz, beam_number = self.deal_data(Beam, self.request_list, position_xyz, satuedict)
        cat_reqandposition_xyz, beam_number = userconnectsate(position_xyz, self.beam, self.request_list,
                                                              self.user_number)
        self.request_position_xyz_info = cat_reqandposition_xyz
        next_state['beam_number'] = beam_number
        self.last_tti_state.iloc[:, 8] = next_state.iloc[:, 8]
        print(self.last_tti_state)
        self.extra_infor = self.generate_extra_info(self.last_tti_state, rbg_list, last_time_request, tb_list)
        self.last_tti_state = next_state
        self.InvFlag = self.generate_InvFlag(next_state['beam_number'].to_numpy())
        done = False
        S_PPO_next = {'Requests': next_state.iloc[:, 0:15].to_numpy().flatten(),
                      'RbgMap': self.RbgMap.flatten(),
                      'InvFlag': self.InvFlag.flatten()}
        # self.tti += 1
        return next_state, S_PPO_next, self.extra_infor, done

    def generate_extra_info(self, state, rbg_list, req, tb_list):
        beam_user_connectlist = state['beam_number'].to_numpy()
        user_rbgbumber_dict = dict(zip(req, rbg_list))
        print('req', req)
        print('rbg_list', rbg_list)
        print('tb_list', tb_list)
        for i in range(int(max(beam_user_connectlist))):
            enb_info = state[state['beam_number'] == i + 1]
            # print("state[beam_number]", state["beam_number"])
            if enb_info.empty:
                continue
            else:
                index = np.where(beam_user_connectlist == i + 1)
                rbg_number_used = 0
                enb_req_total = len(index[0])
                unassigned_total = 0
                enb_rbg_list = []
                for j in index[0]:
                    rbg_number_used += user_rbgbumber_dict[j]
                    enb_rbg_list.append(user_rbgbumber_dict[j])
                    if user_rbgbumber_dict[j] == 0:
                        unassigned_total += 1
                self.extra_infor['enb' + str(i + 1)] = {'enb':i+1,'enb_req_total': enb_req_total,
                                                        'unassigned_total': unassigned_total,
                                                        'number_of_rbg_nedded': enb_info['number_of_rbg_nedded'].sum(),
                                                        'rbg_used': rbg_number_used,
                                                        'newdata': enb_info['newdata'].sum(),
                                                        'waitingdata': enb_info['waitingdata'].sum(),
                                                        'last_time_txdata': enb_info['last_time_txdata'].sum(),
                                                        # 'time_duration': enb_info['time_duration'].sum(),
                                                        'total_txdata': enb_info['total_txdata'].sum(),
                                                        'average_throughput': enb_info['average_throughput'].sum(),
                                                        'rbg_usable': self.rbgnumber,'capacity':enb_info['capacity'].sum()}
        # print(self.extra_infor)
        return self.extra_infor

    def printposition_xyz(self):
        for i in range(len(self.userlist)):
            print('user{0} position_xyz{1}'.format(i, self.userlist[i].position_xyz))

    def generate_InvFlag(self, data):
        flag = np.random.randint(1, 2, size=(len(self.beam_list), self.user_number))
        for i in range(len(self.beam_list)):
            b = np.where(data == i + 1)
            flag[i][b] = 0
        return flag


    def reshape_act_tensor(self, act, request_list):
        act_matrix = np.zeros((len(request_list), self.rbgnumber), dtype='int64')
        assert len(act.shape) == 1, "act维度不为(x,)"
        for i in range(len(request_list)):
            index = np.where(act == request_list[i] + 1)
            index = index[0]
            for y in range(len(index)):
                act_matrix[i][index[y] % self.rbgnumber] = 1
        # print('power action',act)
        # print('act_matrix',act_matrix)
        return act_matrix


if __name__ == '__main__':
    env = Env()
    on = 10
    off = 1
    S0, _ = env.reset(on, off)
    beam_new = np.zeros(12)

    for i in range(100):
        new_ = []
        action = np.zeros(72)
        next_s, _, _, _ = env.step(0, action)
        for i in range(12):
            beam_new = next_s[next_s['beam_number'] == i + 1]['newdata'].sum()
            new_.append(beam_new)
        new_ = np.array(new_)
        beam_new = beam_new + new_
    c = beam_new.sum()
    print(c)
    d = beam_new / c
    print(d)
    power = 3794.7331922020558
    beam_power = d * power
    print(beam_power)
    db_beam = 10 * np.log10(beam_power)
    print(db_beam)
    print(beam_new)