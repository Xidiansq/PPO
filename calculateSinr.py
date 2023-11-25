# -*-coding:utf-8-*-
import numpy as np
import setting

power = False


class calculate_tool:
    def __init__(self):
        # power_action=np.array([22.87974038,24.93794295,26.5090619,25.46126201,25.08990887,23.35311763,21.91114878,24.91139132,23.91362996,27.19319921,26.58206721,24.14658395])
        # power_action=np.array([24.51073719,24.90290911,24.71413423,24.62848788,25.73554529,23.93189764,26.35381523,25.46034243,23.79002661,25.29328968,25.71562453,24.16128749])
        # power_action = np.array([25.00950944,24.61750836,25.55550514,24.70035181,24.78942288,24.10023297,25.62994134,25.02212127,23.54360225,25.48291179,26.37034709,24.45415818])
        power_action = np.random.randint(20, 21, size=27)
        self.Gt_sate = 1000
        self.Gr_user = 40
        self.loss_path = -213.49  # DB
        self.PowerT_beam = power_action
        self.user_number = 0
        self.Hgeo = 35786000
        # self.G_peak = 65 - self.PowerT_beam  11.15之前是65
        self.G_peak = 65 - self.PowerT_beam
        self.ones = np.repeat(0.2524, 28)
        self.gama = 0.5
        self.noisy = 2.5118864315095823e-12
        self.bw = 500e6
        self.rbg_number = setting.rbg
        self.rb_bw = self.bw / self.rbg_number
        self.angle = 0

    def get_loss_path(self):
        return self.loss_path

    def get_gain(self, position_info):
        self.user_number = len(position_info)
        beam_number_connect = self.user_number
        theta_matrix = np.zeros((self.user_number, beam_number_connect))
        distance_matrix = np.zeros((self.user_number, beam_number_connect))
        Gain_matrix = np.zeros((self.user_number, beam_number_connect), dtype=np.float)
        # theta_beam_user = []
        for i in range(self.user_number):
            user_position = position_info[i][1:4]
            # beam_position = position_info[i][4:7]
            # distance2 = np.sqrt(np.sum((user_position - beam_position) ** 2))
            # theta_b_u = (np.arctan(distance2 / self.Hgeo) / (2 * np.pi)) * 360
            # theta_beam_user.append(theta_b_u)
            for j in range(beam_number_connect):
                beam_position2 = position_info[j][1:4]
                beam_label = int(position_info[j][-1])
                distance = np.sqrt(np.sum((user_position - beam_position2) ** 2))
                distance_matrix[i][j] = distance
                theta = np.degrees(np.arctan(distance / self.Hgeo))
                theta_matrix[i][j] = theta
                Gain_matrix[i][j] = self.G_peak[beam_label - 1] - (
                        (12 * (10 ** (self.G_peak[beam_label - 1] / 10))) / self.gama) * np.square(
                    theta_matrix[i][j] / (70 * np.pi))
        self.angle = np.diag(theta_matrix)
        print(self.angle)
        # input()
        # print('len',self.angle)
        # print(theta_matrix)
        # print(theta_beam_user)
        # print(distance_matrix)
        # Gain_matrix = self.G_peak - ((12 * (10 ** (self.G_peak / 10))) / self.gama) * np.square(theta_matrix / (70 * np.pi))  #

        # print(Gain_matrix)

        # print('distance', distance_matrix)
        # print('theta_matrix',theta_matrix)
        # print('Gain_matrix', Gain_matrix)
        Gain_matrix = 10 ** (Gain_matrix / 10)
        # print(Gain_matrix)
        # input()
        # print('Gain_matrix',Gain_matrix)
        return Gain_matrix

    def get_sinr(self, action, position_info):
        # print('action',action.shape)
        rbgnumber = action.shape[1]
        Gain_matrix = self.get_gain(position_info)
        sinr_matrix = np.zeros((self.user_number, rbgnumber))
        capa_matrix = np.zeros((self.user_number, rbgnumber))
        # print('action', action)
        for i in range(self.user_number):
            beam_label = int(position_info[i][-1])
            for j in range(rbgnumber):
                if action[i][j] == 0:
                    continue
                else:
                    index = np.where(action[:, j] == 1)
                    Gain_self = 10 * np.log10(Gain_matrix[i][i])
                    power_self = 10 ** ((Gain_self + self.Gr_user + self.loss_path) / 10) * 10 ** (
                            self.PowerT_beam[beam_label - 1] / 10)

                    # print('^^^^^^^^^^^^^^^^^^^^^^^^^',power_self/self.noisy)
                    # print('power_self', power_self)
                    # print('power_self', power_self)
                    if len(index[0]) == 1:
                        sinr = power_self / (self.noisy)
                        sinr_matrix[i][j] = sinr
                        capa_matrix[i][j] = sinr
                        continue
                    index2 = np.where(index[0] == i)
                    other_user_interference_index = np.delete(index[0], index2[0])
                    # print('index', index[0], 'index2', other_user_interference_index)
                    G_r = 10 ** (self.Gr_user / 10)
                    L_p = 10 ** (self.loss_path / 10)
                    interference = 0
                    for k in range(len(other_user_interference_index)):
                        beam_number = int(position_info[other_user_interference_index[k]][-1])
                        # print('__',self.angle[other_user_interference_index[k]])
                        # print('other_user_interference_index[k]',other_user_interference_index[k])
                        if self.angle[other_user_interference_index[k]] < self.ones[beam_number]:
                            inter = 0
                        else:
                            inter = Gain_matrix[i][other_user_interference_index[k]] * 10 ** (
                                        self.PowerT_beam[beam_number - 1] / 10) * G_r * L_p
                        interference += inter
                        # print('interference', interference)
                    sinr = power_self / (self.noisy + interference)
                    capa = power_self / self.noisy
                    # print('interference', interference)
                    # print('power_self', power_self)
                    # print('nosiy', self.noisy)
                    sinr_matrix[i][j] = sinr
                    capa_matrix[i][j] = capa
        print(sinr_matrix)
        # input()
        # print(self.G_peak)
        # print(self.PowerT_beam)
        return sinr_matrix, capa_matrix


def get_tx(action, position_info):
    tool = calculate_tool()
    sinr, capacity = tool.get_sinr(action, position_info)
    # print(sinr)
    tb_matrix = np.log2(sinr + 1) * tool.rb_bw / 1000
    capacity_matrix = np.log2(capacity + 1) * tool.rb_bw / 1000
    # print(tb_matrix)
    tb = np.sum(tb_matrix, axis=1)
    capa = np.sum(capacity_matrix, axis=1)
    # print('tb', tb_matrix)
    rbglist = np.sum(action, axis=1)
    return tb, rbglist, sinr, capa


def RbgCountRequired(cqi):
    # print("cqi", cqi)
    # print("bytes",bytes)
    # data = get_tb(cqi, 1)
    # print("data", data)
    # rbg_need = math.ceil(bytes/data)

    rbg_need = 0
    # print("rbg_need", rbg_need)
    return rbg_need


def get_beam_per_capcaity():
    tool = calculate_tool()
    label = []
    for i in range(12):
        # Gain_self = 10 * np.log10(tool.G_peak[i])
        power_self = 10 ** ((tool.G_peak[i] + tool.Gr_user + tool.loss_path) / 10) * (10 ** (tool.PowerT_beam[i] / 10))
        sinr = power_self / tool.noisy
        print(sinr)
        cap = np.log2(sinr + 1) * tool.bw / 1000
        label.append(cap)

    return label


if __name__ == '__main__':
    label = get_beam_per_capcaity()
    print(label)
