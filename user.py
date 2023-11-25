import numpy as np
import pandas as pd
import math as m
import random
from table import *


class user:
    def __init__(self, maxdistfromorigin, lat_long_areacenter, cbrrate, ontime=10, offtime=2):
        """
        :param maxdistfromorigin: 距离中心点的最大距离单位米
        :param ontime: 单位ms,指数分布的均值
        :param offtime: 单位ms,指数分布的均值
        """
        self.maxdistance = maxdistfromorigin  # 距离波束中心的最大距离
        self.position = np.array([0, 0, 0], dtype="float64")  # 以地心为原点建立的笛卡尔坐标系
        # self.lat_long_areacenter = np.array([0, 0, 0], dtype='float64')  # 波束参数，共三个参数，分别代表纬度，经度，距离地表的海拔高度
        self.lat_long_areacenter = lat_long_areacenter
        self.log_lat_coordinates = np.array([0, 0, 0], dtype="float64")  # 用户参数，共三个参数，分别代表纬度，经度，距离地表的海拔高度
        self.nextjingweiposision = np.array([0, 0, 0], dtype="float64")  # 更新后的经纬度坐标
        # 业务
        self.throughput = 0  # Mbps
        self.request = 0  # 0表示无请求 1表示有请求
        self.ontime = 0  # 业务持续时间
        self.offtime_restore = offtime
        self.offtime = np.random.exponential(offtime)
        self.traffictype = {'text': ontime/4, 'voice': ontime/2, 'video': ontime}  # 业务类型指数分布均值参数
        self.qci_type = {'None': 0, 'text': 1, 'voice': 2, 'video': 3}
        self.qci = 0
        self.waiting_data_finally = 0  # 采取动作后剩余数据
        self.waitingbit = 0  # 当前时刻剩余待传数据
        self.cbrrate = cbrrate  # 单位bit 每毫秒
        self.transmit_rate_one_channal = 10
        self.newarrivaldata = 0  # 新到数据
        self.current_txdata = 0
        self.total_txdata = 0
        self.type = None  # 最终生成的业务类型
        self.number_of_rbg_nedded = 0
        self.max_number_of_rbg = g_nofRbg
        self.current_waiting_data = 0
        self.index = 0
        self.average_throughput = 0.000001  # 1bytes
        self.capacity = 0

        # 随机位置
        self.movespeed = 30  # 用户移动速度 米每秒
        self.earth_radius = 6371000  # 地球半径
        # 由GRS80和WGS84定义的以米为单位的地球半长轴
        self.earth_semimajor_axis = 6378137
        # GRS80定义的地球第一偏心率
        self.earth_grs80_eccentricity = 0.0818191910428158
        # WGS84定义的地球第一偏心率
        self.earth_wgs84_eccentricity = 0.0818191908426215
        self.earthspheroidtype = {'sphere': 0, 'grs80': 1, 'wgs84': 2}  # 三种地球模型
        self.initial_random_position(self.lat_long_areacenter)  # 初始化用户位置函数
        self.movedistance = 0  # 每次更新用户移动距离
        self.randomangle = self.random_angle()  # 产生基于用户速度的经纬度的变化角度
        # 时延
        self.time_delay = 0
        self.height = 35786000
        self.angle_user = 0
        self.capacity = 0
        self.generate_angele_user(self.lat_long_areacenter, self.position, self.height)

    # 产生基于用户速度的经纬度的变化角度
    def random_angle(self):
        direction = np.cos(np.random.uniform(0, math.pi, size=3))
        speed = self.movespeed
        # speed = np.random.uniform(self.movespeed - 10, self.movespeed, size=3)
        randomangle = speed * direction
        randomangle = (randomangle / (2 * np.pi * self.earth_radius)) * 360
        zaxischangerate = np.random.uniform(-1, 1)
        randomangle[2] = 0

        # print(self.speed)
        # print(self.direction)
        # print(self.randomangle)
        return randomangle

    # 更新模型1，每次更新选择随机的移动方向和行进速率
    def model1_update(self, tb, bler, cqi, time_duration=0.001):
        self.randomangle = self.random_angle()
        currentpositionxyz = self.position
        # print(self.randomangle)
        # input()
        # print(speed_vector)
        self.log_lat_coordinates[0] += self.randomangle[0] * time_duration
        self.log_lat_coordinates[1] += self.randomangle[1] * time_duration
        self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
        if self.log_lat_coordinates[2] < 0:
            self.log_lat_coordinates[2] = 0
        userxyz_afterupdate = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                   self.log_lat_coordinates[1],
                                                                   self.log_lat_coordinates[2],
                                                                   self.earthspheroidtype['sphere'])
        areacenterxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                             self.lat_long_areacenter[2],
                                                             self.earthspheroidtype['sphere'])
        user_beamcenter_distance = np.sum(np.square(userxyz_afterupdate - areacenterxyz)) ** 0.5
        if user_beamcenter_distance >= self.maxdistance:
            self.randomangle = -self.randomangle
            self.log_lat_coordinates[0] += self.randomangle[0] * time_duration
            self.log_lat_coordinates[1] += self.randomangle[1] * time_duration
            self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
            if self.log_lat_coordinates[2] < 0:
                self.log_lat_coordinates[2] = 0
            self.position = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                 self.log_lat_coordinates[1],
                                                                 self.log_lat_coordinates[2],
                                                                 self.earthspheroidtype['sphere'])
        self.position = userxyz_afterupdate
        updatepositionxyz = self.position
        self.movedistance = np.sum(np.square(updatepositionxyz - currentpositionxyz)) ** 0.5

        self.traffic_updata(tb, bler, cqi)

    # 更新模型2 按照恒定的行进速率和方向前进，直到到达边界，然后重新调用random_angle函数产生随机方向和距离
    def model2_update(self, tb, capacity, time_duration=0.001):
        currentpositionxyz = self.position
        self.log_lat_coordinates[0] = self.log_lat_coordinates[0] + self.randomangle[0] * time_duration
        self.log_lat_coordinates[1] = self.log_lat_coordinates[1] + self.randomangle[1] * time_duration
        self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
        if self.log_lat_coordinates[2] < 0:
            self.log_lat_coordinates[2] = 0
        userxyz_afterupdate = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                   self.log_lat_coordinates[1],
                                                                   self.log_lat_coordinates[2],
                                                                   self.earthspheroidtype['sphere'])
        areacenterxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                             self.lat_long_areacenter[2],
                                                             self.earthspheroidtype['sphere'])

        user_beamcenter_distance = np.sum(np.square(userxyz_afterupdate - areacenterxyz)) ** 0.5
        if user_beamcenter_distance <= self.maxdistance:
            self.position = userxyz_afterupdate
        else:
            # self.position[0],self.position[1]=self.calcu_intersection(self.log_lat_coordinates)
            while (True):
                self.randomangle = self.random_angle()
                self.log_lat_coordinates[0] = self.log_lat_coordinates[0] + self.randomangle[0] * time_duration
                self.log_lat_coordinates[1] = self.log_lat_coordinates[1] + self.randomangle[1] * time_duration
                self.log_lat_coordinates[2] = self.log_lat_coordinates[2] + self.randomangle[2] * time_duration
                # print('-----------------')
                if self.log_lat_coordinates[2] < 0:
                    self.log_lat_coordinates[2] = 0
                userxyz_afterupdate2 = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                            self.log_lat_coordinates[1],
                                                                            self.log_lat_coordinates[2],
                                                                            self.earthspheroidtype['sphere'])
                user_areacenter_distance2 = np.sum(np.square(userxyz_afterupdate2 - areacenterxyz)) ** 0.5
                if user_areacenter_distance2 <= self.maxdistance:
                    self.position = userxyz_afterupdate2
                    break
        updatepositionxyz = self.position
        self.movedistance = np.sum(np.square(updatepositionxyz - currentpositionxyz)) ** 0.5
        self.traffic_updata(tb, capacity)

    # 随机选择三种业务并按照指数分布随机产生业务的持续时间
    def trafficduration(self):
        type = 'None'
        if self.offtime > 0:
            self.offtime -= 1
            if self.offtime < 0:
                self.offtime = 0
                ################
                traffic_choice = np.random.choice([1, 2, 3])
                if traffic_choice == 1:
                    self.ontime = np.random.exponential(self.traffictype['text'])
                    type = 'text'
                    self.qci = self.qci_type[type]
                elif traffic_choice == 2:
                    self.ontime = np.random.exponential(self.traffictype['voice'])
                    type = 'voice'
                    self.qci = self.qci_type[type]
                else:
                    self.ontime = np.random.exponential(self.traffictype['video'])
                    type = 'video'
                    self.qci = self.qci_type[type]
        elif self.offtime == 0 and self.ontime > 0:
            self.ontime -= 1
            if self.ontime < 0:
                self.ontime = 0
                self.offtime = np.random.exponential(self.offtime_restore)
                self.qci = 0

        return self.ontime

    def waiting_data(self, tb, capacity):
        # rand_number = np.random.uniform(0, 1)
        # if rand_number <= bler:
        #     self.waiting_data_finally = self.waitingbit + self.newarrivaldata
        # else:
        #     self.waiting_data_finally = self.waitingbit + self.newarrivaldata - tb

        self.waiting_data_finally = self.waitingbit + self.newarrivaldata - tb
        if self.request == 1 and self.waiting_data_finally >= 0:
            self.current_txdata = tb
        elif self.request == 1 and self.waiting_data_finally < 0:
            self.current_txdata = self.waitingbit + self.newarrivaldata
        else:
            self.current_txdata = 0

        # if self.request == 1 and rand_number > bler and self.waiting_data_finally >= 0:
        #     self.current_txdata = tb
        # elif self.request == 1 and rand_number > bler and self.waiting_data_finally < 0:
        #     self.current_txdata = self.waitingbit + self.newarrivaldata
        # else:
        #     self.current_txdata = 0
        if self.current_txdata == 0 and self.request == 1:
            self.time_delay += 1
        if self.waiting_data_finally < 0:
            self.waiting_data_finally = 0
        self.throughput=((self.current_txdata/0.001))/(1024**2)
        self.waitingbit = self.waiting_data_finally
        self.newarrivaldata = self.cbrrate * 1 if self.ontime > 1 else self.cbrrate * self.ontime
        self.current_waiting_data=self.waitingbit+self.newarrivaldata
        current_data_total = self.waitingbit + self.newarrivaldata
        if current_data_total > 0:
            self.request = 1
        else:
            self.request = 0
        self.capacity = capacity
        ######判断传输完等待数据所需资源块数目
        self.number_of_rbg_nedded = RbgCountRequired(current_data_total)
        # if self.number_of_rbg_nedded > self.max_number_of_rbg:
        #     self.number_of_rbg_nedded = self.max_number_of_rbg
        ###############################

        self.index+=1
        self.total_txdata = self.total_txdata+self.current_txdata#+ 0.000001 if self.index<=1 else self.total_txdata+self.current_txdata
        self.average_throughput = (self.total_txdata / (self.index / 1000)) / 1024 ** 2 # if self.index>1 else 0.000001

    def traffic_updata(self, tb, capacity):
        self.trafficduration()
        self.waiting_data(tb, capacity)

    # 以波束中心点为中心，随机在self.maxdistance范围内产生经纬度坐标，即用户初始位置
    def initial_random_position(self, beampara):

        originlatitude = beampara[0]
        originlongitude = beampara[1]
        maxaltitude = beampara[2]

        # 除去南北极
        if originlatitude >= 90:
            originlatitude = 89.999
        elif originlatitude <= -90:
            originlatitude = -89.999

        if maxaltitude < 0:
            maxaltitude = 0

        originlatituderadians = originlatitude * (np.pi / 180)
        originlongituderadians = originlongitude * (np.pi / 180)
        origincolatitude = (np.pi / 2) - originlatituderadians

        # 圆心角弧度数的最大值
        a = 0.99 * self.maxdistance / self.earth_radius
        if a > np.pi:
            a = np.pi

        d = np.random.uniform(0, self.earth_radius - self.earth_radius * np.cos(a))
        phi = np.random.uniform(0, np.pi * 2)
        alpha = math.acos((self.earth_radius - d) / self.earth_radius)
        theta = np.pi / 2 - alpha
        randpointlatitude = math.asin(
            math.sin(theta) * math.cos(origincolatitude) + math.cos(theta) * math.sin(origincolatitude) * math.sin(phi))
        intermedlong = math.asin((math.sin(randpointlatitude) * math.cos(origincolatitude) - math.sin(theta)) / (
                math.cos(randpointlatitude) * math.sin(origincolatitude)))
        intermedlong = intermedlong + np.pi / 2
        if phi > (np.pi / 2) and phi <= ((3 * np.pi) / 2):
            intermedlong = -intermedlong
        randpointlongtude = intermedlong + originlongituderadians
        randaltitude = np.random.uniform(0, maxaltitude)

        self.position = self.GeographicTocartesianCoordinate(randpointlatitude * (180 / np.pi),
                                                             randpointlongtude * (180 / np.pi), randaltitude,
                                                             self.earthspheroidtype['sphere'])
        self.log_lat_coordinates = [randpointlatitude * (180 / np.pi), randpointlongtude * (180 / np.pi),
                                    randaltitude]  # 度数为单位
        # print(self.pointposition)
        return self.position, self.log_lat_coordinates

    def generate_angele_user(self, beam_center, user_position, height):
        beam_position = self.GeographicTocartesianCoordinate(beam_center[0], beam_center[1], beam_center[2], self.earthspheroidtype['sphere'])
        beam2user = m.sqrt((beam_position[0]-user_position[0])**2+(beam_position[1]-user_position[1])**2
                           +(beam_position[2]-user_position[2])**2)
        # print("beam2user", beam2user)
        angle_user = m.atan(beam2user/height)
        # print("angle_user", self.angle_user)
        self.angle_user = m.degrees(angle_user)
        # print("angle_user", self.angle_user)
        return self.angle_user


    # 将经纬度坐标转换为笛卡尔坐标系
    def GeographicTocartesianCoordinate(self, latitude, longitude, altitude, sphType):
        latitudeRadians = latitude * m.pi / 180
        longitudeRadians = longitude * m.pi / 180
        # print("longitudeRadians", longitudeRadians)
        # print("latitudeRadians", latitudeRadians)
        # a: semi - major axis of earth
        # e: first eccentricity of earth
        EARTH_RADIUS = 6371e3
        EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
        EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
        EARTH_SEMIMAJOR_AXIS = 6378137
        EARTH_SEMIMAJOR_BXIS = 6356752.3142451793
        if sphType == "SPHERE":
            a = EARTH_RADIUS
            e = 0
        if sphType == "GRS80":
            a = EARTH_SEMIMAJOR_AXIS
            e = EARTH_GRS80_ECCENTRICITY
        else:  # if sphType == WGS84
            a = EARTH_SEMIMAJOR_AXIS
            e = EARTH_WGS84_ECCENTRICITY
        Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of  curvature
        # print("rn", Rn)
        x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
        y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
        z = (Rn + altitude) * m.sin(latitudeRadians)
        # z = ((1 - pow(e, 2)) * Rn + altitude) * m.sin(latitudeRadians)
        cartesianCoordinates = np.array([x, y, z], dtype='float64')
        return cartesianCoordinates

    def get_distance(self):
        areaxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                       self.lat_long_areacenter[2], self.earthspheroidtype['sphere'])
        distance = (np.sum(np.square(self.position - areaxyz))) ** 0.5
        return distance


def updata(user, tb, last_time_request, capacity):
    user_list = user
    tb_list = np.zeros(len(user_list))
    capacity_list = np.zeros(len(user_list))
    # bler_list = np.zeros(len(user_list))
    # cqi_list = last_cqi
    # cqi_list=np.zeros(len(user_list),dtype='int')

    for i in range(len(last_time_request)):
        tb_list[last_time_request[i]] = tb[i]
        capacity_list[last_time_request[i]] = capacity[i]
        # bler_list[last_time_request[i]] = bler[i]
        # cqi_list[last_time_request[i]] = cqi[i]
    for i in range(len(user_list)):
        user_list[i].model2_update(tb_list[i], capacity_list[i])
    user_position_xyz, user_position_log_lat_coordinates = get_user_position(user_list)
    traffic_info, user_request = get_user_traffic_info(user_list)
    return user_position_xyz, user_position_log_lat_coordinates, traffic_info, user_request


def initial_all_user(maxdistance, numofuser,beam_position, ontime=10, offtime=2):
    position=beam_position[:,1:]
    userlist1 = [user(maxdistance, position[0],
                      240000, ontime, offtime) for i in range(numofuser)]
    userlist2 = [user(maxdistance, position[1],
                      350000, ontime, offtime) for i in range(numofuser)]
    userlist3 = [user(maxdistance, position[2],
                      350000, ontime, offtime) for i in range(numofuser)]
    userlist4 = [user(maxdistance, position[3],
                      370000, ontime, offtime) for i in range(numofuser)]
    userlist5 = [user(maxdistance, position[4],
                      390000, ontime, offtime) for i in range(numofuser)]
    userlist6 = [user(maxdistance, position[5],
                      270000, ontime, offtime) for i in range(numofuser)]
    userlist7 = [user(maxdistance, position[6],
                      420000, ontime, offtime) for i in range(numofuser)]
    userlist8 = [user(maxdistance, position[7],
                      350000, ontime, offtime) for i in range(numofuser)]
    userlist9 = [user(maxdistance, position[8],
                      280000, ontime, offtime) for i in range(numofuser)]
    userlist10 = [user(maxdistance, position[9],
                       400000, ontime, offtime) for i in range(numofuser)]
    userlist11 = [user(maxdistance, position[10],
                       440000, ontime, offtime) for i in range(numofuser)]
    userlist12 = [user(maxdistance, position[11],
                       320000, ontime, offtime) for i in range(numofuser)]
    userlist13 = [user(maxdistance, position[12],
                       240000, ontime, offtime) for i in range(numofuser)]
    userlist14 = [user(maxdistance, position[13],
                       340000, ontime, offtime) for i in range(numofuser)]
    userlist15 = [user(maxdistance, position[14],
                       360000, ontime, offtime) for i in range(numofuser)]
    userlist16 = [user(maxdistance, position[15],
                       400000, ontime, offtime) for i in range(numofuser)]
    userlist17 = [user(maxdistance, position[16],
                       360000, ontime, offtime) for i in range(numofuser)]
    userlist18 = [user(maxdistance, position[17],
                       280000, ontime, offtime) for i in range(numofuser)]
    userlist19 = [user(maxdistance, position[18],
                       250000, ontime, offtime) for i in range(numofuser)]
    userlist20 = [user(maxdistance, position[19],
                       240000, ontime, offtime) for i in range(numofuser)]
    userlist21 = [user(maxdistance, position[20],
                       290000, ontime, offtime) for i in range(numofuser)]
    userlist22 = [user(maxdistance, position[21],
                       440000, ontime, offtime) for i in range(numofuser)]
    userlist23 = [user(maxdistance, position[22],
                       420000, ontime, offtime) for i in range(numofuser)]
    userlist24 = [user(maxdistance, position[23],
                       250000, ontime, offtime) for i in range(numofuser)]
    userlist25 = [user(maxdistance, position[24],
                       450000, ontime, offtime) for i in range(numofuser)]
    userlist26 = [user(maxdistance, position[25],
                       340000, ontime, offtime) for i in range(numofuser)]
    userlist27 = [user(maxdistance, position[26],
                       370000, ontime, offtime) for i in range(numofuser)]


    userlist = userlist1 + userlist2 + userlist3 + userlist4 + userlist5 + userlist6 + userlist7 + userlist8 + userlist9 + userlist10 + userlist11 + userlist12+userlist13+ userlist14 + userlist15+ userlist16+ userlist17 + userlist18 + userlist19 + userlist20+ userlist21 + userlist22 + userlist23 + userlist24+userlist25+userlist26+userlist27

    print("len", len(userlist))
    return userlist

# 获取发起业务请求用户的位置和编号
def get_user_position(user):
    ##########初始化用户和泊松分布均值
    userlist = user
    user_position_XYZ = []
    user_position_log_lat_coordinates=[]
    ###############
    #####随机选择len(index)个用户来产生业务，len(index)服从泊松分布
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            # user_positionAndnumber.append((i, userlist[i].log_lat_coordinates))  # 以元组形式将发起业务请求的用户编号和对应的位置存放进列表
            position = userlist[i].position
            position2=userlist[i].log_lat_coordinates
            user_position_XYZ.append(position)  # 只保留位置信息
            user_position_log_lat_coordinates.append(position2)
    ################################
    # for i in range(numofuser):
    #     userlist[i].model2_update()
    #     user_position.append(userlist[i].jingweiposition)
    # print('----',user_positionAndnumber)

    return user_position_XYZ,user_position_log_lat_coordinates


def get_user_log_lat_coordinates(user):
    userlist = user
    user_position_list=[]
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            position=userlist[i].log_lat_coordinates
            user_position_list.append(position)
    return user_position_list


def get_user_traffic_info(user):
    userlist = user
    user_request = []
    traffic_info = []
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            user_request.append(i)
        traffic_info.append(
            (i, userlist[i].log_lat_coordinates[0], userlist[i].log_lat_coordinates[1], userlist[i].log_lat_coordinates[2],
             userlist[i].angle_user, userlist[i].current_waiting_data, userlist[i].newarrivaldata, userlist[i].request,
             userlist[i].current_txdata,  userlist[i].qci,
             userlist[i].number_of_rbg_nedded, userlist[i].total_txdata, userlist[i].throughput, userlist[i].average_throughput,
             userlist[i].capacity))
    traffic_info = np.array(traffic_info, dtype='float')
    traffic_info = pd.DataFrame(traffic_info,
                                columns=['user', 'jing', 'wei', 'gao', 'angle', 'waitingdata', 'newdata', 'request',
                                         'last_time_txdata','qci', 'number_of_rbg_nedded','total_txdata',
                                         'throughput(mbps)', 'average_throughput', 'capacity'])

    return traffic_info, user_request


def get_all_user_position_and_request(user):
    userlist = user
    position_and_req = []
    for i in range(len(userlist)):
        position = userlist[i].position.tolist()
        position_and_req.append((i, position[0], position[1], position[2], userlist[i].request))
    position_and_req = np.array(position_and_req, dtype='float')

    return position_and_req





