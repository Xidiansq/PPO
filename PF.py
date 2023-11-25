import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import pandas as pd
from table import *
import copy
import setting

PfType0AllocationRbg = [10, 26, 63, 110]  # see table 7.1.6.1-1 of 36.213
m_timeWindow = 100
factor = (1.0 / m_timeWindow)


def TxMode2LayerNum(txMode):
    if txMode == 0:  # Tx MODE 1: SISO
        return 1
    if txMode == 1:  # Tx MODE 2: MIMO Tx Diversity
        return 1
    if txMode == 2:  # Tx MODE 3: MIMO Spatial Multiplexity Open Loop
        return 2
    if txMode == 3:  # Tx MODE 4: MIMO Spatial Multiplexity Closed Loop
        return 2
    if txMode == 4:  # Tx MODE 5: MIMO Multi-User
        return 2
    if txMode == 5:  # Tx MODE 6: Closer loop single layer percoding
        return 1
    if txMode == 6:  # Tx MODE 7: Single antenna port 5
        return 1
    else:
        return 0


def GetLayer(rnti):
    # m_uesTxMode= 0 #lte-enb-mac中把传输模式设为0
    n_Layer = TxMode2LayerNum(0)

    return n_Layer


def GetWorstCqi(maxreq, alloc):
    n_Layer = TxMode2LayerNum(0)
    worstCqi = []
    # if nLayer
    for i in range(n_Layer):
        worstCqi.append(15)

    if (maxreq.iat[0, -1] == 0):
        for j in range(n_Layer):
            worstCqi[j] = 1  # try with lowest MCS in RBG with no info on channel
        return worstCqi

    sbCqi = maxreq.iat[0, -1]
    for k in range(alloc):
        rbg = alloc
        if (g_nofRbg > rbg):
            for j in range(n_Layer):
                if (g_nofRbg > j):
                    if (sbCqi[j] < worstCqi[j]):
                        worstCqi[j] = sbCqi[j]
                else:
                    # no CQI for this layer of this suband -> worst one
                    worstCqi[j] = 1

        else:
            for j in range(n_Layer):
                worstCqi[j] = 1  # try with lowest MCS in RBG with no info on channel

    return worstCqi


def CalcTbSize(maxreq, alloc):
    worstCqi = GetWorstCqi(maxreq, alloc)
    rbgSize = g_nRbgSize
    tbSize = 0
    for j in range(len(worstCqi)):
        mcs = GetMcsFromCqi(worstCqi[j])
        tbSize += (GetDlTbSizeFromMcs(mcs, 1 * rbgSize) / 8)
    return tbSize


def Rbgmap(rbgsize):
    rbgmap = np.zeros(rbgsize)
    Usefulrbg = np.where(rbgmap == 0)
    UsefulrbgIdex = Usefulrbg[0]
    return rbgmap, UsefulrbgIdex


def GetAchievableRate(req):
    sbCqi = req.iat[0, -1]
    # print(sbCqi)
    achievableRate = 0.0
    n_Layer = GetLayer(req['user'])
    for k in range(n_Layer):
        mcs = 0
        lenwbcqi = 1
        if (lenwbcqi > k):
            mcs = GetMcsFromCqi(sbCqi[k])
        else:
            # no info on this subband -> worst MCS
            mcs = 0
        rbgSize = g_nRbgSize
        achievableRate += ((GetDlTbSizeFromMcs(mcs, rbgSize) / 8) / 0.001)  # = TB size / TTI bit
    # print('achievableRate',achievableRate)

    return achievableRate


def PF(request):
    request = copy.deepcopy(request)
    # print(request)
    # if len(request)==0:
    #     return
    # print(request)
    Availabreq = request[request['request'] == 1]
    # print('Availabreq',Availabreq)
    # print(len(Availabreq))
    Beam = Availabreq['beam_number']
    # print('beam',Beam)
    allocationmap = np.zeros((len(Availabreq), 2))
    allocationmap1 = np.zeros((len(Availabreq), g_nofRbg + 1), dtype='int')
    allocationmap[:, 0] = Availabreq['user']
    allocationmap1[:, 0] = Availabreq['user']
    # print('allocationmap1',allocationmap1)
    # print('allocationmap',allocationmap)
    if pd.isnull(Beam).all() == True:
        # return np.zeros((env.user_number,env.rbgnumber)),np.zeros((env.user_number,2))
        return np.zeros(len(Availabreq)), np.zeros(len(Availabreq))
    Beam1 = np.unique(np.array(Beam))
    # print('beam1',Beam1)
    rbgsize = g_nofRbg
    Availabreq1 = pd.DataFrame(Availabreq,
                               columns=['user', 'number_of_rbg_nedded', 'average_throughput', 'beam_number', 'sbcqi'])
    rbgmap, UsefulrbgIdex = Rbgmap(rbgsize)
    # print('rbgmap',rbgmap,'usefulrbgidex',UsefulrbgIdex)
    for i in range(len(Beam1)):
        newreq = Availabreq1['beam_number'] == Beam1[i]
        # print('newreq',newreq)
        newreq = Availabreq1[newreq].head(10)
        print('newreq', newreq)
        # print('newreq', newreq)
        for j in range(rbgsize):
            if rbgmap[j] == True or newreq.empty:
                continue
            itMax = []
            rcqiMax = 0.0
            for k in range(len(newreq)):  # 每个rbg的分配对所有的用户进行优先级计算
                # print('--------------------------')
                # print(newreq[k:k+1])
                # input()
                achieveableRate = GetAchievableRate(newreq[k:k + 1])  #################################################
                avthroughput = newreq.iat[k, 2]  # average_throughput
                # print('avthroughput',avthroughput)
                if avthroughput == 0:
                    avthroughput = 0.0001
                rcqi = (achieveableRate * 8 / 1e6) / avthroughput  # Mbps
                # print('rcqi',rcqi)
                if rcqi > rcqiMax:
                    rcqiMax = rcqi
                    itMax = newreq[k:k + 1]
            userId = np.array(itMax['user'])
            # print('userId',userId)
            idex = np.where(allocationmap1[:, 0] == userId[0])
            # print('idex',idex)
            # print('idex[0][0]',idex[0][0])
            allocationmap[idex[0][0]][1] += 1
            allocationmap1[idex[0][0]][j + 1] = 1
            # print(allocationmap,'\n',allocationmap1)
            maxreq = newreq['user'] == userId[0]
            maxreq = newreq[maxreq].head(10)
            # print('maxreq',maxreq)
            tbsize = CalcTbSize(maxreq,
                                1)  # 分配给某个ue所占的一个资源的数据量bytes   #################################################
            current = newreq.loc[newreq['user'] == userId[0], 'average_throughput']
            newreq.loc[newreq['user'] == userId[0], 'average_throughput'] = (0.99 * current) + (
                    0.01 * (tbsize / 0.001)) * 8 * 1e-6  # 分配一个资源块后更新该ue的吞吐量 #Mpbs
            if newreq.loc[newreq['user'] == userId[0], 'number_of_rbg_nedded'].item() >= 1:
                newreq.loc[newreq['user'] == userId[0], 'number_of_rbg_nedded'] = newreq.loc[newreq['user'] == userId[
                    0], 'number_of_rbg_nedded'] - 1
                if newreq.loc[newreq['user'] == userId[0], 'number_of_rbg_nedded'].item() == 0:
                    newreq.drop(index=userId[0], inplace=True)
                    print('delete', newreq)
            # print('new after',newreq)
        # print("+++",allocationmap)#
        print(allocationmap1)  #
    #     print(allocationmap1)
    return allocationmap1[:, 1:], allocationmap
