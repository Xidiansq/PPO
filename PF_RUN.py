# -*-coding:utf-8-*-
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import time
import core
import json
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
#################实时动态图################
import matplotlib.pyplot as plt
import satellite_run
import table
from PF import PF
from MaxCI import MaxCI
import os
import setting
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs

env = satellite_run.Env()
rbg_capacity = setting.rbgcapa
trace_dir = os.getcwd() + "/PF"
logger_kwargs = setup_logger_kwargs("PF_TEST", data_dir=trace_dir, datestamp=True)
logger = EpochLogger(**logger_kwargs)
onofflist = [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
def run(on,off):
    for on, off in onofflist:
        ontime = on
        offtime = off
        s, _ = env.reset(ontime, offtime)
        for epoch in range(10):
            ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler, ep_rbg_used, ep_len = 0, 0, 0, 0, 0, 0, 0, 0, 0
            for step in range(100):
                print('-' * 100)
                print(s)
                a1, _ = PF(s)
                print(a1)
                next_o, _, extra, d = env.step(a1)
                waiting_bytes, tx_bytes, capacity_bytes, new_bytes, bler, rbg_used = 0, 0, 0, 0, 0, 0
                for _, cell in extra.items():
                    tx_bytes += cell['last_time_txdata']
                    capacity_bytes += rbg_capacity * int(cell['rbg_usable'])
                    new_bytes += cell['newdata']
                    print('new_bytes', new_bytes)
                    rbg_used += cell['rbg_used']
                    bler += cell['bler']
                    # capacity=upper_per_rbg * int(cell['rbg_usable'])
                ep_tx += tx_bytes
                # ep_waiting += waiting_bytes
                ep_capacity += capacity_bytes
                ep_newbytes += new_bytes
                ep_rbg_used += rbg_used
                ep_bler += bler
                ep_len += 1
                s = next_o
                if ep_len == 100:
                    print('ep_newbytes', ep_newbytes)
                    logger.store(Ep_tx=ep_tx, EP_new=ep_newbytes, EP_WAIT=ep_waiting, Ep_rbgused=ep_rbg_used,
                                 Ep_bler=ep_bler, Ep_capacity=ep_capacity)
                    s, _ = env.reset(ontime, offtime)
                    logger.log_tabular('epoch       ', epoch)
                    logger.log_tabular("ep_tx       ", ep_tx)
                    logger.log_tabular("newbytes    ", ep_newbytes)
                    # logger.log_tabular("final_wiating   ", final_waiting)
                    # logger.log_tabular('sum_tx      ', sum_tx)
                    logger.log_tabular('ep_rbg', ep_rbg_used)
                    logger.log_tabular('ep_bler     ', ep_bler)
                    logger.log_tabular("ep_capa     ", ep_capacity)
                    # logger.log_tabular("ERROR       ", error)
                    logger.dump_tabular()
