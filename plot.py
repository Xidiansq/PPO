# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.ndimage.filters import gaussian_filter1d

plt.style.use('ggplot')
file = 'result/2021-10-31_ppo-raxunhuanshaoliangcenglianjie/progress.txt'
df_news = pd.read_table(file, header=None)
# df_news=df_news.drop(0)
path = './jpg'
try:
    os.makedirs(path)
except:
    pass
x = df_news[:984][0].to_numpy()
y = df_news[:984][7].to_numpy()
z = df_news[:984][8].to_numpy()
r1 = df_news[:984][2].to_numpy()
r2 = df_news[:984][3].to_numpy()
reward = df_news[:984][1].to_numpy()
tx = df_news[:984][4].to_numpy()
new = df_news[:984][5].to_numpy()
throughput = tx / new
fig, ax = plt.subplots(3, 2, figsize=(14, 7))
ax[0, 0].plot(y, c='b', alpha=0.3)
ax[0, 0].plot(gaussian_filter1d(y, sigma=5), c='b', label='rbg_used')
ax[0, 0].set_xlabel('epoch')
ax[0, 0].set_ylabel('rbg_used')
ax[0, 1].plot(z, c='b', alpha=0.3)
ax[0, 1].plot(gaussian_filter1d(z, sigma=5), c='b', label='bler')
ax[0, 1].set_xlabel('epoch')
ax[0, 1].set_ylabel('bler')
ax[1, 0].plot(throughput, c='b', alpha=0.3)
ax[1, 0].plot(gaussian_filter1d(throughput, sigma=5), c='b', label='throughput')
ax[1, 0].set_xlabel('epoch')
ax[1, 0].set_ylabel('throughput')
ax[1, 1].plot(reward, c='b', alpha=0.3)
ax[1, 1].plot(gaussian_filter1d(reward, sigma=5), c='b', label='reward')
ax[1, 1].set_xlabel('epoch')
ax[1, 1].set_ylabel('reward')
ax[2, 0].plot(r1, c='b', alpha=0.3)
ax[2, 0].plot(gaussian_filter1d(r1, sigma=5), c='b', label='fairness')
ax[2, 0].set_xlabel('epoch')
ax[2, 0].set_ylabel('fairness')
ax[2, 1].plot(tx, c='b', alpha=0.3)
ax[2, 1].plot(gaussian_filter1d(tx, sigma=5), c='b', label='tx')
ax[2, 1].set_xlabel('epoch')
ax[2, 1].set_ylabel('tx')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
plt.savefig(os.path.join(path, 'r =循环减少两链接.png'))
plt.show()
