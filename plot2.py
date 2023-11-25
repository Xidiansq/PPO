# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.ndimage.filters import gaussian_filter1d

plt.style.use('ggplot')
file1 = 'result/2021-11-09_ppo-ra非循环/progress.txt'
file2='result/2021-11-10_ppo-ra/progress.txt'
df_news1 = pd.read_table(file1, header=None,skiprows=[0])
df_news2 = pd.read_table(file2, header=None,skiprows=[0])

# df_news1=df_news1.drop(0)
path = './jpg'
try:
    os.makedirs(path)
except:
    pass
x = df_news1[0].to_numpy()
y = df_news1[7].to_numpy()
z = df_news1[8].to_numpy()
r1 = df_news1[2].to_numpy()
r2 = df_news1[3].to_numpy()
reward = df_news1[1].to_numpy()
tx = df_news1[4].to_numpy()
new = df_news1[5].to_numpy()

x2 = df_news2[0].to_numpy()
y2= df_news2[7].to_numpy()
z2 = df_news2[8].to_numpy()
r1_2 = df_news2[2].to_numpy()
r2_2 = df_news2[3].to_numpy()
reward2 = df_news2[1].to_numpy()
tx2 = df_news2[4].to_numpy()
new2 = df_news2[5].to_numpy()
throughput2 = tx2 / new2
fig, ax = plt.subplots(2, 2, figsize=(14, 7))
ax[0, 0].plot(y, c='b', alpha=0.3)
ax[0, 0].plot(gaussian_filter1d(y, sigma=5), c='b', label='rbg_used')
ax[0, 0].plot(y2, c='r', alpha=0.3)
ax[0, 0].plot(gaussian_filter1d(y2, sigma=5), c='r', label='rbg_used')
ax[0, 0].set_xlabel('epoch')
ax[0, 0].set_ylabel('rbg_used')
# ax[0, 1].plot(z, c='b', alpha=0.3)
# ax[0, 1].plot(gaussian_filter1d(z, sigma=5), c='b', label='bler')
# ax[0, 1].set_xlabel('epoch')
# ax[0, 1].set_ylabel('bler')
# ax[1, 0].plot(throughput, c='b', alpha=0.3)
# ax[1, 0].plot(gaussian_filter1d(throughput, sigma=5), c='b', label='throughput')
# ax[1, 0].set_xlabel('epoch')
# ax[1, 0].set_ylabel('throughput')
ax[0, 1].plot(reward, c='b', alpha=0.3)
ax[0, 1].plot(gaussian_filter1d(reward, sigma=5), c='b', label='reward')
ax[0, 1].plot(reward2, c='r', alpha=0.3)
ax[0, 1].plot(gaussian_filter1d(reward2, sigma=5), c='r', label='reward')
ax[0, 1].set_xlabel('epoch')
ax[0, 1].set_ylabel('reward')
ax[1, 0].plot(r1, c='b', alpha=0.3)
ax[1, 0].plot(gaussian_filter1d(r1, sigma=5), c='b', label='fairness')
ax[1, 0].plot(r1_2, c='r', alpha=0.3)
ax[1, 0].plot(gaussian_filter1d(r1_2, sigma=5), c='r', label='fairness')
ax[1, 0].set_xlabel('epoch')
ax[1, 0].set_ylabel('fairness')
ax[1, 1].plot(tx, c='b', alpha=0.3)
ax[1, 1].plot(gaussian_filter1d(tx, sigma=5), c='b', label='tx')
ax[1, 1].plot(tx2, c='r', alpha=0.3)
ax[1, 1].plot(gaussian_filter1d(tx2, sigma=5), c='r', label='tx')
ax[1, 1].set_xlabel('epoch')
ax[1, 1].set_ylabel('tx')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
plt.savefig(os.path.join(path, '循环非循环对比'))
plt.show()
