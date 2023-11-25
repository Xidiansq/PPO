# -*-coding:utf-8-*-
from user import *
from beam_init import *
import matplotlib.pyplot as plt
beam, lat_log = setInitBeamCenterPos(0, [0, 0, 0], type='IRIDIUM')
user_list=initial_all_user(500000,20,lat_log)
for i in range(len(user_list)):
    user_list[i].model2_update(tb=0,capacity=0)
position1,position2=get_user_position(user_list)
print(len(position2))
_,request=get_user_traffic_info(user_list)
position2=np.array(position2)
a=position_plot(lat_log,position2)
aaa,beam_connect=userconnectsate(position1,beam,request,20)
# fig, ax = plt.subplots(1, 1)
# x = position2[:, 0]
# y = position2[:, 1]
# ax.scatter(x, y,s=5,c='r')
# plt.show()
# plt.close()
print(lat_log)