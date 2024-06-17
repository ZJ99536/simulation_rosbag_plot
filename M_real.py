import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,atan2,asin
from tensorflow import keras
import seaborn as sns
from numpy import loadtxt

import rosbag

bag_file = '/home/zhoujin/Desktop/bags/2024-06-04-14-43-27.bag'
bag = rosbag.Bag(bag_file, "r")   # 载入bag文件
position_x= []
time_data = []
init_count = 6420
end_count = 8000
count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_position']):
        # print(topic)
        if count > init_count and count < end_count:
            position_x.append(msg.pose.position.x)
            time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1
bag.close()


cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/M_cpc_D.txt', delimiter=',')
# cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/M_cpc.csv')
# sns.set_theme()
sns.set(style="darkgrid", font_scale=1.0)
# plt.subplot(4,5,10)
plt.plot(cpc[:,0:1], cpc[:,1:2])
plt.plot(time_data,position_x)

# sns.lineplot(data=cpc[:,1:2])
# sns.lineplot(data=cpc, x = 't', y = 'x')
plt.show()