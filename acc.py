import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,atan2,asin
from tensorflow import keras
import seaborn as sns
from numpy import loadtxt

import rosbag

bag_file = '/home/zhoujin/Desktop/bags/2023-09-12-20-46-03.bag'
bag = rosbag.Bag(bag_file, "r")   # 载入bag文件
position_x= []
position_y= []
position_z= []
velocity_x= []
velocity_y= []
velocity_z= []
accel_x= []
accel_y= []
accel_z= []

time_data = []
init_count = 2425
end_count = 3240
count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_position']):
        # print(topic)
        if count > init_count and count < end_count:
            position_x.append(msg.pose.position.x)
            position_y.append(msg.pose.position.y)
            position_z.append(msg.pose.position.z)
            time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1
count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_velocity']):
        # print(topic)
        if count > init_count and count < end_count:
            velocity_x.append(msg.twist.linear.x)
            velocity_y.append(msg.twist.linear.y)
            velocity_z.append(msg.twist.linear.z)
            # time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1

count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_acc']):
        # print(topic)
        if count > init_count and count < end_count:
            accel_x.append(msg.twist.linear.x)
            accel_y.append(msg.twist.linear.y)
            accel_z.append(msg.twist.linear.z)
            # time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1
bag.close()

cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/M_cpc.txt', delimiter=',')
# cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/M_cpc.csv')
# sns.set_theme()
sns.set(style="darkgrid", font_scale=1.0)
plt.subplot(3,3,1)
plt.plot(cpc[:,0:1], cpc[:,1:2])
plt.plot(time_data,position_x)
plt.subplot(3,3,2)
plt.plot(cpc[:,0:1], cpc[:,2:3])
plt.plot(time_data,position_y)
plt.subplot(3,3,3)
plt.plot(cpc[:,0:1], cpc[:,3:4])
plt.plot(time_data,position_z)
plt.subplot(3,3,4)
plt.plot(cpc[:,0:1], cpc[:,8:9])
plt.plot(time_data,velocity_x)
plt.subplot(3,3,5)
plt.plot(cpc[:,0:1], cpc[:,9:10])
plt.plot(time_data,velocity_y)
plt.subplot(3,3,6)
plt.plot(cpc[:,0:1], cpc[:,10:11])
plt.plot(time_data,velocity_z)
plt.subplot(3,3,7)
plt.plot(cpc[:,0:1], cpc[:,5:6])
plt.plot(time_data,accel_x)
plt.subplot(3,3,8)
plt.plot(cpc[:,0:1], cpc[:,6:7])
plt.plot(time_data,accel_y)
plt.subplot(3,3,9)
plt.plot(cpc[:,0:1], cpc[:,7:8])
plt.plot(time_data,accel_z)


# sns.lineplot(data=cpc[:,1:2])
# sns.lineplot(data=cpc, x = 't', y = 'x')
plt.show()