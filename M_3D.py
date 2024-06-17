import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,atan2,asin
from tensorflow import keras
import seaborn as sns
from numpy import loadtxt

import rosbag

bag_file = '/home/zhoujin/Desktop/bags/2023-09-12-18-08-50.bag'

# bag_file = '/home/zhoujin/Desktop/bags/good1.bag'
bag = rosbag.Bag(bag_file, "r")   # 载入bag文件
position_x= []
position_y= []
position_z= []
time_data = []
init_count = 3725
end_count = 4520

# mind = np.zeros(2,7)

# init_count = 4125
# end_count = 4940

count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_position']):
        # print(topic)
        if count > init_count and count < end_count:
            position_x.append(msg.pose.position.x)
            position_y.append(msg.pose.position.y)
            position_z.append(msg.pose.position.z)
            # time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1
bag.close()

bag_file = '/home/zhoujin/Desktop/bags/good1.bag'
bag = rosbag.Bag(bag_file, "r")   # 载入bag文件
position2_x= []
position2_y= []
position2_z= []
init_count = 4125
end_count = 4940
count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_position']):
        # print(topic)
        if count > init_count and count < end_count:
            position2_x.append(msg.pose.position.x)
            position2_y.append(msg.pose.position.y)
            position2_z.append(msg.pose.position.z)
            # time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1
bag.close()

bag_file = '/home/zhoujin/Desktop/bags/2023-09-12-20-46-03.bag'
bag = rosbag.Bag(bag_file, "r")   # 载入bag文件
position1_x= []
position1_y= []
position1_z= []
init_count = 2425
end_count = 3240
count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_position']):
        # print(topic)
        if count > init_count and count < end_count:
            position1_x.append(msg.pose.position.x)
            position1_y.append(msg.pose.position.y)
            position1_z.append(msg.pose.position.z)
            # time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1
bag.close()

bag_file = '/home/zhoujin/Desktop/bags/cpc4.bag'
bag = rosbag.Bag(bag_file, "r")   # 载入bag文件
cpc_x= []
cpc_y= []
cpc_z= []
# time_data = []
init_count = 2925
end_count = 3740
count = 0
for topic, msg, t in bag.read_messages(topics=['/outer_position']):
        # print(topic)
        if count > init_count and count < end_count:
            cpc_x.append(msg.pose.position.x)
            cpc_y.append(msg.pose.position.y)
            cpc_z.append(msg.pose.position.z)
            # time_data.append(t.to_sec()-init_time)
        elif count < init_count:
            init_time =  t.to_sec()
        count += 1
bag.close()

# sns.set(style="darkgrid", font_scale=1.0)
# plt.plot(time_data,cpc_x)

# cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/M_cpc4.txt', delimiter=',')

fig = plt.figure()
plt.style.use('classic')
ax1 = plt.axes(projection='3d')
ax1.view_init(elev=13, azim=60)
color = [(75, 102, 173)
    , (98, 190, 166)
    , (205, 234, 157)
    , (253, 186, 107)
    , (235, 96, 70)]
ax1.plot3D(cpc_x, cpc_y, cpc_z, color=np.array(color[4]) / 255, linewidth=1.5,ls='-',
            label='CPC', alpha=0.8)
snap = loadtxt('/home/zhoujin/trajectory-generation/trajectory/mm_snap.txt', delimiter=',')
ax1.plot3D(snap[1,:], snap[2,:], snap[3,:], color=np.array(color[0]) / 255, linewidth=1.5,ls='-',
            label='Minimum Snap', alpha=0.8)
ax1.plot3D(position_x, position_y, position_z, color=np.array(color[3]) / 255, linewidth=1.2,
            label='WN&CNets: test 1', alpha=0.8)
ax1.plot3D(position1_x, position1_y, position1_z, color=np.array(color[2]) / 255, linewidth=1.2,
            label='WN&CNets: test 2', alpha=0.8)
ax1.plot3D(position2_x, position2_y, position2_z, color=np.array(color[1]) / 255, linewidth=1.2,
            label='WN&CNets: test 3', alpha=0.8)



x2 = np.linspace(-3.5, 3.5, 9)
y2 = np.linspace(-2.5, 2.5, 9)
z2 = np.linspace(0, 2.5, 9)
X2, Y2 = np.meshgrid(x2, y2)
T2, Z2 = np.meshgrid(y2, z2)

ax1.plot_surface(X=X2, Y=Y2, Z=X2 * 0 + 0, color='white', alpha=0.1, edgecolors='white')
ax1.plot_surface(X=X2, Y=Y2, Z=X2 * 0 + 2.5, color='white', alpha=0.1, edgecolors='white')
ax1.plot_surface(X=X2, Y=X2 * 0 - 2.5, Z=Z2, color='white', alpha=0.1, edgecolors='white')
ax1.plot_surface(X=X2, Y=X2 * 0 + 2.5, Z=Z2, color='white', alpha=0.1, edgecolors='white')
ax1.plot_surface(X=X2 * 0 - 3.5, Y=T2, Z=Z2, color='white', alpha=0.1, edgecolors='white')
ax1.plot_surface(X=X2 * 0 + 3.5, Y=T2, Z=Z2, color='white', alpha=0.1, edgecolors='white')

ax1.scatter3D(-3.0, 2.0, 1.0, s=200, marker="*", color=np.array(color[4]) / 255)

x_list = [0.0, 0.4, 0.3, 0.0]
y_list = [1.0, -1.0, 1.0, -1.0]
z_list = [1.4, 0.6, 1.5, 0.5]
r = 0.5
# line_list = ["--", "--", "-", "--", "--"]
for i in range(4):
    ax1.plot3D([x_list[i], x_list[i], x_list[i], x_list[i], x_list[i]], [y_list[i] - r, y_list[i] - r, y_list[i] + r, y_list[i] + r, y_list[i] - r], [z_list[i] - r, z_list[i] + r, z_list[i] + r, z_list[i] - r, z_list[i] - r], 'red', linewidth=4)
ax1.plot3D([2.90 - r, 2.90 - r, 2.90 + r, 2.90 + r, 2.90 - r], [0, 0, 0, 0, 0], [1.0 - r, 1.0  + r, 1.0 + r, 1.0  - r, 1.0  - r], 'red', linewidth=4)
ax1.plot3D([-2.90 - r, -2.90 - r, -2.90 + r, -2.90 + r, -2.90 - r], [0, 0, 0, 0, 0], [1.0 - r, 1.0  + r, 1.0 + r, 1.0  - r, 1.0  - r], 'red', linewidth=4)


ax1.set_xlabel("X [m]", fontsize=18)
ax1.set_ylabel("Y [m]", fontsize=18)
ax1.set_zlabel("Z [m]", fontsize=18)
ax1.legend(fontsize=18, loc=1)

plt.show()