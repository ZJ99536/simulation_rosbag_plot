# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 下午1:39
# @Author  : 赵方国
# @File    : plot_traj.py
import numpy as np
from matplotlib import pyplot as plt

pos_sp_1 = "/home/zfg/Graduate_Test/Helper_draw_plot/swarm_plot/imav/MPC_good/offb1_node-position_setpoint.csv"
pos_sp_2 = "/home/zfg/Graduate_Test/Helper_draw_plot/swarm_plot/imav/MPC_good/offb2_node-position_setpoint.csv"
pos_tr_1 = "/home/zfg/Graduate_Test/Helper_draw_plot/swarm_plot/imav/MPC_good/vrpn_client_node-jiahao1-pose.csv"
pos_tr_2 = "/home/zfg/Graduate_Test/Helper_draw_plot/swarm_plot/imav/MPC_good/vrpn_client_node-jiahao2-pose.csv"
pos_setpoint = []
pos_fly = []


def load_setpoint(name):
    with open(name) as f:
        data = f.readlines()
        x = [float(item.split(",")[5]) for item in data]
        y = [float(item.split(",")[6]) for item in data]
        z = [float(item.split(",")[7]) for item in data]
        pos_setpoint.append([x, y, z])
        print("")


def load_true(name):
    with open(name) as f:
        data = f.readlines()
        x = [float(item.split(",")[5]) for item in data]
        y = [float(item.split(",")[6]) for item in data]
        z = [float(item.split(",")[7]) for item in data]
        pos_fly.append([x, y, z])


if __name__ == '__main__':
    font = {'family': 'serif',
            'serif': 'Arial',
            'weight': 'normal',
            'size': 35}
    plt.rc('font', **font)
    load_setpoint(pos_sp_1)
    load_setpoint(pos_sp_2)
    load_true(pos_tr_1)
    load_true(pos_tr_2)
    fig = plt.figure()
    plt.style.use('classic')
    ax1 = plt.axes(projection='3d')
    ax1.view_init(elev=13, azim=60)
    color = [(75, 102, 173)
        , (98, 190, 166)
        , (205, 234, 157)
        , (253, 186, 107)
        , (235, 96, 70)]
    ax1.plot3D(pos_setpoint[0][0], pos_setpoint[0][1], pos_setpoint[0][2], color=np.array(color[0]) / 255, linewidth=3,
               label='Reference 1',
               linestyle='--', alpha=0.8)
    ax1.plot3D(pos_setpoint[1][0], pos_setpoint[1][1], pos_setpoint[1][2], color=np.array(color[1]) / 255, linewidth=3,
               label='Reference 2', linestyle='--', alpha=0.8)
    ax1.plot3D(pos_fly[0][0], pos_fly[0][1], pos_fly[0][2], color=np.array(color[0]) / 255, linewidth=4,
               label='Quadrotor 1', alpha=0.8)
    ax1.plot3D(pos_fly[1][0], pos_fly[1][1], pos_fly[1][2], color=np.array(color[1]) / 255, linewidth=4,
               label='Quadrotor2',
               alpha=0.8)
    alpha_list = [0.4, 0.6, 0.8, 0.6, 0.4]
    line_list = ["--", "--", "-", "--", "--"]
    for i in range(5):
        ax1.plot3D([2 - i * 0.3, 2.3 - i * 0.3, 2.3 - i * 0.3, 2 - i * 0.3,
                    2 - i * 0.3], [0, 0, 0, 0, 0], [0.5, 0.5, 1.5, 1.5, 0.5], 'red', linewidth=4,
                   alpha=alpha_list[i], linestyle=line_list[i])
    # for i in range(5):
    #     ax1.plot3D([-2 + i * 0.1 * 2.3, -2.3 + i * 0.1 * 2.3, -2.3 + i * 0.1 * 2.3, -2 + i * 0.1 * 2.3,
    #                 -2 + i * 0.1 * 2.3], [0, 0, 0, 0, 0], [0.75, 0.75, 1.25, 1.25, 0.75], 'orange', linewidth=2,
    #                alpha=1 - (i * 0.1))
    # ax1.plot3D([0.5, -0.3, -0.3, -0.3, -0.3], [0, 0, 0, 0, 0], [1, 1, 2, 2, 1], 'orange', linewidth=4)
    # ax.plot3D([0.8, 1.5, 1.5, 0.8, 0.8], [0, 0, 0, 0, 0], [1, 1, 2, 2, 1], 'orange', linewidth=4)
    # ax.plot3D([0.8, 1.5, 1.5, 0.8, 0.8], [0, 0, 0, 0, 0], [1, 1, 2, 2, 1], 'orange', linewidth=4)
    # ax1.plot3D([0.6, 0.6, 0.6, 0.6, 0.6], [-1, -0.5, -0.5, -1, -1], [1, 1, 2, 2, 1], 'orange', linewidth=4)
    # ax1.plot3D([0.6, 0.6, 0.6, 0.6, 0.6], [-1, -0.5, -0.5, -1, -1], [1, 1, 2, 2, 1], 'orange', linewidth=4)
    # ax1.plot3D([0.8, 0.8, 0.8, 0.8, 0.8], [0.5, 0, 0, 0.5, 0.5], [1, 1, 2, 2, 1], 'orange', linewidth=4)
    # ax1.plot3D([0.8, 0.8, 0.8, 0.8, 0.8], [0.5, 0, 0, 0.5, 0.5], [1, 1, 2, 2, 1], 'orange', linewidth=4)
    x2 = np.linspace(-2.5, 2.5, 9)
    y2 = np.linspace(-1.5, 1.5, 9)
    z2 = np.linspace(0, 2, 9)
    X2, Y2 = np.meshgrid(x2, y2)
    T2, Z2 = np.meshgrid(y2, z2)

    ax1.plot_surface(X=X2, Y=Y2, Z=X2 * 0 + 0, color='white', alpha=0.1, edgecolors='white')
    ax1.plot_surface(X=X2, Y=Y2, Z=X2 * 0 + 2, color='white', alpha=0.1, edgecolors='white')
    ax1.plot_surface(X=X2, Y=X2 * 0 - 1.5, Z=Z2, color='white', alpha=0.1, edgecolors='white')
    ax1.plot_surface(X=X2, Y=X2 * 0 + 1.5, Z=Z2, color='white', alpha=0.1, edgecolors='white')
    ax1.plot_surface(X=X2 * 0 - 2.5, Y=T2, Z=Z2, color='white', alpha=0.1, edgecolors='white')
    ax1.plot_surface(X=X2 * 0 + 2.5, Y=T2, Z=Z2, color='white', alpha=0.1, edgecolors='white')

    ax1.scatter3D(pos_setpoint[0][0][0], pos_setpoint[0][1][0], pos_setpoint[0][2][0], s=320, marker="*",
                  color=np.array(
                      color[0]) / 255)
    ax1.scatter3D(pos_setpoint[1][0][0], pos_setpoint[1][1][0], pos_setpoint[1][2][0], s=320, marker="*",
                  color=np.array(color[1]) / 255)

    # ax1.scatter3D(setx[0], sety[0], setz[0], linewidth=2)
    ax1.text3D(1.5, 1, 5, "Dynamic Waypoint", fontsize=20,color="red", alpha=0.8)
    # ax1.text3D(pos_setpoint[1][0][0]+0.2, pos_setpoint[1][1][0]+0.2, pos_setpoint[1][2][0]+0.2, "start", fontsize=30)
    # ax1.scatter3D(posx[len(posx) - 1], posy[len(posy) - 1], posz[len(posz) - 1], linewidth=2)
    # ax1.scatter3D(setx[len(setx) - 1], sety[len(sety) - 1], setz[len(setz) - 1], linewidth=2)
    # ax1.text3D(posx[len(posx) - 1], posy[len(posy) - 1], posz[len(posz) - 1], "end", fontsize=30)
    # ax1.text3D(setx[len(setx) - 1], sety[len(sety) - 1], setz[len(setz) - 1], "end", fontsize=30)

    # ax1.w_xaxis.set_pane_color((1.0, 1.0, 1, 1))
    # ax1.w_zaxis.set_pane_color((1.0, 1.0, 1, 1))
    # ax1.w_zaxis.set_pane_color((1.0, 1.0, 1, 1))

    # ax.annotate('text', xy=(posx[0], posy[0]), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # ax1.set_xlim(-2.3, 2.3)
    ax1.set_xticks([-3, -2, -1, 0, 1, 2, 3], fontsize=100)
    ax1.set_yticks([-2, -1, 0, 1, 2], fontsize=100)
    ax1.set_zticks([0, 1, 2, 3, 4], fontsize=100)
    ax1.set_xlabel("X [m]", fontsize=20)

    # ax1.set_ylim(-1.3, 1.3)
    ax1.set_ylabel("Y [m]", fontsize=20)
    # ax1.set_zlim(0, 2.5)
    ax1.set_zlabel("Z [m]", fontsize=20)

    # ax.set_title('Two UAVs Trajectories')
    ax1.legend(fontsize=20, loc=1)
    plt.gca().set_box_aspect((3, 2, 0.5))
    plt.show()
    print(" ")
