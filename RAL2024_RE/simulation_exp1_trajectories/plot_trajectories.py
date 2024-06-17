import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,atan2,asin
from tensorflow import keras
import seaborn as sns
from numpy import loadtxt

fig = plt.figure()
plt.style.use('classic')
ax1 = plt.axes(projection='3d')
ax1.view_init(elev=13, azim=60)
color = [(75, 102, 173)
    , (98, 190, 166)
    , (205, 234, 157)
    , (253, 186, 107)
    , (235, 96, 70)]

t_error = 0

for i in range(13):
    cpci = loadtxt('trajectories_simulation_cpc.txt', delimiter=',',skiprows = i,max_rows = 1)
    torchi = loadtxt('trajectories_simulation_torch.txt', delimiter=',',skiprows = i,max_rows = 1)
    cpcx = []
    cpcy = []
    cpcz = []
    torchx = []
    torchy = []
    torchz = []
    # print(len(cpci[0,:])/3 - 1)
    for j in range(int((len(cpci)-8)/3)):
        cpcx.append(cpci[j*3+6])
        cpcy.append(cpci[j*3+7])
        cpcz.append(cpci[j*3+8])
    for j in range(int((len(torchi)-8)/3)):
        torchx.append(torchi[j*3+6])
        torchy.append(torchi[j*3+7])
        torchz.append(torchi[j*3+8])

    t_error += torchi[-2] - cpci[-2]
    ax1.plot3D(cpcx, cpcy, cpcz, color='cornflowerblue', linewidth=1.2,ls='-',label='CPC', alpha=0.8)
    ax1.plot3D(torchx, torchy, torchz, color='lightcoral', linewidth=1.2,ls='-',label='NN', alpha=0.8)
    ax1.scatter3D(cpci[3], cpci[4], cpci[5], s=200, marker="*", color='cornflowerblue')

ax1.scatter3D(0, 0, 1, s=300, marker="*", color=np.array(color[3]) / 255)
print(t_error)

plt.show()

