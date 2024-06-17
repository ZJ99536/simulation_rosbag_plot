import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)
np.set_printoptions(linewidth=100)
 
path = [[-3, 2], [0,1],[3, 0], [0,-1],[-3, 0], [0,1],[3, 0], [0,-1],[-3, -2]]
#path = [[1, 3], [1, 5], [1, 2]]
path = np.array(path)
 
x = path[:, 0]
deltaT = 3 ** 0.5 * 0.5
T = np.linspace(0, deltaT * (len(x) - 1), len(x))
 
K = 4                   # jerk为3阶导数，取K=3
n_order = 2 * K - 1     # 多项式阶数
M = len(x) - 1          # 轨迹的段数
N = M * (n_order + 1)   # 矩阵Q的维数
 
def getQk(T_down, T_up):
    Q = np.zeros((8, 8))
    Q[4][5] = 1440 * (T_up**2 - T_down**2)
    Q[4][6] = 2880 * (T_up**3 - T_down**3)
    Q[4][7] = 5040 * (T_up**4 - T_down**4)
    Q[5][6] = 10800 * (T_up**4 - T_down**4)
    Q[5][7] = 20160 * (T_up**5 - T_down**5)
    Q[6][7] = 50400 * (T_up**6 - T_down**6)
    Q = Q + Q.T # Q为对称矩阵
    Q[4][4] = 576 * (T_up**1 - T_down**1)
    Q[5][5] = 4800 * (T_up**3 - T_down**3)
    Q[6][6] = 25920 * (T_up**5 - T_down**5)
    Q[7][7] = 100800 * (T_up**7 - T_down**7)
    return Q
 
Q = np.zeros((N, N))
for k in range(1, M + 1):
    Qk = getQk(T[k - 1], T[k])
    Q[(8 * (k - 1)) : (8 * k), (8 * (k - 1)) : (8 * k)] = Qk
Q = Q * 2
 
#A0 = np.zeros((2 * K + M - 3, N))
A0 = np.zeros((2 * K + M - 1, N))
b0 = np.zeros(len(A0))
 
#for k in range(K-1):
for k in range(K):
    for i in range(k, 8):
        c = 1
        for j in range(k):
            c *= (i - j)
        A0[0 + k * 2][i]                = c * T[0]**(i - k)
        A0[1 + k * 2][(M - 1) * 8 + i]  = c * T[M]**(i - k)
b0[0] = x[0]
b0[1] = x[M]
 
 
for m in range(1, M):
    for i in range(8):
        #A0[6 + m - 1][m * 8 + i] = T[m]**i
        A0[8 + m - 1][m * 8 + i] = T[m]**i
    #b0[6 + m - 1] = x[m]
    b0[8 + m - 1] = x[m]
 
#print(A0.shape)
#print(b0)
#print(b0.shape)
 
A1 = np.zeros(((M - 1) * K, N))
b1 = np.zeros(len(A1))
for m in range(M - 1):
    for k in range(K): # 最多两阶导数相等    
        for i in range(k, 8):
            c = 1
            for j in range(k):
                c *= (i - j)
            index = m * 4  + k
            A1[index][m * 8 + i] = c * T[m + 1]**(i - k)
            A1[index][(m + 1)* 8 + i] = -c * T[m + 1]**(i - k)
 
#print(A1)
#print(A1.shape)
#print(b1)
#print(b1.shape)
 
 
A = np.vstack((A0, A1))
b = np.hstack((b0, b1))
 
#print(A)
 
print(Q.shape)
print(A.shape)
print(b.shape)
 
from cvxopt import matrix, solvers
 
Q = matrix(Q)
q = matrix(np.zeros(N))
 
A = matrix(A)
b = matrix(b)
print(np.linalg.matrix_rank(A))
print(np.linalg.matrix_rank(np.vstack((Q,A))))
 
#print(np.linalg.matrix_rank(A;b))
#print(np.linalg.matrix_rank(b))
result = solvers.qp(Q, q, A=A, b=b)
p_coff = np.asarray(result['x']).flatten()
 
Pos = []
Vel = []
Acc = []
Jer = []
for k in range(M):
    t = np.linspace(T[k], T[k + 1], 100)
    t_pos = np.vstack((t**0, t**1, t**2, t**3, t**4, t**5, t**6, t**7))
    t_vel = np.vstack((t*0, t**0, 2 * t**1, 3 * t**2, 4 * t**3, 5 * t**4, 6*t**5, 7*t**6))
    t_acc = np.vstack((t*0, t*0, 2 * t**0, 3 * 2 * t**1, 4 * 3 * t**2, 5 * 4 * t**3, 6*5*t**4, 7*6*t**5))
    t_jer = np.vstack((t*0, t*0, t*0, 3 * 2 * t**0, 4 * 3 *2* t**1, 5 * 4 *3* t**2, 6*5*4*t**3, 7*6*5*t**4))
    coef = p_coff[k * 8 : (k + 1) * 8]
    coef = np.reshape(coef, (1, 8))
    pos = coef.dot(t_pos)
    vel = coef.dot(t_vel)
    acc = coef.dot(t_acc)
    jer = coef.dot(t_jer)
    Pos.append([t, pos[0]])
    Vel.append([t, vel[0]])
    Acc.append([t, acc[0]])
    Jer.append([t, jer[0]])
 
Pos = np.array(Pos)
Vel = np.array(Vel)
Acc = np.array(Acc)
Jer = np.array(Jer)
plt.subplot(4, 1, 1)
plt.plot(Pos[:, 0, :].T, Pos[:, 1, :].T)
# plt.title("position")
plt.xlabel("time(s)")
plt.ylabel("position(m)")
plt.subplot(4, 1, 2)
plt.plot(Vel[:, 0, :].T, Vel[:, 1, :].T)
# plt.title("velocity")
plt.xlabel("time(s)")
plt.ylabel("velocity(m/s)")
plt.subplot(4, 1, 3)
plt.plot(Acc[:, 0, :].T, Acc[:, 1, :].T)
# plt.title("accel")
plt.xlabel("time(s)")
plt.ylabel("accel(m/s^2)")
plt.show()
plt.subplot(4, 1, 4)
plt.plot(Jer[:, 0, :].T, Jer[:, 1, :].T)
# plt.title("jerk")
plt.xlabel("time(s)")
plt.ylabel("jer(m/s^3)")
plt.show()