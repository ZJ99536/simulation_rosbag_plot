import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,atan2,asin
from tensorflow import keras
import seaborn as sns
from numpy import loadtxt

class DroneControlSim:
    def __init__(self):
        self.sim_time = 2.82
        self.sim_step = 0.027
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 15))
        self.drone_states[0, 0] = 0.0
        self.drone_states[0, 1] = 0.0
        self.drone_states[0, 2] = 1.0
        self.status = 0
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.t_cmd = np.zeros(int(self.sim_time/self.sim_step)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 1.0
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])
        self.cflag = 1
        self.nflag = 1

    def drone_dynamics(self,T,M):
        x = self.drone_states[self.pointer,0]
        y = self.drone_states[self.pointer,1]
        z = self.drone_states[self.pointer,2]
        vx = self.drone_states[self.pointer,3]
        vy = self.drone_states[self.pointer,4]
        vz = self.drone_states[self.pointer,5]
        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]
        p = self.drone_states[self.pointer,9]
        q = self.drone_states[self.pointer,10]
        r = self.drone_states[self.pointer,11]
        


        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])


        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        d_position = np.array([vx,vy,vz])
        d_velocity = np.array([.0,.0,-self.g]) + R_E_B.transpose()@np.array([.0,.0,T])
        self.drone_states[self.pointer,12] = d_velocity[0]
        self.drone_states[self.pointer,13] = d_velocity[1]
        self.drone_states[self.pointer,14] = d_velocity[2]
        d_angle = R_d_angle@np.array([p,q,r])
        # d_angle = np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q,np.array([0, 0, 0])))

        return dx 

    def run(self):
        
        # model = keras.models.load_model('/home/zhoujin/learning/model/quad5_m5.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        # model = keras.models.load_model('/home/zhoujin/learning/model/quad4_75t2.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        # model = keras.models.load_model('/home/zhoujin/learning/model/quadb2_8m.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        model = keras.models.load_model('/home/zhoujin/learning/model/quadf2_8m.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        # model = keras.models.load_model('/home/zhoujin/learning/model/quad2_8m.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        last_velocity = np.zeros(3)
        for self.pointer in range(self.drone_states.shape[0]-1):      
            input = np.zeros(15)
            t = self.pointer
            if t > 2:
                t -= 2
            else:
                t = 0
            current_position = np.array([self.drone_states[t, 0], self.drone_states[t, 1], self.drone_states[t, 2]])
            current_velocity = np.array([self.drone_states[t, 3], self.drone_states[t, 4], self.drone_states[t, 5]])
            if self.status == 0 :
                waypoint0 = np.array([3.6, 1.2, 1.4])
                waypoint1 = np.array([6.9, 0.0, 1.0])
                error = waypoint1 - current_position
                if error[0]**2 + error[1]**2 + error[2]**2 < 0.04:
                    self.status = 0
            if self.status == 1 :
                waypoint0 = np.array([0.0, 1.0, 1.4])
                waypoint1 = np.array([-4.0, 2.0, 1.0])
            
            error0 = waypoint0 - current_position
            error1 = waypoint1 - current_position
            
            for i in range(3):
                input[i] = error0[i]
                input[i+3] = error1[i]
                input[i+6] = self.drone_states[t, i+3]
                input[i+9] = (current_velocity[i] - last_velocity[i]) / self.sim_step
                input[i+12] = self.drone_states[t, i+9]
            input[11] += 9.81
            last_velocity = current_velocity
            output = model(input.reshape(-1,15))

            self.time[self.pointer] = self.pointer * self.sim_step
            psi_cmd = 0.0
            
            x_cmd = ((waypoint0[0] - output[0, 0]) + (waypoint1[0] - output[0, 3])) / 2
            y_cmd = ((waypoint0[1] - output[0, 1]) + (waypoint1[1] - output[0, 4])) / 2
            z_cmd = ((waypoint0[2] - output[0, 2]) + (waypoint1[2] - output[0, 5])) / 2
        
            self.position_cmd[self.pointer] = [x_cmd, y_cmd, z_cmd]
            self.velocity_cmd[self.pointer] = [output[0, 6], output[0, 7], output[0, 8]]
            
            acc_des = np.array([output[0, 9], output[0, 10], output[0, 11]])
            psi = 0
            z_b_des = np.array(acc_des / np.linalg.norm(acc_des))
            y_c = np.array([-sin(psi),cos(psi),0])
            x_b_des = np.cross(y_c,z_b_des) / np.linalg.norm(np.cross(y_c,z_b_des))
            y_b_des = np.cross(z_b_des,x_b_des)
            R_E_B = np.transpose(np.array([x_b_des,y_b_des,z_b_des]))
            psi_cmd = atan2(R_E_B[1,0],R_E_B[0,0])
            theta_cmd = asin(-R_E_B[2,0])
            phi_cmd = atan(R_E_B[2,1]/R_E_B[2,2])
            self.attitude_cmd[self.pointer] = [phi_cmd, theta_cmd, psi_cmd]

            self.rate_cmd[self.pointer] = [output[0, 12], output[0, 13], output[0, 14]]
            thrust_cmd = 0
            for i in range(3):
                thrust_cmd += output[0, i+9] ** 2
            thrust_cmd = thrust_cmd ** 0.5
            self.t_cmd[self.pointer] = thrust_cmd
            # self.rate_cmd[self.pointer] = [1,0,0]
            # self.attitude_cmd[self.pointer] = [output[0, 9], output[0, 10], thrust_cmd]
            # self.attitude_cmd[self.pointer] = [input[9], input[10], input[11]]

            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m
            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
        self.time[-1] = self.sim_time



    def rate_controller(self,cmd):
        kp_p = 0.045
        kp_q = 0.045
        kp_r = 0.05
        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])

   

    def plot_states(self):
        cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/single8mfaster1.txt', delimiter=',')
        # snap = loadtxt('/home/zhoujin/trajectory-generation/trajectory/M_snap.txt', delimiter=',')
        sns.set(style="darkgrid", font_scale=1.0)
        for i in cpc[:,0:1]:
            i -= 0.07
        fig1, ax1 = plt.subplots(2,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='WN&CNet',linewidth = 4.56, color = 'cornflowerblue')
        ax1[0,0].plot(cpc[:,0:1], cpc[:,1:2],label='CPC',linewidth = 4.56, color = 'lightcoral', alpha = 0.7, ls = '--')
        # ax1[0,0].plot(snap[0,:], snap[1,:],label='snap',linewidth = 4.56)
        ax1[0,0].set_ylabel('x [$m$]', fontsize=20)
        ax1[0,0].set_xlabel('time [$s$]', fontsize=20)
        ax1[0,1].plot(cpc[:,0:1], cpc[:,2:3],linewidth = 4.56, color = 'lightcoral', alpha = 0.7, ls = '--')
        ax1[0,1].plot(self.time,self.drone_states[:,1],linewidth = 4.56, color = 'cornflowerblue')
        ax1[0,1].set_ylabel('y [$m$]', fontsize=20)
        ax1[0,1].set_xlabel('time [$s$]', fontsize=20)
        ax1[0,2].plot(cpc[:,0:1], cpc[:,3:4],linewidth = 4.56, color = 'lightcoral', alpha = 0.7, ls = '--')
        ax1[0,2].plot(self.time,self.drone_states[:,2],linewidth = 4.56, color = 'cornflowerblue')
        ax1[0,2].set_xlabel('time [$s$]', fontsize=20)
        ax1[0,2].set_ylabel('z [$m$]', fontsize=20)
        ax1[0,0].legend(prop = {'size':16})

        self.rate_cmd[-1] = self.rate_cmd[-2]
        # ax1[1,0].plot(self.time,self.drone_states[:,9],label='WN&CNet',linewidth = 4.56, color = 'cornflowerblue')
        ax1[1,0].plot(self.time,self.rate_cmd[:,0],label='WN&CNet',linewidth = 4.56, color = 'cornflowerblue')
        ax1[1,0].plot(cpc[:,0:1], cpc[:,11:12],label='CPC',linewidth = 4.56, color = 'lightcoral', alpha = 0.7, ls = '--')
        ax1[1,0].set_ylabel('$\omega x_c$ [$rad/s$]', fontsize=20)
        ax1[1,0].set_xlabel('time [$s$]', fontsize=20)
        ax1[1,1].plot(cpc[:,0:1], cpc[:,12:13],linewidth = 4.56, color = 'lightcoral', alpha = 0.7, ls = '--')
        # ax1[1,1].plot(self.time,self.drone_states[:,10],linewidth = 4.56, color = 'cornflowerblue')
        ax1[1,1].plot(self.time,self.rate_cmd[:,1],linewidth = 4.56, color = 'cornflowerblue')
        ax1[1,1].set_ylabel('$\omega y_c$ [$rad/s$]', fontsize=20)
        ax1[1,1].set_xlabel('time [$s$]', fontsize=20)
        ax1[1,2].plot(cpc[:,0:1], cpc[:,13:14],linewidth = 4.56, color = 'lightcoral', alpha = 0.7, ls = '--')
        # ax1[1,2].plot(self.time,self.drone_states[:,11],linewidth = 4.56, color = 'cornflowerblue')
        ax1[1,2].plot(self.time,self.rate_cmd[:,2],linewidth = 4.56, color = 'cornflowerblue')
        ax1[1,2].set_ylabel('$\omega z_c$ [$rad/s$]', fontsize=20)
        ax1[1,2].set_xlabel('time [$s$]', fontsize=20)
        ax1[1,0].legend(prop = {'size':16},loc=3)

        # self.t_cmd[-1] = self.t_cmd[-2]
        # plt.plot(self.time,self.t_cmd,label='nn',linewidth = 4.56, color = 'cornflowerblue')
        # plt.plot(cpc[:,0:1], cpc[:,-1],label='CPC',linewidth = 4.56, color = 'lightcoral', alpha = 0.6, ls = '--')
        # plt.xlabel('time [$s$]', fontsize=20)
        # plt.ylabel('thrust cmd [$m/s^2$]', fontsize=20)        
        # plt.legend(prop = {'size':18})

        # self.velocity_cmd[-1] = self.velocity_cmd[-2]
        # ax1[1,0].plot(self.time,self.drone_states[:,3])
        # ax1[1,0].plot(cpc[:,0:1], cpc[:,8:9])
        # ax1[1,0].set_ylabel('vx[m/s]')
        # ax1[1,1].plot(self.time,self.drone_states[:,4])
        # ax1[1,1].plot(cpc[:,0:1], cpc[:,9:10])
        # ax1[1,1].set_ylabel('vy[m/s]')
        # ax1[1,2].plot(self.time,self.drone_states[:,5])
        # ax1[1,2].plot(cpc[:,0:1], cpc[:,10:11])
        # ax1[1,2].set_ylabel('vz[m/s]')

        # self.attitude_cmd[-1] = self.attitude_cmd[-2]
        # ax1[2,0].plot(self.time,self.drone_states[:,12])
        # ax1[2,0].plot(cpc[:,0:1], cpc[:,5:6])
        # ax1[2,0].set_ylabel('phi[rad]')
        # ax1[2,1].plot(self.time,self.drone_states[:,13])
        # ax1[2,1].plot(cpc[:,0:1], cpc[:,6:7])
        # ax1[2,1].set_ylabel('theta[rad]')
        # ax1[2,2].plot(self.time,self.drone_states[:,14])
        # ax1[2,2].plot(cpc[:,0:1], cpc[:,7:8])
        # ax1[2,2].set_ylabel('psi[rad]')

        # self.rate_cmd[-1] = self.rate_cmd[-2]
        # ax1[3,0].plot(self.time,self.drone_states[:,9])
        # ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        # ax1[3,0].set_ylabel('p[rad/s]')
        # ax1[3,1].plot(self.time,self.drone_states[:,10])
        # ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        # ax1[3,0].set_ylabel('q[rad/s]')
        # ax1[3,2].plot(self.time,self.drone_states[:,11])
        # ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        # ax1[3,0].set_ylabel('r[rad/s]')

    def plot_3d(self):
        cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/single8mfaster1.txt', delimiter=',')
        cpcx = np.zeros(len(cpc))
        cpcy = np.zeros(len(cpc))
        cpcz = np.zeros(len(cpc))
        nnx = np.zeros(len(self.drone_states[:,0:1]))
        nny = np.zeros(len(self.drone_states[:,0:1]))
        nnz = np.zeros(len(self.drone_states[:,0:1]))
        for i in range(len(cpc)):
            cpcx[i] = cpc[i,1:2]
            cpcy[i] = cpc[i,2:3]
            cpcz[i] = cpc[i,3:4]
            if self.cflag:
                if (cpcx[i]-6.9)**2+cpcy[i]**2+(cpcz[i]-1)**2<0.09:
                    print('cpc arrrrrr')
                    print(i)
                    self.cflag = 0
        for i in range(len(nnx)):
            nnx[i] = self.drone_states[i,0:1]
            nny[i] = self.drone_states[i,1:2]
            nnz[i] = self.drone_states[i,2:3]
            if self.nflag:
                if (nnx[i]-6.9)**2+nny[i]**2+(nnz[i]-1)**2<0.09:
                    print('nn arrrrrr')
                    print(i)
                    self.nflag = 0
        # snap = loadtxt('/home/zhoujin/trajectory-generation/trajectory/M_snap.txt', delimiter=',')
        sns.set(style="darkgrid", font_scale=1.0)
        plt.style.use('classic')
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(nnx, nny, nnz,label='WN&CNet',linewidth = 2.56, color = 'cornflowerblue')
        ax1.plot3D(cpcx, cpcy, cpcz,label='CPC',linewidth = 2.56, color = 'lightcoral', alpha = 0.7, ls = '--')
        ax1.scatter3D(0,0,1,s=520,color='lightcoral',marker = '*',label='Start point')
        ax1.scatter3D(6.9,0,1,s=520,color='cornflowerblue',marker = '*',label='End point')
        # ax1.scatter3D(3.6,1.2,1.4,s=320,color='cornflowerblue',marker = '*',label='Waypoint',type='s')
        # ax1.scatter3D(3.6,1.2,1.4,s=1620,linewidth=12,color='cornflowerblue')
        # ax1.scatter3D(6.9,0,1.0,s=320, color='cornflowerblue',marker = '*',label='Endpoint')
        x_list = [3.6]
        y_list = [1.2]
        z_list = [1.4]
        r = 0.25
        # line_list = ["--", "--", "-", "--", "--"]
        for i in range(1):
            ax1.plot3D([x_list[i], x_list[i], x_list[i], x_list[i], x_list[i]], [y_list[i] - r, y_list[i] - r, y_list[i] + r, y_list[i] + r, y_list[i] - r], [z_list[i] - r, 1.45, 1.45, z_list[i] - r, z_list[i] - r], 'red', linewidth=3)
        plt.xlabel('x [$m$]', fontsize=20)
        plt.ylabel('y [$m$]', fontsize=20)
        ax1.set_zlabel('z [$m$]', fontsize=20)
        plt.xlim((0,7))
        plt.ylim((-0.5,1.5))
        ax1.set_zlim(0.7,1.45)
        plt.legend(prop = {'size':18})

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    # drone.plot_states()
    drone.plot_3d()
    plt.show()