from casadi import MX, DM, vertcat, horzcat, veccat, norm_2, dot, mtimes, nlpsol, diag, repmat, sum1, sin, cos, tan
import casadi as ca
# from cv2 import sqrt
import numpy as np
# from pygments import lex
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,atan2,asin

import torch
import torch.nn as nn

class DroneControlSim:
    def __init__(self):
        self.sim_time = 2.5
        self.sim_step = 0.025
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.drone_states[0, 0] = 0.0
        self.drone_states[0, 1] = 0.0
        self.drone_states[0, 2] = 1.0
        self.status = 0
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
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

        # +psi
        self.drone_states[0, 8] = 0.0
        self.NW = int(self.sim_time/self.sim_step)

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
        d_angle = R_d_angle@np.array([p,q,r])
        # d_angle = np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q))

        return dx 

    def run(self,waypoint0,waypoint1,psi_cmd):
        model = torch.jit.load("space_f2_psi.pt")
        model.eval()
        # Step 5: Prepare input data
        input_data = torch.randn(1, 13)  # Preprocess your input data
        last_velocity = np.zeros(3)
        for self.pointer in range(self.drone_states.shape[0]-1):      
            # input = np.zeros(15)
            t = self.pointer
            if t > 1:
                t -= 1
            else:
                t = 0
            current_position = np.array([self.drone_states[t, 0], self.drone_states[t, 1], self.drone_states[t, 2]])
            current_velocity = np.array([self.drone_states[t, 3], self.drone_states[t, 4], self.drone_states[t, 5]])
            psi = self.drone_states[t, 8]

            error0 = waypoint0 - current_position
            error1 = waypoint1 - current_position
            
            # +psi
            psi_cmd = 0.0
            error0_ = error0
            error1_ = error1
            error0_[0] = error0[0] * cos(psi) + error0[1] * sin(psi)
            error0_[1] = error0[1] * cos(psi) - error0[0] * sin(psi)
            error1_[0] = error1[0] * cos(psi) + error1[1] * sin(psi)
            error1_[1] = error1[1] * cos(psi) - error1[0] * sin(psi)
            
            acc = np.zeros(3)

            for i in range(3):
                input_data[0][i] = error0_[i]
                input_data[0][i+3] = error1_[i]
                input_data[0][i+6] = self.drone_states[t, i+3]
                acc[i] = (current_velocity[i] - last_velocity[i]) / self.sim_step
                # input_data[0][i+9] = acc[i]
                input_data[0][i+10] = self.drone_states[t, i+6]
                # input_data[0][i+12] = self.drone_states[t, i+9]
            # input_data[0][11] += 9.81
            acc[2] += 9.81

            input_data[0][9] = (acc[0]**2+acc[1]**2+acc[2]**2)**0.5

            last_velocity = current_velocity
            with torch.no_grad():
                output = model(input_data)

            self.time[self.pointer] = self.pointer * self.sim_step

            self.rate_cmd[self.pointer] = [output[0, 0], output[0, 1], output[0, 2]]
            # print(psi_cmd-psi)
            
            # acc_des = np.array([output[0, 0], output[0, 1], output[0, 2]])

            thrust_cmd = output[0, 3]


            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m
            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
        self.time[-1] = self.sim_time



    def rate_controller(self,cmd):
        kp_p = 0.085
        kp_q = 0.075
        kp_r = 0.09
        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])
    
    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real')
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,self.position_cmd[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,self.position_cmd[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6])
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        ax1[3,0].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        ax1[3,0].set_ylabel('r[rad/s]')


if __name__ == "__main__":

    # f = open("trajectories_simulation_cpc.txt",'a')      
    f = open("trajectories_simulation_torch.txt",'a')      
    
    wpx = [1,1]
    wpy = [1,1]
    wpz = [1,1]
    count = 0
    flag = 1
    for x0i in range(-2,3):
        for x1i in range(-2,3):
            for y0i in range(-2,3):
            # for y0i in range(-5,6):
                for y1i in range(-2,3):
                    for z0i in range(-2,3):
                    # for z0i in range(-5,6):
                        for z1i in range(0,1):
                            # wpx[0] = 2.5 + x0i * 0.4 + 0.3
                            # wpx[1] = 6.0 + x1i * 0.4 + 0.3
                            # wpy[0] = y0i * 0.3 + 0.075
                            # wpy[1] = y1i * 2.0
                            # wpz[0] = 1.0 + z0i * 0.3 + 0.075
                            # wpz[1] = 1.0 + z1i * 0.2

                            count += 1
                            if count + 343 == 686:
                                count = 0
                                flag = 1
                            else:
                                flag = 0
                            
                            wpx[0] = 1.0 + x0i * 0.4 + 0.2
                            wpx[1] = 4.0 + x1i * 0.4
                            wpy[0] = y0i * 0.4 + 0.2
                            wpy[1] = y0i + y1i * 0.4
                            wpz[0] = 1.0 + z0i * 0.3 +0.15
                            wpz[1] = 1.0 + z1i * 0.2

                            waypoint0 = np.array([wpx[0], wpy[0], wpz[0]])
                            waypoint1 = np.array([wpx[1], wpy[1], wpz[1]])


                            # wpx[0] = (30 + x0i) / 200.0
                            # wpx[1] = 0.3
                            # wpy[0] = y0i / 50.0
                            # wpy[1] = 0.0
                            # wpz[0] = (50 + z0i) / 50.0
                            # wpz[1] = 1.0
                            
                            # if x0i == 10 and y0i == 5 and z0i == 10 :
                            #     flag = 0

                            print(count, "/ 8820")
                            print(wpx, wpy, wpz)
                            min_error = 10000
                            if flag: #7632
                                drone = DroneControlSim()
                                drone.run(waypoint0,waypoint1,0)
                                # drone.plot_states()
                                # plt.show()

                                f.write(str(wpx[0])+',')
                                f.write(str(wpy[0])+',')
                                f.write(str(wpz[0])+',')
                                f.write(str(wpx[1])+',')
                                f.write(str(wpy[1])+',')
                                f.write(str(wpz[1])+',')  

                                counti = 0

                                for i in range(drone.NW):     
                                    counti = i                           
                                    f.write(str(drone.drone_states[i,0])+',') #x
                                    f.write(str(drone.drone_states[i,1])+',') #y
                                    f.write(str(drone.drone_states[i,2])+',') #z  
                                    if (drone.drone_states[i,0] - wpx[1])**2 + (drone.drone_states[i,1] - wpy[1])**2 + (drone.drone_states[i,2] - wpz[1])**2 < 0.04:
                                        break
                                    if (drone.drone_states[i,0] - wpx[0])**2 + (drone.drone_states[i,1] - wpy[0])**2 + (drone.drone_states[i,2] - wpz[0])**2 < min_error ** 2:
                                        min_error = ((drone.drone_states[i,0] - wpx[0])**2 + (drone.drone_states[i,1] - wpy[0])**2 + (drone.drone_states[i,2] - wpz[0])**2) ** 0.5
                                    

                                f.write(str((counti) * drone.sim_step) + ',')
                                f.write(str(min_error) + '\n')
                                                                                                           
    f.close()
