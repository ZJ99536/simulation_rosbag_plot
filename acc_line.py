import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,atan2,asin
from tensorflow import keras
import seaborn as sns
from numpy import loadtxt

class DroneControlSim:
    def __init__(self, lim):
        self.sim_time = 4.3
        self.sim_step = 0.027
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 15))
        self.drone_states[0, 0] = 0.0
        self.drone_states[0, 1] = 0.0
        self.drone_states[0, 2] = 1.0
        self.status = 0
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.acc_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 1.0
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])

        self.limv = lim

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
    
    def trajPlanning(self, t, p, v, a):
        xMatrix = np.zeros((4*len(t)+2,1))
        for i in range(len(p)-1):
            xMatrix[i,0] = p[i]
            xMatrix[len(t)+i,0] = p[i+1]

        xMatrix[-4,0] = v[0]
        xMatrix[-3,0] = a[0]
        xMatrix[-2,0] = v[1]
        xMatrix[-1,0] = a[1]

        tMatrix = np.zeros((4*len(t)+2,4*len(t)+2)) 
        tMatrix[0,4] = 1 #p0(0)
        tMatrix[len(t)-1,-1] = 1 #pn(0)
        for j in range(5):
            tMatrix[len(t),j] = t[0]**(4-j) #p0(t)
            tMatrix[2*len(t)-1,-1-j] = t[-1]**j #pn(t)

        for j in range(4):
            tMatrix[2*len(t),j] = (4-j)*t[0]**(3-j) #v0(t)   
        for j in range(3):         
            tMatrix[3*len(t)-1,j] = (4-j)*(3-j)*t[0]**(2-j) #a0(t)
            
        tMatrix[3*len(t)-2,-2] = -1 #-vn(0)
        tMatrix[4*len(t)-3,-3] = -2 #-an(0)

        tMatrix[-4,3] = 1 #v0
        tMatrix[-3,2] = 2 #a0

        for j in range(4):
            tMatrix[-2,j-5] = (4-j)*t[-1]**(3-j) #vt
        for j in range(3):         
            tMatrix[-1,j-5] = (4-j)*(3-j)*t[-1]**(2-j) #at

        kMatrix = np.matmul(np.linalg.inv(tMatrix),xMatrix)
        return kMatrix

    def planOnce(self, tseg0, tseg1, waypoint1,point,velocity,acc,current_position,current_velocity,current_acc):
        t = np.ones(2)
        t[0] = tseg0
        t[1] = tseg1
        px = np.zeros(3)
        px[0] = current_position[0]
        px[1] = waypoint1[0]
        px[2] = point[0]
        py = np.zeros(3)
        py[0] = current_position[1]
        py[1] = waypoint1[1]
        py[2] = point[1]
        pz = np.zeros(3)
        pz[0] = current_position[2]
        pz[1] = waypoint1[2]
        pz[2] = point[2]
        vx = np.zeros(2)
        vx[0] = current_velocity[0]
        vx[1] = velocity[0]
        vy = np.zeros(2)
        vy[0] = current_velocity[1]
        vy[1] = velocity[1]
        vz = np.zeros(2)
        vz[0] = current_velocity[2]
        vz[1] = velocity[2]
        ax = np.zeros(2)
        ax[0] = current_acc[0]
        ax[1] = acc[0]
        ay = np.zeros(2)
        ay[0] = current_acc[1]
        ay[1] = acc[1]
        az = np.zeros(2)
        az[0] = current_acc[2]
        az[1] = acc[2] - 9.81
        polyx = self.trajPlanning(t,px,vx,ax)
        polyy = self.trajPlanning(t,py,vy,ay)
        polyz = self.trajPlanning(t,pz,vz,az)
        return polyx,polyy,polyz
    
    

    def predictOnce(self, point0,waypoint3,waypoint4,limit,model):
        point = point0
        velocity = np.array([0.0, 0.0, 0.0])
        acc = np.array([0.0, 0.0, 9.81])
        pqr = np.array([0.0, 0.0, 0.0])

        input = np.zeros(15)
        # print(current_position)
        error3 = waypoint3 - point
        error4 = waypoint4 - point
        for i in range(3):
            input[i] = error3[i]
            input[i+3] = error4[i]
            input[i+6] = velocity[i]
            input[i+9] = acc[i]
            input[i+12] = pqr[i]
        output = model(input.reshape(-1,15))

        count = 0
        if abs(output[0, 9]) < abs(acc[0]):
            count += 1
        
        # while count < 4 :
        while point[0] < limit :
            point[0] = (((waypoint3[0] - output[0, 0]) + (waypoint4[0] - output[0, 3])) / 2)
            point[1] = (((waypoint3[1] - output[0, 1]) + (waypoint4[1] - output[0, 4])) / 2)
            point[2] = (((waypoint3[2] - output[0, 2]) + (waypoint4[2] - output[0, 5])) / 2)
            velocity[0] = output[0, 6]
            velocity[1] = output[0, 7]
            velocity[2] = output[0, 8]
            acc[0] = output[0, 9]
            acc[1] = output[0, 10]
            acc[2] = output[0, 11]
            pqr[0] = output[0, 12]
            pqr[1] = output[0, 13]
            pqr[2] = output[0, 14]

            input = np.zeros(15)
            # print(current_position)
            error3 = waypoint3 - point
            error4 = waypoint4 - point
            for i in range(3):
                input[i] = error3[i]
                input[i+3] = error4[i]
                input[i+6] = velocity[i]
                input[i+9] = acc[i]
                input[i+12] = pqr[i]
            output = model(input.reshape(-1,15))

            if abs(output[0, 9]) < abs(acc[0]):
                count += 1
            else:
                count = 0
            
        return point,velocity,acc
    
    def df_control(self,position_cmd,velocity_cmd,aref,j):
        pos_err = position_cmd - self.drone_states[self.pointer,0:3]
        vel_err = velocity_cmd - self.drone_states[self.pointer,3:6]
        psi = self.drone_states[self.pointer,8]
        R_E_B = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        vel_err = R_E_B @ vel_err
        pos_err = R_E_B @ pos_err

        K_pos = np.array([[5,0,0],[0,5,0],[0,0,5]])
        K_vel = np.array([[3.5,0,0],[0,3.5,0],[0,0,3.5]])
        ades = aref + self.g*np.array([0,0,1]) + K_pos @ pos_err + K_vel @ vel_err                              
        acc_des = ades
        # print(acc_des[2])

        if acc_des[2] < 0.1:
            acc_des[2] = 0.1   

        psi = 0
        z_b_des = np.array(acc_des / np.linalg.norm(acc_des))
        y_c = np.array([-sin(psi),cos(psi),0])
        x_b_des = np.cross(y_c,z_b_des) / np.linalg.norm(np.cross(y_c,z_b_des))
        y_b_des = np.cross(z_b_des,x_b_des)
        
        R_E_B = np.transpose(np.array([x_b_des,y_b_des,z_b_des]))

        psi_cmd = atan2(R_E_B[1,0],R_E_B[0,0])
        theta_cmd = asin(-R_E_B[2,0])
        phi_cmd = atan(R_E_B[2,1]/R_E_B[2,2])        
        thrust_cmd = np.linalg.norm(acc_des)

        theta_cmd = self.bound(theta_cmd,-1.5,1.5)
        phi_cmd = self.bound(phi_cmd,-1.5,1.5)
        # thrust_cmd = self.bound(thrust_cmd,0,0.9)

        psi = 0
        yc = np.array([-sin(psi),cos(psi),0])
        alpha = aref + self.g*np.array([0,0,1])
        xb = np.cross(yc,alpha)
        # print(xb)
        xb = xb / np.linalg.norm(xb)
        yb = np.cross(alpha,xb)
        yb = yb / np.linalg.norm(yb)
        zb = np.cross(xb, yb)
        c = np.dot(zb, alpha)
        w = np.zeros(3)
        w[0] = -np.dot(yb,j)/c
        w[1] = np.dot(xb,j)/c
        w[2] = w[1]*np.dot(yc,zb)/np.linalg.norm(np.cross(yc,zb))

        att_cmd = np.array([phi_cmd,theta_cmd,psi_cmd])
        w_fb = self.attitude_controller(att_cmd)
        w_cmd = w + w_fb

        return w_cmd,thrust_cmd



    def bound(self,data,min_value,max_value):
        if data >=max_value:
            data = max_value
        elif data <=min_value:
            data = min_value
        return data


    def run(self):
        
        # model = keras.models.load_model('/home/zhoujin/learning/model/quad5_m5.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        # model = keras.models.load_model('/home/zhoujin/learning/model/quad4_75t2.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        model0 = keras.models.load_model('/home/zhoujin/learning/model/quadb2_8m.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        model1 = keras.models.load_model('/home/zhoujin/learning/model/quadf2_8m.h5') # quad5 m4 m6(softplus 64) m5(softplus 640)
        last_velocity = np.zeros(3)
        point0,velocity0,acc0 = self.predictOnce(np.array([6.5, 0.0, 1.0]),np.array([10.0, 1.0, 0.6]),np.array([14.0, 2.0, 1.0]),8.0 + self.limv * 2.0,model1)
        point1,velocity1,acc1 = self.predictOnce(np.array([14.0, 2.0, 1.0]),np.array([17.2, 3.0, 1.4]),np.array([20.5, 4.0, 1.0]),15.5,model1)
        point2,velocity2,acc2 = self.predictOnce(np.array([3.0, 0.0, 1.0]),np.array([0.0, -1.0, 0.5]),np.array([-3.0, -2.0, 1.0]),1.5,model0)
        print(point0)
        print(point1)
        print(point2)

        kmv0 = 0.4
        kmv = 0.4

        count = 0

        max_v = 0
    
        for self.pointer in range(self.drone_states.shape[0]-1):      
            input = np.zeros(15)
            t = self.pointer
            if t > 2:
                t -= 2
            else:
                t = 0
            current_position = np.array([self.drone_states[t, 0], self.drone_states[t, 1], self.drone_states[t, 2]])
            current_velocity = np.array([self.drone_states[t, 3], self.drone_states[t, 4], self.drone_states[t, 5]])
            current_acc = np.array([self.drone_states[t, 12], self.drone_states[t, 13], self.drone_states[t, 14]])
            if self.status == 0 :
                model = model1
                waypoint0 = np.array([3.5, -1.0, 1.5])
                waypoint1 = np.array([6.5, 0.0, 1.0])
                if self.limv < 0:
                    error = waypoint1 - current_position
                    if error[0]**2 + error[1]**2 + error[2]**2 < 0.25:
                        self.status = 1
                else: 
                    if self.pointer > 1:
                        # if self.drone_states[self.pointer-1,12] < 0 and self.drone_states[self.pointer-1,12] + self.drone_states[self.pointer-3,12] > 2 * self.drone_states[self.pointer-2,12]:
                        if current_acc[0] < 0 and current_velocity[0] < max_v * self.limv:
                            count += 1
                        else :
                            count = 0
                        if max_v < current_velocity[0] :
                            max_v = current_velocity[0]
                        if count > 0:
                            self.status = 4
                            waypoint1 = np.array([6.5, 0.0, 1.0])
                            ts0 = abs(current_position[0] - waypoint1[0])**1 * kmv0
                            ts1 = abs(point0[0] - waypoint1[0])**1 * kmv
                            qqq = abs(current_position[0] - waypoint1[0]) / (abs((point0[0] - waypoint1[0]) + abs(point0[0] - waypoint1[0])))
                            ts0 = abs((current_position[0] - waypoint1[0]) / ((1-qqq) * current_velocity[0] + qqq * velocity0[0])) * 0.95
                            ts1 = abs((point0[0] - waypoint1[0]) / (qqq * current_velocity[0] + (1-qqq) * velocity0[0])) * 0.95
                            # qqq = abs(current_position[0] - waypoint1[0]) / (abs((current_position[0] - waypoint1[0]) + abs(point0[0] - waypoint1[0])))
                            # v_mid = (1-qqq) * current_velocity[0] + qqq * velocity0[0]
                            # ts0 = abs(2*(current_position[0] - waypoint1[0]) / (current_velocity[0] + v_mid)) * 1.3
                            # ts1 = abs(2*(point0[0] - waypoint1[0]) / (velocity0[0] + v_mid)) * 1.3
                            print(ts0,ts1)
                            print(current_position[0])
                            
                            position = np.array([self.drone_states[self.pointer, 0], self.drone_states[self.pointer, 1], self.drone_states[self.pointer, 2]])
                            velocity = np.array([self.drone_states[self.pointer, 3], self.drone_states[self.pointer, 4], self.drone_states[self.pointer, 5]])
                            acc = np.array([self.drone_states[self.pointer, 12], self.drone_states[self.pointer, 13], self.drone_states[self.pointer, 14]])
                            polyx,polyy,polyz = self.planOnce(ts0,ts1,waypoint1,point0,velocity0,acc0,position,velocity,acc)
                            polyt = self.pointer
                            print(polyt)
                            count = 0
            if self.status == 1 :
                model = model1
                waypoint0 = np.array([10.0, 1.0, 0.6])
                waypoint1 = np.array([14.0, 2.0, 1.0])
                error = waypoint1 - current_position
                # if error[0]**2 + error[1]**2 + error[2]**2 < 0.2:
                #     self.status = 2
                # if current_position[0] < -1.0:
                if self.drone_states[self.pointer,0] > 12.5:
                # if self.drone_states[self.pointer-1,12] > 0 and self.drone_states[self.pointer-1,12] + self.drone_states[self.pointer-3,12] > 2 * self.drone_states[self.pointer-2,12]:
                    count += 1
                else :
                    count = 0
                # if count > 0:
                #     self.status = 5
                #     waypoint1 = np.array([14.0, 2.0, 1.0])
                #     # ts0 = abs(current_position[0] - waypoint1[0])**0.5 * kmv / abs(self.drone_states[self.pointer,3])
                #     # ts1 = abs(point1[0] - waypoint1[0])**0.5 * kmv / abs(self.drone_states[self.pointer,3])
                #     ts0 = abs(current_position[0] - waypoint1[0])**0.5 * kmv0
                #     ts1 = abs(point1[0] - waypoint1[0])**0.5 * kmv
                #     print(current_position[0])
                    
                #     position = np.array([self.drone_states[self.pointer, 0], self.drone_states[self.pointer, 1], self.drone_states[self.pointer, 2]])
                #     velocity = np.array([self.drone_states[self.pointer, 3], self.drone_states[self.pointer, 4], self.drone_states[self.pointer, 5]])
                #     acc = np.array([self.drone_states[self.pointer, 12], self.drone_states[self.pointer, 13], self.drone_states[self.pointer, 14]])
                #     polyx,polyy,polyz = self.planOnce(ts0,ts1,waypoint1,point1,velocity1,acc1,position,velocity,acc)
                #     polyt = self.pointer
                #     print(polyt)
            if self.status == 2 :
                model = model1
                waypoint0 = np.array([17.2, 3.0, 1.4])
                waypoint1 = np.array([20.5, 4.0, 1.0])
                error = waypoint1 - current_position
                # if error[0]**2 + error[1]**2 + error[2]**2 < 0.2:
                #     self.status = 3
                # if current_position[0] > 0 and self.drone_states[self.pointer-1,12] < 0 and self.drone_states[self.pointer-1,12] + self.drone_states[self.pointer-3,12] > 2 * self.drone_states[self.pointer-2,12]:
                #     count += 1
                # else :
                #     count = 0
                # if count > 1:
                #     self.status = 6
                #     waypoint1 = np.array([2.9, 0.0, 1.0])
                #     # ts0 = abs(current_position[0] - waypoint1[0])**0.5 * kmv / abs(self.drone_states[self.pointer,3])
                #     # ts1 = abs(point2[0] - waypoint1[0])**0.5 * kmv / abs(self.drone_states[self.pointer,3])                    
                #     ts0 = abs(current_position[0] - waypoint1[0])**0.5 * kmv0
                #     ts1 = abs(point2[0] - waypoint1[0])**0.5 * kmv
                #     print(current_position[0])
                    
                #     position = np.array([self.drone_states[self.pointer, 0], self.drone_states[self.pointer, 1], self.drone_states[self.pointer, 2]])
                #     velocity = np.array([self.drone_states[self.pointer, 3], self.drone_states[self.pointer, 4], self.drone_states[self.pointer, 5]])
                #     acc = np.array([self.drone_states[self.pointer, 12], self.drone_states[self.pointer, 13], self.drone_states[self.pointer, 14]])
                #     polyx,polyy,polyz = self.planOnce(ts0,ts1,waypoint1,point2,velocity2,acc2,position,velocity,acc)
                #     polyt = self.pointer
                #     print(polyt)
            if self.status == 3 :
                model = model0
                waypoint0 = np.array([0.0, -1.0, 0.6])
                waypoint1 = np.array([-3.0, -2.0, 1.0])
                error = waypoint1 - current_position
                if error[0]**2 + error[1]**2 + error[2]**2 < 0.:
                    self.status = 3

            if self.status >= 4:
                current_t = (self.pointer - polyt) * self.sim_step
                if current_t < ts0 :
                    ts = current_t
                    tt = np.array([ts**4, ts**3, ts**2, ts, 1])
                    vt = np.array([4*ts**3, 3*ts**2, 2*ts, 1, 0])
                    at = np.array([12*ts**2, 6*ts, 2, 0, 0])
                    jt = np.array([24*ts, 6, 0, 0, 0])
                    aax = np.array([polyx[0],polyx[1],polyx[2],polyx[3],polyx[4]])
                    aay = np.array([polyy[0],polyy[1],polyy[2],polyy[3],polyy[4]])
                    aaz = np.array([polyz[0],polyz[1],polyz[2],polyz[3],polyz[4]])
                elif current_t < ts0+ts1 :
                    ts = current_t - ts0
                    tt = np.array([ts**4, ts**3, ts**2, ts, 1])
                    vt = np.array([4*ts**3, 3*ts**2, 2*ts, 1, 0])
                    at = np.array([12*ts**2, 6*ts, 2, 0, 0])
                    jt = np.array([24*ts, 6, 0, 0, 0])
                    aax = np.array([polyx[5],polyx[6],polyx[7],polyx[8],polyx[9]])
                    aay = np.array([polyy[5],polyy[6],polyy[7],polyy[8],polyy[9]])
                    aaz = np.array([polyz[5],polyz[6],polyz[7],polyz[8],polyz[9]])
                else :
                    self.status = self.status - 3            
            

            self.time[self.pointer] = self.pointer * self.sim_step
            psi_cmd = 0.0
            
            if self.status < 4:
                # print(self.status)
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
                x_cmd = ((waypoint0[0] - output[0, 0]) + (waypoint1[0] - output[0, 3])) / 2
                y_cmd = ((waypoint0[1] - output[0, 1]) + (waypoint1[1] - output[0, 4])) / 2
                z_cmd = ((waypoint0[2] - output[0, 2]) + (waypoint1[2] - output[0, 5])) / 2
            
                self.position_cmd[self.pointer] = [x_cmd, y_cmd, z_cmd]
                self.velocity_cmd[self.pointer] = [output[0, 6], output[0, 7], output[0, 8]]
                
                acc_des = np.array([output[0, 9], output[0, 10], output[0, 11]])
                psi = 0
                z_b_des = np.array(acc_des / np.linalg.norm(acc_des))
                # print(acc_des.shape)
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

            else:
                x_cmd = np.dot(tt, aax)
                y_cmd = np.dot(tt, aay)
                z_cmd = np.dot(tt, aaz)
                vx_cmd = np.dot(vt, aax)
                vy_cmd = np.dot(vt, aay)
                vz_cmd = np.dot(vt, aaz)
                ax_cmd = np.dot(at, aax)
                ay_cmd = np.dot(at, aay)
                az_cmd = np.dot(at, aaz)
                jx_cmd = np.dot(jt, aax)
                jy_cmd = np.dot(jt, aay)
                jz_cmd = np.dot(jt, aaz)
                self.position_cmd[self.pointer] = [x_cmd[0], y_cmd[0], z_cmd[0]]
                self.velocity_cmd[self.pointer] = [vx_cmd[0], vy_cmd[0], vz_cmd[0]]
                self.acc_cmd[self.pointer] = [ax_cmd[0], ay_cmd[0], az_cmd[0]]
                acc_des = np.array([ax_cmd[0], ay_cmd[0], az_cmd[0]])
                j_des = np.array([jx_cmd[0], jy_cmd[0], jz_cmd[0]])

                self.rate_cmd[self.pointer], thrust_cmd = self.df_control(self.position_cmd[self.pointer], self.velocity_cmd[self.pointer], acc_des,j_des)
            # print(self.status)
            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m
            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
        self.time[-1] = self.sim_time


    def attitude_controller(self,cmd):
        kp_phi = 3.5 
        kp_theta = 3.5 
        kp_psi = 3.5
        error = cmd - self.drone_states[self.pointer,6:9]
        return np.array([kp_phi*error[0],kp_theta*error[1],kp_psi*error[2]])

    def rate_controller(self,cmd):
        kp_p = 0.06
        kp_q = 0.06
        kp_r = 0.06
        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])

   

    def plot_states(self,ax1):
        
        # WN&CNet = loadtxt('/home/zhoujin/trajectory-generation/trajectory/nn_line.txt', delimiter=',')

        # f = open("/home/zhoujin/trajectory-generation/trajectory/nn_line.txt",'a')
        # count = 0
        # for i in self.drone_states[:,0:1]:
        #     if count < len(self.drone_states[:,0:1]) - 1:
        #         f.write(str(self.drone_states[:,0:1]) + ',')
        #     else:
        #         f.write(str(self.drone_states[:,0:1]))
        # f.write('\n')

        
        self.position_cmd[-1] = self.position_cmd[-2]
        # ax1[0,0].plot(self.time * 6.5/7,self.drone_states[:,0] * 6.5/7,label='real')
        if self.limv < 0:
            ax1[0].plot(self.time,self.drone_states[:,0],label='WN&CNet (stop and go)',linewidth = 2.76, alpha = 1, color = 'cornflowerblue')
        else:
            ax1[0].plot(self.time,self.drone_states[:,0],label='WN&CNet+Transition, $v_{in}=$'+str(self.limv)+'$v_{max}$',linewidth = 2.76, alpha = 1 - (1-self.limv)*3,color = 'mediumaquamarine')
        # ax1[0].plot(self.time,WN&CNet[:,0:1],label='WN&CNet',linewidth = 2.76, color = 'cornflowerblue')
        # ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0].set_ylabel('x[$m$]', fontsize=24)
        # ax1[0,1].plot(self.time,self.drone_states[:,1])
        # # ax1[0,1].plot(self.time,self.position_cmd[:,1])
        # ax1[0,1].plot(cpc[:,0:1], cpc[:,2:3])
        # ax1[0,1].set_ylabel('y[m]')
        # ax1[0,2].plot(self.time,self.drone_states[:,2])
        # # ax1[0,2].plot(self.time,self.position_cmd[:,2])
        # ax1[0,2].plot(cpc[:,0:1], cpc[:,3:4])
        # ax1[0,2].set_ylabel('z[m]')
        # ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]

        if self.limv < 0:
            ax1[1].plot(self.time,self.drone_states[:,3],label='WN&CNet (stop and go)',linewidth = 2.76, alpha = 1, color = 'cornflowerblue')
        else:
            ax1[1].plot(self.time, self.drone_states[:,3],label='WN&CNet+Transition, $v_{in}=$'+str(self.limv)+'$v_{max}$',linewidth = 2.76, alpha = 1 - (1-self.limv)*3,color = 'mediumaquamarine')
        

        # ax1[1].plot(self.time,self.drone_states[:,3],linewidth = 2.76, alpha = 1 - (1-self.limv) * 1, color = 'mediumaquamarine')
        # ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        
        ax1[1].set_ylabel('vx[$m/s$]', fontsize=24)
        ax1[1].set_xlabel('time [$s$]', fontsize=24)
        # ax1[1,1].plot(self.time,self.drone_states[:,4])
        # # ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        # ax1[1,1].plot(cpc[:,0:1], cpc[:,9:10])
        # ax1[1,1].set_ylabel('vy[m/s]')
        # ax1[1,2].plot(self.time,self.drone_states[:,5])
        # # ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        # ax1[1,2].plot(cpc[:,0:1], cpc[:,10:11])
        # ax1[1,2].set_ylabel('vz[m/s]')

        # self.acc_cmd[-1] = self.acc_cmd[-2]
        # ax1[2,0].plot(self.time,self.drone_states[:,12])
        # # ax1[2,0].plot(self.time,self.acc_cmd[:,0])
        # ax1[2,0].plot(cpc[:,0:1], cpc[:,5:6])
        # ax1[2,0].set_ylabel('ax[m/s]')
        # ax1[2,1].plot(self.time,self.drone_states[:,13])
        # # ax1[2,1].plot(self.time,self.acc_cmd[:,1])
        # ax1[2,1].plot(cpc[:,0:1], cpc[:,6:7])
        # ax1[2,1].set_ylabel('ay[m/s]')
        # ax1[2,2].plot(self.time,self.drone_states[:,14])
        # # ax1[2,2].plot(self.time,self.acc_cmd[:,2])
        # ax1[2,2].plot(cpc[:,0:1], cpc[:,7:8])
        # ax1[2,2].set_ylabel('az[m/s]')

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

if __name__ == "__main__":
    sns.set(style="darkgrid", font_scale=1.0)
    fig1, ax1 = plt.subplots(2,1)
    drone = DroneControlSim(-1)
    drone.run()
    drone.plot_states(ax1)
    drone = DroneControlSim(1.0)
    drone.run()
    drone.plot_states(ax1)
    drone = DroneControlSim(0.9)
    drone.run()
    drone.plot_states(ax1)
    drone = DroneControlSim(0.8)
    drone.run()
    drone.plot_states(ax1)
    cpc = loadtxt('/home/zhoujin/trajectory-generation/trajectory/line_cpc.txt', delimiter=',')
    snap = loadtxt('/home/zhoujin/trajectory-generation/trajectory/line_snap.txt', delimiter=',')
    ax1[0].plot(snap[0,:], snap[1,:],label='MSTG&C',linewidth = 2.76, color = 'hotpink',alpha = 0.8, ls = (5, (10, 3)))
    ax1[0].plot(cpc[:,0:1], cpc[:,1:2],label='CPC',linewidth = 2.76, color = 'lightcoral', alpha = 0.7, ls = '--')
    ax1[1].plot(cpc[:,0:1], cpc[:,8:9],linewidth = 2.76, color = 'lightcoral', alpha = 0.7, ls = '--')
    ax1[1].plot(snap[0,:], snap[2,:],linewidth = 2.76, color = 'hotpink',alpha = 0.8, ls = (5, (10, 3)))

    
    
    ax1[0].legend(prop = {'size':16})
    plt.show()