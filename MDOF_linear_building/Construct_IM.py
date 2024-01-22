#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:04:12 2023

@author: taeyongkim
"""

# Import libraries
import numpy as np
from Newmark_TK import Newmark_TK

#%% Integration
def integrate_accel(Xs, t):
    
    time = list(t)
    Vs = []
    Ds = []
    for jj in range(1):
        velo = []
        velo.append(0)
        displ = []
        displ.append(0)
        for ii in range(len(time)-1):
            temp_accel = Xs[0+ii:2+ii]
            temp = np.trapz(temp_accel.reshape(2,), x= time[0+ii:2+ii])+velo[ii]
            velo.append(temp)
            
            temp_velo = np.asarray(velo[0+ii:2+ii])
            temp2 = np.trapz(temp_velo.reshape(2,), x= time[0+ii:2+ii])+displ[ii]
            displ.append(temp2)
        Vs.append(velo)
        Ds.append(displ)
    
    del jj, ii, temp_accel, temp, temp_velo, temp2
    
    Vs = np.transpose(np.asarray(Vs))
    Ds = np.transpose(np.asarray(Ds))
    
    return Vs, Ds
#%% Estimate the features (IM) of ground motions

# Period for Housner intensity
str_period = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 
                       0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 
                       0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 
                       0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2, 
                       1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.25, 2.5])

# Sa_period
Sa_period = np.asarray([0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,
                        0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,
                        0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,
                        0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,
                        0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,
                        0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.55,0.6,0.65,0.7,0.75,
                        0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,
                        2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.2,3.4,3.6,3.8,
                        4,4.2,4.4,4.6,4.8,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
    

g = 9.8
Ground_info = []
for ii in range(2): # Number of ground motions

    # Ground motion information
    tmp_name1 = './Ground_motions/'+'GM_%d' %ii
    tmp_name2 = './Ground_motions/'+'time_%d' %ii
    tmp_name3 = './Ground_motions/'+'EQ_%d.npy' %ii

    gm = np.loadtxt(tmp_name1)      # Unit, g
    dt = np.loadtxt(tmp_name2)      # Unit, sec
    Eq = np.load(tmp_name3)         # Infromation about earthquake (A,B,C, and D as 1,2,3, and 4)

    # (1) Earthquake information
    Magnitude = Eq[0]

    Epi_dist = Eq[1]
    
    Site = int(Eq[2] - 1)    
    Site = np.eye(5)[Site]

    # (2) Ground motion information
    xgt = gm*g                   # ground acceleration times history
    gt = np.arange(len(gm))*dt   # corresponding time history
    num_step = int(len(xgt))
    total_duration = gt[-1]
    #plt.figure()
    #plt.plot(gt, xgt)
    
    # 1 .FInd strong duration
    temp = np.trapz((xgt)**2, dx= dt)
    total_temp2 = []
    for jj in range(len(xgt)-1):
        temp2 = np.trapz((xgt[0:jj+2])**2, x= gt[0:jj+2])/temp
        total_temp2.append(temp2)
    total_temp2 = np.asarray(total_temp2)

    time_005 = int(np.asarray(np.where(np.min(np.abs(total_temp2 - 0.05)) == np.abs((total_temp2 - 0.05)))))
    time_095 = int(np.asarray(np.where(np.min(np.abs(total_temp2 - 0.95)) == np.abs((total_temp2 - 0.95)))))        
    strong_duration = (np.asarray(gt)[time_095]
                       - np.asarray(gt)[time_005])
    
    strong_duration = np.round(strong_duration,4)


    # 2. zero crossing
    my_array = xgt[time_005:time_095+1]
    zeros = ((my_array[:-1] * my_array[1:]) < 0).sum()
    
    # 3. Kind of Arias intensity
    IM_AI = np.trapz((xgt)**2, dx= dt)    
    
    # 4. Housner intensity    
    # Housner's intensity (1959),i.e. kinetic energy stored in the structural system
    IM_Response_acc = []; IM_Response_vel = []; IM_Response_dis = [];
    for jj in range(len(str_period)):
        period = str_period[jj]
        w = 2*np.pi/period        
        results = Newmark_TK(1, w**2, 0.05, xgt/g, dt, dt/10, 1/2, 1/6)
        IM_Response_acc.append(np.max(np.abs(results[:,2])))
        IM_Response_vel.append(np.max(np.abs(results[:,2]))/w*9.8)
        IM_Response_dis.append(np.max(np.abs(results[:,2]))/(w**2)*9.8)

    IM_Response_acc = np.asarray(IM_Response_acc)*9.8    # unit: m/s^2 
    IM_Response_vel = np.asarray(IM_Response_vel)*100    # unit: cm/s
    IM_Response_dis = np.asarray(IM_Response_dis)*100    # unit: cm
    
    IM_EPV = []
    temp2 = np.asarray(IM_Response_vel)
    IM_EPV = np.trapz(temp2, x= str_period)


    # 5. Spectral acceleration
    IM_Response_acc = [];
    for jj in range(len(Sa_period)):
        period = Sa_period[jj]
        w = 2*np.pi/period        
        results = Newmark_TK(1, w**2, 0.05, xgt/g, dt, dt/10, 1/2, 1/6)
        IM_Response_acc.append(np.max(np.abs(results[:,2])))    

    IM_Response_acc = np.asarray(IM_Response_acc)    # unit: g
    #plt.plot(Sa_period, IM_Response_acc)

    # 6. PGA, PGV, PGD
    vgt, dgt = integrate_accel(xgt, gt)
    PGA = np.max(np.abs(xgt/g))                      # unit: g
    PGV = np.max(np.abs(vgt))*100                    # unit: cm/s
    PGD = np.max(np.abs(dgt))*100                    # unit: cm
    
    
    tmp_Ground_info={'Earthquake': np.r_[Magnitude, Epi_dist, Site],           # M, R, site
                     'Peak_value': np.r_[PGA, PGV, PGD],                       # PGA, PGV, PGD
                     'Spect_acce': IM_Response_acc,                            # Sepctral acceleration
                     'Duration': total_duration,
                     'Duration_strong': strong_duration,
                     'zerocross_strong': zeros,
                     'Arias_intensity': IM_AI,
                     'Housner': IM_EPV,
                     'RSN' : ii,
                     }
    
    Ground_info.append(tmp_Ground_info)    # Ground_info[0]{'Earthquake'}    

# Save the dataset
np.save('./Ground_motions/Ground_info_artificial.npy', Ground_info) 
    