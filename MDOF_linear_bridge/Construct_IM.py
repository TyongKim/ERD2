#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:27:40 2023

@author: taeyongkim
"""
import numpy as np
from Newmark_TK import Newmark_TK

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

# Calculate response spectrum
num_GM = 3
g = 9.8
Ground_info = []
# Load each ground motions
idx = 0
for ii in range(num_GM):
    
    # Load ground motion
    gm1 = np.loadtxt('./Ground_motion/GM1_%d' %(ii))
    gm2 = np.loadtxt('./Ground_motion/GM1_%d' %(ii))
    gm_td = np.loadtxt('./Ground_motion/time_%d' %(ii))
    
    IM_Response_acc1 = [];IM_Response_acc2 = [];
    for jj in range(len(Sa_period)):
        period = Sa_period[jj]
        w = 2*np.pi/period        
        results1 = Newmark_TK(1, w**2, 0.05, gm1, gm_td, gm_td/10, 1/2, 1/6)
        results2 = Newmark_TK(1, w**2, 0.05, gm2, gm_td, gm_td/10, 1/2, 1/6)
        
        IM_Response_acc1.append(np.max(np.abs(results1[:,2])))
        IM_Response_acc2.append(np.max(np.abs(results2[:,2])))


    
    IM_Response_acc2 = np.asarray(IM_Response_acc2)*9.8    # unit: m/s^2 
    
    # Save all Intensity measure
    temp_RSN = np.int32(RSN[ii])
    Query_String3 = "SELECT * FROM EQ_Records WHERE \"Record Sequence Number\" = %d " %(temp_RSN)
    db1.execute(Query_String3) 
    bcd_origin = db1.fetchall()
    tmp_Ground_info={'Earthquake': np.r_[Magnitude[ii], bcd_origin[0][3], Site[ii,:]], # M, R, site
                     'Peak_value1': bcd_origin[0][45:48],                               # PGA, PGV, PGD
                     'Spect_acce1': bcd_origin[0][48:158],                               # Sepctral acceleration
                     'Spect_acc_time1': IM_Response_time1,
                     
                     'Peak_value2': bcd_origin[0][158:161],                               # PGA, PGV, PGD
                     'Spect_acce2': bcd_origin[0][161:271],                               # Sepctral acceleration
                     'Spect_acc_time2': IM_Response_time2,
                     
                     'RSN' : temp_RSN,
                     }
    
    Ground_info.append(tmp_Ground_info)    # Ground_info[0]{'Earthquake'}    
    print(ii)

# Save the dataset
np.save('Ground_info_bridge.npy', Ground_info) 

"""    
import matplotlib.pyplot as plt
plt.close('all'); plt.figure()
plt.plot(Sa_period, bcd_origin[0][48:158], 'k')
plt.plot(Sa_period, IM_Response_acc1/9.8, 'r--')
"""

"""
IM_Response_acc = []; IM_Response_vel = []; IM_Response_dis = [];
for jj in range(len(str_period)):
    period = str_period[jj]
    w = 2*np.pi/period        
    results = Newmark_TK(1, w**2, 0.05, gm_info2[4], gm_info2[2], gm_info2[2]/10, 1/2, 1/6)
    IM_Response_acc.append(np.max(np.abs(results[:,2])))

IM_Response_acc = np.asarray(IM_Response_acc)*9.8    # unit: m/s^2 

# Check the values
import matplotlib.pyplot as plt
plt.close('all'); plt.figure()
plt.plot(Sa_period, bcd_origin[0][161:271], 'k')
plt.plot(str_period, IM_Response_acc/9.8, 'r--')

# Load
#ggm = np.load('Ground_info_final.npy',allow_pickle='TRUE')
#ggm[0]
#aa = np.load('GM_dir.npy')
"""
    