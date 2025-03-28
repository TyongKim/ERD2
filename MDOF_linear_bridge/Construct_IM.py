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
num_GM = 100
g = 9.8
Ground_info = []
# Load each ground motions
idx = 0
for ii in range(num_GM):
    
    # Load ground motion
    gm1 = np.loadtxt('./Ground_motion/GM1_%d.txt' %(ii))
    gm2 = np.loadtxt('./Ground_motion/GM2_%d.txt' %(ii))
    gm_td = np.loadtxt('./Ground_motion/time_0' )
    
    IM_Response_acc1 = [];IM_Response_acc2 = [];
    for jj in range(len(Sa_period)):
        period = Sa_period[jj]
        w = 2*np.pi/period        
        results1 = Newmark_TK(1, w**2, 0.05, gm1, gm_td, gm_td/10, 1/2, 1/6)
        results2 = Newmark_TK(1, w**2, 0.05, gm2, gm_td, gm_td/10, 1/2, 1/6)
        
        IM_Response_acc1.append(np.max(np.abs(results1[:,2])))
        IM_Response_acc2.append(np.max(np.abs(results2[:,2])))


    
    IM_Response_acc1 = np.asarray(IM_Response_acc1) # unit g
    IM_Response_acc2 = np.asarray(IM_Response_acc2) # unit g
    

    tmp_Ground_info={'Spec_value1': IM_Response_acc1,
                     'Spec_value2': IM_Response_acc2,
                     'Peak_value1': np.max(np.abs(gm1)),
                     'Peak_value2': np.max(np.abs(gm2))}
    
    Ground_info.append(tmp_Ground_info)    # Ground_info[0]{'Earthquake'}    

# Save the dataset
np.save('./Ground_info/Ground_info_bridge.npy', Ground_info) 