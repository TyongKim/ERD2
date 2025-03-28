"""
This script is to perform dynamic anlaysis of multi-degree-of-freedom system.
OpenSeespy is employed when performing dynamic analysis.

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""

import numpy as np
from i_MDOF_ops import OPS_3span, OPS_4span, OPS_5span, OPS_6span
import pickle
#%% Load the MDOF systems
with open('./generated_MDOF_systems/Example_3_span.pickle', 'rb') as f:
    p3_value = pickle.load(f)
    
with open('./generated_MDOF_systems/Example_4_span.pickle', 'rb') as f:
    p4_value = pickle.load(f)
    
with open('./generated_MDOF_systems/Example_5_span.pickle', 'rb') as f:
    p5_value = pickle.load(f)
    
with open('./generated_MDOF_systems/Example_6_span.pickle', 'rb') as f:
    p6_value = pickle.load(f)

p3_value = p3_value[0][0]
p4_value = p4_value[0][0]
p5_value = p5_value[0][0]
p6_value = p6_value[0][0]

#%% 3-span
Results_displ = []
Results_accel = []
for ii in range(100): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'            
     
        
    # OpenSees analysis
    [results, results1, results2] = OPS_3span(p3_value, tmp1_1, tmp1_2, tmp2)

    results_displ = np.array([np.max(np.sqrt(results[:,1]**2+results[:,2]**2)),
                              np.max(np.sqrt(results[:,3]**2+results[:,4]**2)),
                              np.max(np.sqrt(results[:,5]**2+results[:,6]**2)),
                              ])
    
    results_accel = np.array([np.max(np.sqrt(results1[:,1]**2+results2[:,1]**2)),
                              np.max(np.sqrt(results1[:,2]**2+results2[:,2]**2)),
                              np.max(np.sqrt(results1[:,3]**2+results2[:,3]**2)),
                              ])
    
    Results_displ.append(results_displ)
    Results_accel.append(results_accel)

        
np.save('./Results_Summary_MDOF/Results_3span_displ.npy', Results_displ)
np.save('./Results_Summary_MDOF/Results_3span_accel.npy', Results_accel)

#%% 4-span
Results_displ = []
Results_accel = []
for ii in range(100): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'            
     

    # OpenSees analysis
    [results, results1, results2] = OPS_4span(p4_value, tmp1_1, tmp1_2, tmp2)

    
    results_displ = np.array([np.max(np.sqrt(results[:,1]**2+results[:,2]**2)),
                             np.max(np.sqrt(results[:,3]**2+results[:,4]**2)),
                             np.max(np.sqrt(results[:,5]**2+results[:,6]**2)),
                             np.max(np.sqrt(results[:,7]**2+results[:,8]**2)),
                             ])

    results_accel = np.array([np.max(np.sqrt(results1[:,1]**2+results2[:,1]**2)),
                             np.max(np.sqrt(results1[:,2]**2+results2[:,2]**2)),
                             np.max(np.sqrt(results1[:,3]**2+results2[:,3]**2)),
                             np.max(np.sqrt(results1[:,4]**2+results2[:,4]**2))
                             ])
    
    Results_displ.append(results_displ)
    Results_accel.append(results_accel)

np.save('./Results_Summary_MDOF/Results_4span_displ.npy', Results_displ)
np.save('./Results_Summary_MDOF/Results_4span_accel.npy', Results_accel)

#%% 5-span
Results_displ = []
Results_accel = []
for ii in range(100): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'            
         
     

    # OpenSees analysis
    [results, results1, results2] = OPS_5span(p5_value, tmp1_1, tmp1_2, tmp2)

    results_displ = np.array([np.max(np.sqrt(results[:,1]**2+results[:,2]**2)),
                             np.max(np.sqrt(results[:,3]**2+results[:,4]**2)),
                             np.max(np.sqrt(results[:,5]**2+results[:,6]**2)),
                             np.max(np.sqrt(results[:,7]**2+results[:,8]**2)),
                             np.max(np.sqrt(results[:,9]**2+results[:,10]**2)),
                             ])
    
    results_accel = np.array([np.max(np.sqrt(results1[:,1]**2+results2[:,1]**2)),
                             np.max(np.sqrt(results1[:,2]**2+results2[:,2]**2)),
                             np.max(np.sqrt(results1[:,3]**2+results2[:,3]**2)),
                             np.max(np.sqrt(results1[:,4]**2+results2[:,4]**2)),
                             np.max(np.sqrt(results1[:,5]**2+results2[:,5]**2))
                             ])

    Results_displ.append(results_displ)
    Results_accel.append(results_accel)

np.save('./Results_Summary_MDOF/Results_5span_displ.npy', Results_displ)
np.save('./Results_Summary_MDOF/Results_5span_accel.npy', Results_accel)

#%% 6-span
Results_displ = []
Results_accel = []
for ii in range(100): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'               
     

    # OpenSees analysis
    [results, results1, results2] = OPS_6span(p6_value, tmp1_1, tmp1_2, tmp2)

    results_displ = np.array([np.max(np.sqrt(results[:,1]**2+results[:,2]**2)),
                             np.max(np.sqrt(results[:,3]**2+results[:,4]**2)),
                             np.max(np.sqrt(results[:,5]**2+results[:,6]**2)),
                             np.max(np.sqrt(results[:,7]**2+results[:,8]**2)),
                             np.max(np.sqrt(results[:,9]**2+results[:,10]**2)),
                             np.max(np.sqrt(results[:,11]**2+results[:,12]**2)),
                             ])
    
    results_accel = np.array([np.max(np.sqrt(results1[:,1]**2+results2[:,1]**2)),
                             np.max(np.sqrt(results1[:,2]**2+results2[:,2]**2)),
                             np.max(np.sqrt(results1[:,3]**2+results2[:,3]**2)),
                             np.max(np.sqrt(results1[:,4]**2+results2[:,4]**2)),
                             np.max(np.sqrt(results1[:,5]**2+results2[:,5]**2)),
                             np.max(np.sqrt(results1[:,6]**2+results2[:,6]**2))
                             ])

    Results_displ.append(results_displ)
    Results_accel.append(results_accel)

np.save('./Results_Summary_MDOF/Results_6span_displ.npy', Results_displ)
np.save('./Results_Summary_MDOF/Results_6span_accel.npy', Results_accel)


