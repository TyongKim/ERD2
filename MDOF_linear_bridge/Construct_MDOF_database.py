"""
This script is to perform dynamic anlaysis of multi-degree-of-freedom system.
OpenSeespy is employed when performing dynamic analysis.

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""

import numpy as np
from i_MDOF_ops import OPS_3span, OPS_4span, OPS_5span, OPS_6span

#%% Load the MDOF systems
p3_value = np.load('./generated_MDOF_systems/3_span.npy', allow_pickle=True)
p4_value = np.load('./generated_MDOF_systems/4_span.npy', allow_pickle=True)
p5_value = np.load('./generated_MDOF_systems/5_span.npy', allow_pickle=True)
p6_value = np.load('./generated_MDOF_systems/6_span.npy', allow_pickle=True)

#%% 3-span
Results_displ = []
Results_accel = []
for ii in range(3): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d' %(ii)  # longitudinal
    tmp1_2 = './Ground_motion/GM2_%d' %(ii)  # transverse
    tmp2   = './Ground_motion/time_%d' %(ii)        
     

    for jj in range(2): # for each structure

        # [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]
        tmp_value2 = p3_value[jj, :]   
        
        # OpenSees analysis
        [results, results1, results2] = OPS_3span(tmp_value2, tmp1_1, tmp1_2, tmp2)

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
for ii in range(3): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d' %(ii)  # longitudinal
    tmp1_2 = './Ground_motion/GM2_%d' %(ii)  # transverse
    tmp2   = './Ground_motion/time_%d' %(ii)        
     

    for jj in range(2): # for each structure

        # [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]
        tmp_value2 = p4_value[jj, :]   
        
        # OpenSees analysis
        [results, results1, results2] = OPS_4span(tmp_value2, tmp1_1, tmp1_2, tmp2)

        
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
for ii in range(3): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d' %(ii)  # longitudinal
    tmp1_2 = './Ground_motion/GM2_%d' %(ii)  # transverse
    tmp2   = './Ground_motion/time_%d' %(ii)        
     

    for jj in range(2): # for each structure

        # [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]
        tmp_value2 = p5_value[jj, :]   
        
        
        # OpenSees analysis
        [results, results1, results2] = OPS_5span(tmp_value2, tmp1_1, tmp1_2, tmp2)

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
for ii in range(3): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d' %(ii)  # longitudinal
    tmp1_2 = './Ground_motion/GM2_%d' %(ii)  # transverse
    tmp2   = './Ground_motion/time_%d' %(ii)        
     

    for jj in range(2): # for each structure

        # [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]
        tmp_value2 = p6_value[jj, :]   
        
        # OpenSees analysis
        [results, results1, results2] = OPS_6span(tmp_value2, tmp1_1, tmp1_2, tmp2)

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


