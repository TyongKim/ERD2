"""
This script is to perform dynamic anlaysis for each mode and save peak 
displacement and acceleration values

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""


import numpy as np
from i_MDOF_matrix_all import MDOF_3span_analysis, MDOF_4span_analysis, MDOF_5span_analysis, MDOF_6span_analysis
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
Results_max_displ = []; Results_max_accel = []
Results_gm_direction = []
for ii in range(100): # Number of ground motions

    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'      
     
    
    gm_length = len(np.loadtxt(tmp1_1))
    gm_time_ratio = np.max([1, np.loadtxt(tmp2)/0.001])
    gm_timeseries = np.linspace(0,np.loadtxt(tmp2)*gm_length*1.3, int(gm_length*gm_time_ratio*1.3))
    
    # [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]
    
    # SDOF analysis
    results_displ, results_accel = MDOF_3span_analysis(p3_value, tmp1_1, tmp1_2, tmp2)        
    
    for kk in range(2): # For each response
        if kk == 0:
            results = np.asarray(results_displ)
        
        else:
            results = np.asarray(results_accel)
        
        
        # Save responses for each mode, only the center node of each span is saved   
        if np.max(np.abs(results[0,0,:])) == 0: # first mode transverse
            # 1st mode: out ouf plane
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,2,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:]))], # Node 2
                                   
                                   [np.max(np.abs(results[0,6,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:]))], # Node 6

                                   [np.max(np.abs(results[0,10,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:]))], # Node 10
                                    

                                    ])
            Results_gm_direction.append(1) # out of plane
            
        else: # first mode longitudinal
            # 1st mode: longitudinal
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,2,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:]))], # Node 2

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,6,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:]))], # Node 6

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,10,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:]))], # Node 10

                                    
                                    ])       
            Results_gm_direction.append(0) # longitudinal
        
        if kk ==0:
            Results_max_displ.append(results_max_abs)
        else:
            Results_max_accel.append(results_max_abs)
 
    print(ii)
    
np.save('./Results_Summary_SDOF/Results_max_displ_3span.npy', Results_max_displ)
np.save('./Results_Summary_SDOF/Results_max_accel_3span.npy', Results_max_accel)
np.save('./Results_Summary_SDOF/Results_gm_direct_3span.npy', Results_gm_direction)


#%% 4-span
Results_max_displ = []; Results_max_accel = []
Results_gm_direction = []
for ii in range(100): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'            
                
    
    gm_length = len(np.loadtxt(tmp1_1))
    gm_time_ratio = np.max([1, np.loadtxt(tmp2)/0.001])
    gm_timeseries = np.linspace(0,np.loadtxt(tmp2)*gm_length*1.3, int(gm_length*gm_time_ratio*1.3))
    

    # SDOF analysis
    results_displ, results_accel = MDOF_4span_analysis(p4_value, tmp1_1, tmp1_2, tmp2)        
    
    for kk in range(2): # For each response
        if kk == 0:
            results = np.asarray(results_displ)
        
        else:
            results = np.asarray(results_accel)
        
    
        # For each node, max abs for the top three modes        
        if np.max(np.abs(results[0,0,:])) == 0:
            # 1st mode out ouf plane
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,2,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:])),
                                    np.max(np.abs(results[12,2,:])), np.max(np.abs(results[13,2,:])), np.max(np.abs(results[14,2,:])),
                                    np.max(np.abs(results[15,2,:]))
                                    ], # Node 2
                                   
                                   [np.max(np.abs(results[0,6,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:])),
                                    np.max(np.abs(results[12,6,:])), np.max(np.abs(results[13,6,:])), np.max(np.abs(results[14,6,:])),
                                    np.max(np.abs(results[15,6,:]))                                
                                   ], # Node 6

                                   [np.max(np.abs(results[0,10,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:])),
                                    np.max(np.abs(results[12,10,:])), np.max(np.abs(results[13,10,:])), np.max(np.abs(results[14,10,:])),
                                    np.max(np.abs(results[15,10,:]))                                    
                                    ], # Node 10

                                   [np.max(np.abs(results[0,14,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,14,:])),
                                    np.max(np.abs(results[3,14,:])), np.max(np.abs(results[4,14,:])), np.max(np.abs(results[5,14,:])),
                                    np.max(np.abs(results[6,14,:])), np.max(np.abs(results[7,14,:])), np.max(np.abs(results[8,14,:])),
                                    np.max(np.abs(results[9,14,:])), np.max(np.abs(results[10,14,:])), np.max(np.abs(results[11,14,:])),
                                    np.max(np.abs(results[12,14,:])), np.max(np.abs(results[13,14,:])), np.max(np.abs(results[14,14,:])),
                                    np.max(np.abs(results[15,14,:]))                                  
                                    ],  # Node 14

                                    ])
            Results_gm_direction.append(1) # out of plane
            
        else:
            # longitudinal
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,2,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:])),
                                    np.max(np.abs(results[12,2,:])), np.max(np.abs(results[13,2,:])), np.max(np.abs(results[14,2,:])),
                                    np.max(np.abs(results[15,2,:]))                                   
                                    
                                    ], # Node 2

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,6,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:])),
                                    np.max(np.abs(results[12,6,:])), np.max(np.abs(results[13,6,:])), np.max(np.abs(results[14,6,:])),
                                    np.max(np.abs(results[15,6,:]))                                   
                                    
                                    ], # Node 6

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,10,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:])),
                                    np.max(np.abs(results[12,10,:])), np.max(np.abs(results[13,10,:])), np.max(np.abs(results[14,10,:])),
                                    np.max(np.abs(results[15,10,:]))                                     
                                    ], # Node 10
                                    
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,14,:])), np.max(np.abs(results[2,14,:])),
                                    np.max(np.abs(results[3,14,:])), np.max(np.abs(results[4,14,:])), np.max(np.abs(results[5,14,:])),
                                    np.max(np.abs(results[6,14,:])), np.max(np.abs(results[7,14,:])), np.max(np.abs(results[8,14,:])),
                                    np.max(np.abs(results[9,14,:])), np.max(np.abs(results[10,14,:])), np.max(np.abs(results[11,14,:])),
                                    np.max(np.abs(results[12,14,:])), np.max(np.abs(results[13,14,:])), np.max(np.abs(results[14,14,:])),
                                    np.max(np.abs(results[15,14,:]))                                      
                                    
                                    ],  # Node 14
                                   
                                    ])  
            Results_gm_direction.append(0) # longitudinal

        if kk ==0:
            Results_max_displ.append(results_max_abs)
        else:
            Results_max_accel.append(results_max_abs)

    print(ii)
    
np.save('./Results_Summary_SDOF/Results_max_displ_4span.npy', Results_max_displ)
np.save('./Results_Summary_SDOF/Results_max_accel_4span.npy', Results_max_accel)
np.save('./Results_Summary_SDOF/Results_gm_direct_4span.npy', Results_gm_direction)

#%% 5-span
Results_max_displ = []; Results_max_accel = []
Results_gm_direction = []
for ii in range(100): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'            
           
    
    gm_length = len(np.loadtxt(tmp1_1))
    gm_time_ratio = np.max([1, np.loadtxt(tmp2)/0.001])
    gm_timeseries = np.linspace(0,np.loadtxt(tmp2)*gm_length*1.3, int(gm_length*gm_time_ratio*1.3))

    # SDOF analysis
    results_displ, results_accel = MDOF_5span_analysis(p5_value, tmp1_1, tmp1_2, tmp2)        
    
    for kk in range(2): # For each response
        if kk == 0:
            results = np.asarray(results_displ)
        
        else:
            results = np.asarray(results_accel)
    
        # For each node, max abs for the top three modes        
        if np.max(np.abs(results[0,0,:])) == 0:
            # 1st mode out ouf plane
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,2,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:])),
                                    np.max(np.abs(results[12,2,:])), np.max(np.abs(results[13,2,:])), np.max(np.abs(results[14,2,:])),
                                    np.max(np.abs(results[15,2,:])), np.max(np.abs(results[16,2,:])), np.max(np.abs(results[17,2,:])),
                                    np.max(np.abs(results[18,2,:])), np.max(np.abs(results[19,2,:]))
                                    ], # Node 2
                                   
                                   [np.max(np.abs(results[0,6,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:])),
                                    np.max(np.abs(results[12,6,:])), np.max(np.abs(results[13,6,:])), np.max(np.abs(results[14,6,:])),
                                    np.max(np.abs(results[15,6,:])), np.max(np.abs(results[16,6,:])), np.max(np.abs(results[17,6,:])),
                                    np.max(np.abs(results[18,6,:])), np.max(np.abs(results[19,6,:]))                                    
                                    ], # Node 6

                                   [np.max(np.abs(results[0,10,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:])),
                                    np.max(np.abs(results[12,10,:])), np.max(np.abs(results[13,10,:])), np.max(np.abs(results[14,10,:])),
                                    np.max(np.abs(results[15,10,:])), np.max(np.abs(results[16,10,:])), np.max(np.abs(results[17,10,:])),
                                    np.max(np.abs(results[18,10,:])), np.max(np.abs(results[19,10,:]))                                        
                                    ], # Node 10

                                   [np.max(np.abs(results[0,14,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,14,:])),
                                    np.max(np.abs(results[3,14,:])), np.max(np.abs(results[4,14,:])), np.max(np.abs(results[5,14,:])),
                                    np.max(np.abs(results[6,14,:])), np.max(np.abs(results[7,14,:])), np.max(np.abs(results[8,14,:])),
                                    np.max(np.abs(results[9,14,:])), np.max(np.abs(results[10,14,:])), np.max(np.abs(results[11,14,:])),
                                    np.max(np.abs(results[12,14,:])), np.max(np.abs(results[13,14,:])), np.max(np.abs(results[14,14,:])),
                                    np.max(np.abs(results[15,14,:])), np.max(np.abs(results[16,14,:])), np.max(np.abs(results[17,14,:])),
                                    np.max(np.abs(results[18,14,:])), np.max(np.abs(results[19,14,:]))                                                                           
                                    ],  # Node 14
                                   
                                   [np.max(np.abs(results[0,18,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,18,:])),
                                    np.max(np.abs(results[3,18,:])), np.max(np.abs(results[4,18,:])), np.max(np.abs(results[5,18,:])),
                                    np.max(np.abs(results[6,18,:])), np.max(np.abs(results[7,18,:])), np.max(np.abs(results[8,18,:])),
                                    np.max(np.abs(results[9,18,:])), np.max(np.abs(results[10,18,:])), np.max(np.abs(results[11,18,:])),
                                    np.max(np.abs(results[12,18,:])), np.max(np.abs(results[13,18,:])), np.max(np.abs(results[14,18,:])),
                                    np.max(np.abs(results[15,18,:])), np.max(np.abs(results[16,18,:])), np.max(np.abs(results[17,18,:])),
                                    np.max(np.abs(results[18,18,:])), np.max(np.abs(results[19,18,:]))                                    
                                    ],  # Node 18                               
                                   

                                    ])
            Results_gm_direction.append(1) # out of plane
            
        else:
            # longitudinal
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,2,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:])),
                                    np.max(np.abs(results[12,2,:])), np.max(np.abs(results[13,2,:])), np.max(np.abs(results[14,2,:])),
                                    np.max(np.abs(results[15,2,:])), np.max(np.abs(results[16,2,:])), np.max(np.abs(results[17,2,:])),
                                    np.max(np.abs(results[18,2,:])), np.max(np.abs(results[19,2,:]))                                    
                                    ], # Node 2

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,6,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:])),
                                    np.max(np.abs(results[12,6,:])), np.max(np.abs(results[13,6,:])), np.max(np.abs(results[14,6,:])),
                                    np.max(np.abs(results[15,6,:])), np.max(np.abs(results[16,6,:])), np.max(np.abs(results[17,6,:])),
                                    np.max(np.abs(results[18,6,:])), np.max(np.abs(results[19,6,:]))                                        
                                    ], # Node 6

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,10,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:])),
                                    np.max(np.abs(results[12,10,:])), np.max(np.abs(results[13,10,:])), np.max(np.abs(results[14,10,:])),
                                    np.max(np.abs(results[15,10,:])), np.max(np.abs(results[16,10,:])), np.max(np.abs(results[17,10,:])),
                                    np.max(np.abs(results[18,10,:])), np.max(np.abs(results[19,10,:]))                                       
                                    ], # Node 10
                                    
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,14,:])), np.max(np.abs(results[2,14,:])),
                                    np.max(np.abs(results[3,14,:])), np.max(np.abs(results[4,14,:])), np.max(np.abs(results[5,14,:])),
                                    np.max(np.abs(results[6,14,:])), np.max(np.abs(results[7,14,:])), np.max(np.abs(results[8,14,:])),
                                    np.max(np.abs(results[9,14,:])), np.max(np.abs(results[10,14,:])), np.max(np.abs(results[11,14,:])),
                                    np.max(np.abs(results[12,14,:])), np.max(np.abs(results[13,14,:])), np.max(np.abs(results[14,14,:])),
                                    np.max(np.abs(results[15,14,:])), np.max(np.abs(results[16,14,:])), np.max(np.abs(results[17,14,:])),
                                    np.max(np.abs(results[18,14,:])), np.max(np.abs(results[19,14,:]))                                      
                                    ],  # Node 14
                                   
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,18,:])), np.max(np.abs(results[2,18,:])),
                                    np.max(np.abs(results[3,18,:])), np.max(np.abs(results[4,18,:])), np.max(np.abs(results[5,18,:])),
                                    np.max(np.abs(results[6,18,:])), np.max(np.abs(results[7,18,:])), np.max(np.abs(results[8,18,:])),
                                    np.max(np.abs(results[9,18,:])), np.max(np.abs(results[10,18,:])), np.max(np.abs(results[11,18,:])),
                                    np.max(np.abs(results[12,18,:])), np.max(np.abs(results[13,18,:])), np.max(np.abs(results[14,18,:])),
                                    np.max(np.abs(results[15,18,:])), np.max(np.abs(results[16,18,:])), np.max(np.abs(results[17,18,:])),
                                    np.max(np.abs(results[18,18,:])), np.max(np.abs(results[19,18,:]))                                         
                                    ],  # Node 18                                         

                                    ])  
            Results_gm_direction.append(0) # longitudinal

        if kk ==0:
            Results_max_displ.append(results_max_abs)
        else:
            Results_max_accel.append(results_max_abs)

    print(ii)
    
np.save('./Results_Summary_SDOF/Results_max_displ_5span.npy', Results_max_displ)
np.save('./Results_Summary_SDOF/Results_max_accel_5span.npy', Results_max_accel)
np.save('./Results_Summary_SDOF/Results_gm_direct_5span.npy', Results_gm_direction)

#%% 6-span
Results_max_displ = []; Results_max_accel = []
Results_gm_direction = []
for ii in range(100): # ground motions
    
    # GM name
    tmp1_1 = './Ground_motion/GM1_%d.txt' %(ii)
    tmp1_2 = './Ground_motion/GM2_%d.txt' %(ii)
    tmp2   = './Ground_motion/time_0'            
           
    gm_length = len(np.loadtxt(tmp1_1))
    gm_time_ratio = np.max([1, np.loadtxt(tmp2)/0.001])
    gm_timeseries = np.linspace(0,np.loadtxt(tmp2)*gm_length*1.3, int(gm_length*gm_time_ratio*1.3))
    

    # SDOF analysis
    results_displ, results_accel = MDOF_6span_analysis(p6_value, tmp1_1, tmp1_2, tmp2)        
    
    for kk in range(2): # For each response
        if kk == 0:
            results = np.asarray(results_displ)
        
        else:
            results = np.asarray(results_accel)
        
        # For each node, max abs for the top three modes        
        if np.max(np.abs(results[0,0,:])) == 0:
            # 1st mode out ouf plane
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,2,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:])),
                                    np.max(np.abs(results[12,2,:])), np.max(np.abs(results[13,2,:])), np.max(np.abs(results[14,2,:])),
                                    np.max(np.abs(results[15,2,:])), np.max(np.abs(results[16,2,:])), np.max(np.abs(results[17,2,:])),
                                    np.max(np.abs(results[18,2,:])), np.max(np.abs(results[19,2,:])), np.max(np.abs(results[20,2,:])),
                                    np.max(np.abs(results[21,2,:])), np.max(np.abs(results[22,2,:])), np.max(np.abs(results[23,2,:]))                                    
                                    
                                    ], # Node 2
                                   
                                   [np.max(np.abs(results[0,6,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:])),
                                    np.max(np.abs(results[12,6,:])), np.max(np.abs(results[13,6,:])), np.max(np.abs(results[14,6,:])),
                                    np.max(np.abs(results[15,6,:])), np.max(np.abs(results[16,6,:])), np.max(np.abs(results[17,6,:])),
                                    np.max(np.abs(results[18,6,:])), np.max(np.abs(results[19,6,:])), np.max(np.abs(results[20,6,:])),
                                    np.max(np.abs(results[21,6,:])), np.max(np.abs(results[22,6,:])), np.max(np.abs(results[23,6,:]))                                        
                                    ], # Node 6

                                   [np.max(np.abs(results[0,10,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:])),
                                    np.max(np.abs(results[12,10,:])), np.max(np.abs(results[13,10,:])), np.max(np.abs(results[14,10,:])),
                                    np.max(np.abs(results[15,10,:])), np.max(np.abs(results[16,10,:])), np.max(np.abs(results[17,10,:])),
                                    np.max(np.abs(results[18,10,:])), np.max(np.abs(results[19,10,:])), np.max(np.abs(results[20,10,:])),
                                    np.max(np.abs(results[21,10,:])), np.max(np.abs(results[22,10,:])), np.max(np.abs(results[23,10,:]))    
                                    ], # Node 10

                                   [np.max(np.abs(results[0,14,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,14,:])),
                                    np.max(np.abs(results[3,14,:])), np.max(np.abs(results[4,14,:])), np.max(np.abs(results[5,14,:])),
                                    np.max(np.abs(results[6,14,:])), np.max(np.abs(results[7,14,:])), np.max(np.abs(results[8,14,:])),
                                    np.max(np.abs(results[9,14,:])), np.max(np.abs(results[10,14,:])), np.max(np.abs(results[11,14,:])),
                                    np.max(np.abs(results[12,14,:])), np.max(np.abs(results[13,14,:])), np.max(np.abs(results[14,14,:])),
                                    np.max(np.abs(results[15,14,:])), np.max(np.abs(results[16,14,:])), np.max(np.abs(results[17,14,:])),
                                    np.max(np.abs(results[18,14,:])), np.max(np.abs(results[19,14,:])), np.max(np.abs(results[20,14,:])),
                                    np.max(np.abs(results[21,14,:])), np.max(np.abs(results[22,14,:])), np.max(np.abs(results[23,14,:]))                                     
                                    ],  # Node 14
                                   
                                   [np.max(np.abs(results[0,18,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,18,:])),
                                    np.max(np.abs(results[3,18,:])), np.max(np.abs(results[4,18,:])), np.max(np.abs(results[5,18,:])),
                                    np.max(np.abs(results[6,18,:])), np.max(np.abs(results[7,18,:])), np.max(np.abs(results[8,18,:])),
                                    np.max(np.abs(results[9,18,:])), np.max(np.abs(results[10,18,:])), np.max(np.abs(results[11,18,:])),
                                    np.max(np.abs(results[12,18,:])), np.max(np.abs(results[13,18,:])), np.max(np.abs(results[14,18,:])),
                                    np.max(np.abs(results[15,18,:])), np.max(np.abs(results[16,18,:])), np.max(np.abs(results[17,18,:])),
                                    np.max(np.abs(results[18,18,:])), np.max(np.abs(results[19,18,:])), np.max(np.abs(results[20,18,:])),
                                    np.max(np.abs(results[21,18,:])), np.max(np.abs(results[22,18,:])), np.max(np.abs(results[23,18,:]))                                     
                                    ],  # Node 18                               

                                   [np.max(np.abs(results[0,22,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,22,:])),
                                    np.max(np.abs(results[3,22,:])), np.max(np.abs(results[4,22,:])), np.max(np.abs(results[5,22,:])),
                                    np.max(np.abs(results[6,22,:])), np.max(np.abs(results[7,22,:])), np.max(np.abs(results[8,22,:])),
                                    np.max(np.abs(results[9,22,:])), np.max(np.abs(results[10,22,:])), np.max(np.abs(results[11,22,:])),
                                    np.max(np.abs(results[12,22,:])), np.max(np.abs(results[13,22,:])), np.max(np.abs(results[14,22,:])),
                                    np.max(np.abs(results[15,22,:])), np.max(np.abs(results[16,22,:])), np.max(np.abs(results[17,22,:])),
                                    np.max(np.abs(results[18,22,:])), np.max(np.abs(results[19,22,:])), np.max(np.abs(results[20,22,:])),
                                    np.max(np.abs(results[21,22,:])), np.max(np.abs(results[22,22,:])), np.max(np.abs(results[23,22,:]))                                        
                                    ],  # Node 22
                                   

                                    ])
            Results_gm_direction.append(1) # out of plane
            
        else:
            # longitudinal
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,2,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:])),
                                    np.max(np.abs(results[12,2,:])), np.max(np.abs(results[13,2,:])), np.max(np.abs(results[14,2,:])),
                                    np.max(np.abs(results[15,2,:])), np.max(np.abs(results[16,2,:])), np.max(np.abs(results[17,2,:])),
                                    np.max(np.abs(results[18,2,:])), np.max(np.abs(results[19,2,:])), np.max(np.abs(results[20,2,:])),
                                    np.max(np.abs(results[21,2,:])), np.max(np.abs(results[22,2,:])), np.max(np.abs(results[23,2,:]))                                         
                                    ], # Node 2

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,6,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:])),
                                    np.max(np.abs(results[12,6,:])), np.max(np.abs(results[13,6,:])), np.max(np.abs(results[14,6,:])),
                                    np.max(np.abs(results[15,6,:])), np.max(np.abs(results[16,6,:])), np.max(np.abs(results[17,6,:])),
                                    np.max(np.abs(results[18,6,:])), np.max(np.abs(results[19,6,:])), np.max(np.abs(results[20,6,:])),
                                    np.max(np.abs(results[21,6,:])), np.max(np.abs(results[22,6,:])), np.max(np.abs(results[23,6,:]))                                      
                                    ], # Node 6

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,10,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:])),
                                    np.max(np.abs(results[12,10,:])), np.max(np.abs(results[13,10,:])), np.max(np.abs(results[14,10,:])),
                                    np.max(np.abs(results[15,10,:])), np.max(np.abs(results[16,10,:])), np.max(np.abs(results[17,10,:])),
                                    np.max(np.abs(results[18,10,:])), np.max(np.abs(results[19,10,:])), np.max(np.abs(results[20,10,:])),
                                    np.max(np.abs(results[21,10,:])), np.max(np.abs(results[22,10,:])), np.max(np.abs(results[23,10,:]))                                      
                                    ], # Node 10
                                    
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,14,:])), np.max(np.abs(results[2,14,:])),
                                    np.max(np.abs(results[3,14,:])), np.max(np.abs(results[4,14,:])), np.max(np.abs(results[5,14,:])),
                                    np.max(np.abs(results[6,14,:])), np.max(np.abs(results[7,14,:])), np.max(np.abs(results[8,14,:])),
                                    np.max(np.abs(results[9,14,:])), np.max(np.abs(results[10,14,:])), np.max(np.abs(results[11,14,:])),
                                    np.max(np.abs(results[12,14,:])), np.max(np.abs(results[13,14,:])), np.max(np.abs(results[14,14,:])),
                                    np.max(np.abs(results[15,14,:])), np.max(np.abs(results[16,14,:])), np.max(np.abs(results[17,14,:])),
                                    np.max(np.abs(results[18,14,:])), np.max(np.abs(results[19,14,:])), np.max(np.abs(results[20,14,:])),
                                    np.max(np.abs(results[21,14,:])), np.max(np.abs(results[22,14,:])), np.max(np.abs(results[23,14,:]))                                         
                                    ],  # Node 14
                                   
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,18,:])), np.max(np.abs(results[2,18,:])),
                                    np.max(np.abs(results[3,18,:])), np.max(np.abs(results[4,18,:])), np.max(np.abs(results[5,18,:])),
                                    np.max(np.abs(results[6,18,:])), np.max(np.abs(results[7,18,:])), np.max(np.abs(results[8,18,:])),
                                    np.max(np.abs(results[9,18,:])), np.max(np.abs(results[10,18,:])), np.max(np.abs(results[11,18,:])),
                                    np.max(np.abs(results[12,18,:])), np.max(np.abs(results[13,18,:])), np.max(np.abs(results[14,18,:])),
                                    np.max(np.abs(results[15,18,:])), np.max(np.abs(results[16,18,:])), np.max(np.abs(results[17,18,:])),
                                    np.max(np.abs(results[18,18,:])), np.max(np.abs(results[19,18,:])), np.max(np.abs(results[20,18,:])),
                                    np.max(np.abs(results[21,18,:])), np.max(np.abs(results[22,18,:])), np.max(np.abs(results[23,18,:]))                                       
                                    ],  # Node 18                                         

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,22,:])), np.max(np.abs(results[2,22,:])),
                                    np.max(np.abs(results[3,22,:])), np.max(np.abs(results[4,22,:])), np.max(np.abs(results[5,22,:])),
                                    np.max(np.abs(results[6,22,:])), np.max(np.abs(results[7,22,:])), np.max(np.abs(results[8,22,:])),
                                    np.max(np.abs(results[9,22,:])), np.max(np.abs(results[10,22,:])), np.max(np.abs(results[11,22,:])),
                                    np.max(np.abs(results[12,22,:])), np.max(np.abs(results[13,22,:])), np.max(np.abs(results[14,22,:])),
                                    np.max(np.abs(results[15,22,:])), np.max(np.abs(results[16,22,:])), np.max(np.abs(results[17,22,:])),
                                    np.max(np.abs(results[18,22,:])), np.max(np.abs(results[19,22,:])), np.max(np.abs(results[20,22,:])),
                                    np.max(np.abs(results[21,22,:])), np.max(np.abs(results[22,22,:])), np.max(np.abs(results[23,22,:]))                                      
                                    ],  # Node 22                                    
                                    

                                    ])  
            Results_gm_direction.append(0) # longitudinal
            
        if kk ==0:
            Results_max_displ.append(results_max_abs)
        else:
            Results_max_accel.append(results_max_abs)

    print(ii)
    
np.save('./Results_Summary_SDOF/Results_max_displ_6span.npy', Results_max_displ)
np.save('./Results_Summary_SDOF/Results_max_accel_6span.npy', Results_max_accel)
np.save('./Results_Summary_SDOF/Results_gm_direct_6span.npy', Results_gm_direction)




