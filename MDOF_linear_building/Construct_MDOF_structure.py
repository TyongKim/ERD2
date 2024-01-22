###############################################################################
# This code is developed by Prof. Taeyong Kim at Ajou University              #
# taeyongkim@ajou.ac.kr                                                       #
# Jan 20, 2024                                                                #
###############################################################################

# Import libraries
import numpy as np

#%% MDOF systems sitffness and damping matrix
def Make_many_MDOF(story, number_represen, or_stiffness, or_mass):
    Total_scale = []
    for ii in range(number_represen):
        Stiff_mass = np.zeros([story, 2])
        for jj in range(int(story)): # for each story
            # stiffness
            if jj ==0:
                tmp_l = or_stiffness*1.02**(story)*0.5
                tmp_u = or_stiffness*1.02**(story)*2
                tmp_stiff = np.random.uniform(tmp_l, tmp_u, 1)
            else:
                if np.random.uniform()<0.3:
                    tmp_l = tmp_stiff*0.9
                    tmp_u = tmp_stiff*1.1
                    tmp_stiff = np.random.uniform(tmp_l, tmp_u, 1) 
                else:
                    tmp_stiff = tmp_stiff
        
            Stiff_mass[jj,0] = tmp_stiff
            
            # mass
            tmp_l = or_mass*0.8
            tmp_u = or_mass*1.2
            tmp_mass = np.random.uniform(tmp_l, tmp_u, 1)
            Stiff_mass[jj,1] = tmp_mass
            
        Total_scale.append(Stiff_mass)

    return Total_scale
    
    
#%% Modal analysis
def modal_analysis(stiff, mass):
    w,v = np.linalg.eig(np.matmul(np.linalg.inv(mass),stiff))
    idx = np.argsort(w)
    w = w[idx]
    v = v[:,idx]
    eigen_vect=[]
    eigen_val = []
    for kk in range(len(v)):
        temp = v[:,kk]/v[len(v)-1,kk]
        eigen_vect.append(temp)
        eigen_val.append(w[kk])
    eigen_vect = np.transpose(np.asarray(eigen_vect))
    eigen_val = np.transpose(np.asarray(eigen_val))
    natural_angfreq = np.sqrt(eigen_val)
    
    natural_period = 2*np.pi/natural_angfreq
    
    return [natural_period, natural_angfreq, eigen_vect]
#%% Make many MDOF systems
period = 0.2 # np.median(Ta_all[0,:]) --> 0.217
or_stiffness = (2*np.pi/period)**2
or_mass = 1

damping_range = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10])

Total_struct_system = []
for ii in range(2): # how many sets 
    tmp_struct_system = []
    for story_idx in range(2, 20): # A set of 3 to 20 story building structures
        
        # Randomly gneerate (story*500) samples of dataset
        tmp_hysteretic = Make_many_MDOF(story_idx+1, story_idx, or_stiffness, or_mass)
        
        # For each random representations
        for ii in range(len(tmp_hysteretic)):
            tmp_tmp_hysteretic = tmp_hysteretic[ii]
            
            # Run modal analysis to find period
            mass = []; stiff_info = []
            for jj in range(len(tmp_tmp_hysteretic)):
                mass.append(tmp_tmp_hysteretic[jj,1])
                stiff_info.append(tmp_tmp_hysteretic[jj,0])
            mass = np.asarray(mass); stiff_info = np.asarray(stiff_info)
            mass = np.diag(mass)
            
            stiff = np.zeros([len(stiff_info), len(stiff_info)])
            for tt in range(len(stiff_info)):
                tmp_stiff_info = stiff_info[tt]
                if tt == 0:
                    stiff[tt,tt] = stiff[tt,tt] + tmp_stiff_info
                    
                else:
                    stiff[tt,tt] = stiff[tt,tt] + tmp_stiff_info
                    stiff[tt-1,tt-1] = stiff[tt-1,tt-1] + tmp_stiff_info
                    stiff[tt-1,tt] = stiff[tt,tt-1] - tmp_stiff_info
                    stiff[tt,tt-1] = stiff[tt,tt-1] - tmp_stiff_info
            
            [natural_period, natural_angfreq, eigen_vect] = modal_analysis(stiff, mass)   
            
            # For each damping ratio (randomly 1 & 2 or 1 & 3)
            # Damping matrix (Rayleigh damping), first and second modes
            for jj in range(len(damping_range)):
                damping_coef = damping_range[jj]
                if np.random.uniform()<0.5: # 1, and 2 (50% randomness)
                    Rayeligh_mode = 1  # 2nd mode
                else:
                    Rayeligh_mode = 2  # 3rd mode
    
                a0 = (damping_coef*2*natural_angfreq[0]*natural_angfreq[Rayeligh_mode]/
                      (natural_angfreq[0]+natural_angfreq[Rayeligh_mode]))
                a1 = damping_coef*2/(natural_angfreq[0]+natural_angfreq[Rayeligh_mode])
                
                damp_matrix=a0*mass+a1*stiff
                Fancy_c = np.matmul(eigen_vect.T, np.matmul(damp_matrix, eigen_vect))
                Fancy_k = np.matmul(eigen_vect.T, np.matmul(stiff, eigen_vect))
                Fancy_m = np.matmul(eigen_vect.T, np.matmul(mass, eigen_vect))
                
                # Save the modal values
                tmp_diction = {"mass": mass,
                               "stiffness": stiff,
                               "stiff_info": stiff_info,
                               "damping_val": damping_coef,
                               "damping_mode": Rayeligh_mode
                               }
                
                tmp_struct_system.append(tmp_diction)
                
    Total_struct_system.append(tmp_struct_system)

            
# Save the MDOF systems
np.save('./Results_displ/MDOF_systems_ver2.npy', Total_struct_system)

