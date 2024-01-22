###############################################################################
# This code is developed by Prof. Taeyong Kim at Ajou University              #
# taeyongkim@ajou.ac.kr                                                       #
# Jan 20, 2024                                                                #
###############################################################################

# Import libraries
import numpy as np
import pandas as pd
import scipy
from scipy import interpolate
print(scipy.__version__)
#%% MDOF systems
def modal_analysis(stiff, mass, DOF):
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
    
    # Modal analysis
    L = np.matmul(np.transpose(eigen_vect), mass)
    M = np.matmul(np.matmul(np.transpose(eigen_vect), mass), eigen_vect)
    
    # Gamma value
    G = []; M_eff = []
    for kk in range(DOF):
        G_tmp = np.sum(L[kk,:])/M[kk,kk]
        M_eff_tmp = np.sum(L[kk,:]*G_tmp)
        G.append(G_tmp)
        M_eff.append(M_eff_tmp)
        
    G = np.asarray(G)
    M_eff = np.asarray(M_eff)
    
    return [natural_period, natural_angfreq, eigen_vect, G, M_eff]

#%% Period and damping discretization
dp = 0.015
period_candidate = np.arange(np.log(0.05)-6*dp,np.log(4)+dp,dp)
period_candi = np.exp(period_candidate)
del dp

ddm = 0.08
damping_candidate = np.arange(np.log(0.005),np.log(0.25)+ddm,ddm) # 0.5% to 20%
damping_candi = np.exp(damping_candidate)
del ddm
#%%
MDOF_systems_total = np.load('./Results_displ/MDOF_systems_ver2.npy', allow_pickle=True)

# Calculate dynamic analysis for each ground motion
for Eq_idx in range(2): # Index of ground motion

    MDOF_systems = MDOF_systems_total[Eq_idx,:] # a set of MDOF systems

    for Str_idx in range(len(MDOF_systems)): # For each structure

        # a single structural system
        tmp_str = MDOF_systems[Str_idx]

        # Load structural information
        mass = tmp_str['mass']
        stiff = tmp_str['stiffness']
        damping_coef = tmp_str['damping_val']
        damping_mode = tmp_str['damping_mode']

        # Modal anlaysis
        [natural_period, natural_angfreq, eigen_vect, G, M_eff] = modal_analysis(stiff, mass, np.min([len(mass),5]))  

        # Damping matrix (Rayleigh damping), first and third modes
        a0 = (damping_coef*2*natural_angfreq[0]*natural_angfreq[damping_mode]/
              (natural_angfreq[0]+natural_angfreq[damping_mode]))
        a1 = damping_coef*2/(natural_angfreq[0]+natural_angfreq[damping_mode])
        
        damp=a0*mass+a1*stiff
        
        Fancy_m = np.matmul(eigen_vect.T, np.matmul(mass, eigen_vect))
        Fancy_c = np.matmul(eigen_vect.T, np.matmul(damp, eigen_vect))
        
        # Summation of each mode up to 5 (approximately calculate structural responses)
        Interpol_disp = []
        Interpol_time = []
        for ii in range(np.min([len(mass),5])):
            omega_n = natural_angfreq[ii] #np.sqrt(Fancy_k[ii,ii]/Fancy_m[ii,ii])
            period_n = natural_period[ii] #2*np.pi/omega_n
            xi_n = Fancy_c[ii,ii]/Fancy_m[ii,ii]/(2*omega_n) # 2xi sqrt(k/m)        
            
            # Target structure
            target_damping1 = int(np.where(np.min(np.abs(damping_candi-xi_n))==np.abs(damping_candi-xi_n))[0])
            if xi_n - damping_candi[target_damping1] < 0:
                target_damping2 = int(target_damping1-1)
            else:
                target_damping2 = int(target_damping1+1)
                
            target_period1 = int(np.where(np.min(np.abs(np.log(period_candi)-np.log(period_n)))
                                          ==np.abs(np.log(period_candi)-np.log(period_n)))[0])
            
            # Load data (this are the results from "Construct_SDOF_database.py")
            tmp_directory1 = './Results_displ/GM_%d/period_%d.p' %(Eq_idx, target_period1)
    
            df1 = pd.read_pickle(tmp_directory1)
            
            displ1_1 = df1.iloc[:,target_damping1+1].values
            displ1_2 = df1.iloc[:,target_damping2+1].values
           
            time_step = df1.iloc[:,0].values

            # Interpolate
            y = np.array([damping_candi[target_damping1], damping_candi[target_damping2]]) # damping
            displ_interp = []
            for ii in range(len(time_step)):
                z = np.zeros([2])
                z[0] = displ1_1[ii]
                z[1] = displ1_2[ii]

                f = interpolate.interp1d(y, z)
            
                znew = f(xi_n)
                displ_interp.append(znew)
                
            displ_interp = np.asarray(displ_interp)
            Interpol_disp.append(displ_interp)
            Interpol_time.append(time_step)
        
        # Summation for the MDOF systme's response
        s = np.ones([len(Fancy_m),1])
        for ii in range(len(Interpol_disp)):
            f = interpolate.interp1d(np.r_[0,Interpol_time[ii],Interpol_time[ii][-1]+0.1], 
                                     np.r_[0,Interpol_disp[ii].reshape(len(Interpol_disp[ii],)),
                                           Interpol_disp[ii][-1]])

            gamma_n = np.matmul(eigen_vect[:,ii].T,np.matmul(mass,s))/Fancy_m[ii,ii]

            q_t = gamma_n*f(Interpol_time[-1]) # Displacement; 
            u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                            q_t.reshape(1,len(q_t)))

            if ii == 0:
                displ_MDOF = np.zeros_like(u_t)  
                
            displ_MDOF = displ_MDOF + u_t

                    
        # Post process (Find out max drift ratio, and max IDR)
        tmp_IDR = displ_MDOF[1:,:] - displ_MDOF[:-1,:]
        tmp_IDR2 = np.r_[np.max(np.abs(displ_MDOF[0,:])) ,
                         np.max(np.abs(tmp_IDR),axis=1) ,
                         np.max(np.abs(displ_MDOF[-1,:])) ] # meter
        
    
        #MDOF_systems_Results.append(tmp_IDR2)
        if Str_idx == 0:
            MDOF_systems_Results = tmp_IDR2
        else:
            MDOF_systems_Results = np.r_[MDOF_systems_Results, tmp_IDR2]        
            
    # Save the results
    tmp_directory = './Results_displ/MDOF_systems_results_%d.npy' %(Eq_idx)
    np.save(tmp_directory, MDOF_systems_Results)


