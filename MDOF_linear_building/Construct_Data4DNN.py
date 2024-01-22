###############################################################################
# This code is developed by Prof. Taeyong Kim at Ajou University              #
# taeyongkim@ajou.ac.kr                                                       #
# Jan 20, 2024                                                                #
###############################################################################

# Import libraries
import numpy as np
from scipy import interpolate

#%% MDOF systems
# Modal analysis function
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

    # Gamma value
    G1 = np.sum(L[0,:])/M[0,0]
    G2 = np.sum(L[1,:])/M[1,1]
    G3 = np.sum(L[2,:])/M[2,2]
        
    G = np.r_[G1,G2,G3]

    M_eff_1 = np.sum(L[0,:]*G1)
    M_eff_2 = np.sum(L[1,:]*G2)
    M_eff_3 = np.sum(L[2,:]*G3)
    
    M_eff_norm = np.r_[M_eff_1,M_eff_2,M_eff_3]/np.sum(mass)    
    
    return [natural_period, natural_angfreq, eigen_vect, G, M_eff, M_eff_norm]

#%% Data preprocess of SDOF systems
def data_postprocess(MDOF_results,
                     SDOF_results_1st,SDOF_results_2nd,SDOF_results_3rd, 
                     G, eigen_vect):

    SDOF_results_max_1st = SDOF_results_1st*G[0]*eigen_vect[:,0]
    SDOF_results_max_2nd = SDOF_results_2nd*G[1]*eigen_vect[:,1]
    SDOF_results_max_3rd = SDOF_results_3rd*G[2]*eigen_vect[:,2]  
    
    tmp_Input_Response = np.zeros([np.size(MDOF_results,0),4])
    for ii in range(np.size(MDOF_results,0)):
        tmp_Input_Response[ii,0] = MDOF_results[ii] # MDOF; m

        if ii == 0: # Inter-story drift at the first floor
            tmp_Input_Response[ii,1] = SDOF_results_max_1st[ii]
            tmp_Input_Response[ii,2] = SDOF_results_max_2nd[ii]
            tmp_Input_Response[ii,3] = SDOF_results_max_3rd[ii]
            
        elif ii == np.size(MDOF_results,0)-1: # roof top drift
            tmp_Input_Response[ii,1] = SDOF_results_max_1st[ii-1]
            tmp_Input_Response[ii,2] = SDOF_results_max_2nd[ii-1]
            tmp_Input_Response[ii,3] = SDOF_results_max_3rd[ii-1]                    
        else: # Inter-story drift 
            tmp_Input_Response[ii,1] = np.abs(SDOF_results_max_1st[ii] - SDOF_results_max_1st[ii-1])
            tmp_Input_Response[ii,2] = np.abs(SDOF_results_max_2nd[ii] - SDOF_results_max_2nd[ii-1])
            tmp_Input_Response[ii,3] = np.abs(SDOF_results_max_3rd[ii] - SDOF_results_max_3rd[ii-1])  
              
    return tmp_Input_Response  

#%% Load information
# Ground motion intensity measure, "Construct_IM.py"
Ground_info = np.load('./Ground_motions/Ground_info_artificial.npy', allow_pickle='True') 
# Response spectrum, "Construct_RS.py"
RS_SDOF = np.load('./Results_displ/A_Response_spectrum.npy')

dp = 0.015
period_candidate = np.arange(np.log(0.05)-6*dp,np.log(4)+dp,dp)
period_candi = np.exp(period_candidate)
del dp

ddm = 0.08
damping_candidate = np.arange(np.log(0.005),np.log(0.25)+ddm,ddm) # 0.5% to 20%
damping_candi = np.exp(damping_candidate)
del ddm
#%% Preprocessing the dataset
#Calculate MDOF system responses using SDOF systems

# MDOF system, "Construct_MDOF_structure.py"
MDOF_systems = np.load('./Results_displ/MDOF_systems_ver2.npy',allow_pickle=True)

DNN_info_1 = [] # GM info
DNN_info_2 = [] # Str info
DNN_info_3 = [] # SDOF and MDOF response info
for Eq_idx in range(2):
    
    # MDOF results
    tmp_directory = './Results_displ/MDOF_systems_results_%d.npy' %(Eq_idx)
    MDOF_systems_Results = np.load(tmp_directory)
    
    # (1) Ground motion intensity measure
    tmp2 = Ground_info[int(Eq_idx)]
    tmp2_GM = np.r_[tmp2['Earthquake'], np.log(tmp2['Spect_acce']),
                  np.log(tmp2['Peak_value']), np.log(tmp2['zerocross_strong']),
                    np.log(tmp2['Arias_intensity']), np.log(tmp2['Housner'])]
    
    # Response spectrum --> This is for (3)
    tmp_RS_SDOF = RS_SDOF[Eq_idx,:]
    
    # MDOF systems per each GM
    tmp_MDOF_systems = MDOF_systems[Eq_idx,:]
    
    # For each MDOF system
    results_idx = 0 # Index of MDOF systems
    for Str_idx in range(len(tmp_MDOF_systems)):
        
        # (2) Structural information        
        tmp2_MDOF_systems = tmp_MDOF_systems[Str_idx]

        mass = tmp2_MDOF_systems['mass']
        stiff = tmp2_MDOF_systems['stiffness']
        stiff_info = tmp2_MDOF_systems['stiff_info']
        damping_coef = tmp2_MDOF_systems['damping_val']
        damping_mode = tmp2_MDOF_systems['damping_mode']
        
        # Modal analysis
        [natural_period, natural_angfreq, eigen_vect,
         G, M_eff, M_eff_norm] = modal_analysis(stiff, mass, np.min([len(mass),5]))  

        # Damping matrix (Rayleigh damping), first and third modes
        a0 = (damping_coef*2*natural_angfreq[0]*natural_angfreq[damping_mode]/
              (natural_angfreq[0]+natural_angfreq[damping_mode]))
        a1 = damping_coef*2/(natural_angfreq[0]+natural_angfreq[damping_mode])

        damp=a0*mass+a1*stiff

        Fancy_m = np.matmul(eigen_vect.T, np.matmul(mass, eigen_vect))
        Fancy_k = np.matmul(eigen_vect.T, np.matmul(stiff, eigen_vect))
        Fancy_c = np.matmul(eigen_vect.T, np.matmul(damp, eigen_vect))
        
        damping = []
        for ii in range(3):
            xi_n = Fancy_c[ii,ii]/Fancy_m[ii,ii]/(2*natural_angfreq[ii])
            damping.append(xi_n)
        
        # (2) Structural information
        tmp2_Str = np.r_[natural_period[:3], M_eff_norm, damping]    
        tmp2_Str = np.log(tmp2_Str)


        # (3) SDOF: Res1, Res2, Res3 from the displacement response spectrum
        
        # What I need is RES1 RES2 RES3
        f = interpolate.interp2d(damping_candidate, period_candidate, tmp_RS_SDOF, kind='linear')

        tmp_RES = np.zeros([3,1]) # Peak dispalcement without multiplying eigen vector
        for ii in range(3):
            # damping, period
            znew = f(tmp2_Str[6+ii],tmp2_Str[ii]) 
            tmp_RES[ii,0] = znew


        # (4) MDOF results
        tmp2_MDOF_results = MDOF_systems_Results[results_idx:results_idx+len(mass)+1]
        results_idx = results_idx+len(mass)+1

        
        # Post process
        tmp_Input_Response = data_postprocess(tmp2_MDOF_results,
                             tmp_RES[0],tmp_RES[1],tmp_RES[2], 
                             G, eigen_vect)
        

        # Save the data
        DNN_info_1.append(tmp2_GM)
        DNN_info_2.append(tmp2_Str)
        DNN_info_3.append(tmp_Input_Response)

DNN_info_3 = np.array(DNN_info_3, dtype=object)
np.save('./DNN_results/DNN_input1_GM_train', DNN_info_1)
np.save('./DNN_results/DNN_input2_str_train', DNN_info_2)
np.save('./DNN_results/DNN_input3_response_train', DNN_info_3)

