"""
This script is to perform modal analysis of bridge structural systems.

Written by Taeyong Kim at Ajou University.
taeyongkim@ajou.ac.kr
"""

def modal_analysis(stiff, mass):
    import numpy as np
    
    w,v = np.linalg.eig(np.matmul(np.linalg.inv(mass),stiff))
    idx = np.argsort(w)
    w = w[idx]
    v = v[:,idx]
    eigen_vect=[]
    eigen_val = []
    for kk in range(len(v)):
        temp = v[:,kk]
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
    G1 = np.sum(L[0,:])/M[0,0]
    G2 = np.sum(L[1,:])/M[1,1]
    G3 = np.sum(L[2,:])/M[2,2]
    G4 = np.sum(L[3,:])/M[3,3]
    G5 = np.sum(L[4,:])/M[4,4]
        
    G = np.r_[G1,G2,G3]

    M_eff_1 = np.sum(L[0,:]*G1)
    M_eff_2 = np.sum(L[1,:]*G2)
    M_eff_3 = np.sum(L[2,:]*G3)
    M_eff_4 = np.sum(L[3,:]*G4)
    M_eff_5 = np.sum(L[4,:]*G5)
    
    M_eff = np.r_[M_eff_1,M_eff_2,M_eff_3]
    
    return [natural_period, natural_angfreq, eigen_vect, G, M_eff]


#%%

def MDOF_3span_analysis(tmp_value, tmp1_1, tmp1_2, tmp2):   
    import numpy as np
    from i_SDOF import A_SDOF_span

    mass_column, sitff_y_super, stiff_x_column, stiff_y_column, damping, random_values = tmp_value
    
    # Stiffness y of super structure 
    sitff_y_super1 = sitff_y_super*random_values[0]
    sitff_y_super2 = sitff_y_super*random_values[1]
    sitff_y_super3 = sitff_y_super*random_values[2]

    # Stiffness y of column
    stiff_y_column1 = stiff_y_column*random_values[3]
    stiff_y_column2 = stiff_y_column*random_values[4]

    # Modal results
    mass_total = np.ones([12,])
    mass_total[0] = 13 + 2*mass_column
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    
    mass = np.diag(mass_total)
    
    stiff = np.array([[2*stiff_x_column, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0 ],
                      [0, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                      [0, -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0 , 0, 0, 0, 0, 0, 0, 0 ],
                      [0, 0, -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -sitff_y_super1, sitff_y_super1+sitff_y_super2+stiff_y_column1, -sitff_y_super2, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -sitff_y_super2, stiff_y_column2+sitff_y_super2+sitff_y_super3, -sitff_y_super3, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3],
                              ])


    [natural_period, natural_angfreq, eigen_vect,  G, M_eff] = modal_analysis(stiff, mass) 

    Fancy_m = np.matmul(eigen_vect.T, np.matmul(mass, eigen_vect))

    
    a0 = (damping*2*natural_angfreq[0]*natural_angfreq[1]/
      (natural_angfreq[0]+natural_angfreq[1]))
    a1 = damping*2/(natural_angfreq[0]+natural_angfreq[1])
    
    xi = 1/2*(a0/natural_angfreq + a1*natural_angfreq)
    
    results_displ = []; results_accel = []
    s = np.ones([12,1])
    for ii in range(len(s)):
        # Each mode information
        period_n = 2*np.pi/natural_angfreq[ii]
        stiff_n = (2*np.pi/period_n)**2 # (2*np.pi/period_n)**2
        
        #xi_n = 2*damping_coef*np.sqrt(stiff/1) # 2xi sqrt(k/m)
        damping_n = 2*xi[ii]*np.sqrt(stiff_n/1)  # 2xi sqrt(k/m)
    
        # Run SDOF analysis (depends on the modal resopnse)
        if eigen_vect[0,ii] == 1:
            A_SDOF_span(stiff_n, damping_n, tmp1_1, tmp2)
        else:
            A_SDOF_span(stiff_n, damping_n, tmp1_2, tmp2)
            
        displ_SDOF = np.loadtxt('./Results_SDOF/NodeDisp.out')
        accel_SDOF = np.loadtxt('./Results_SDOF/NodeAccel.out')
        
        # Modal values
        gamma_n = np.matmul(eigen_vect[:,ii].T,np.matmul(mass,s))/Fancy_m[ii,ii]
        
        # Summation
        q_t = gamma_n*displ_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_displ.append(u_t)

        # Summation
        q_t = gamma_n*accel_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_accel.append(u_t)        
            
    return results_displ, results_accel 


def MDOF_4span_analysis(tmp_value, tmp1_1, tmp1_2, tmp2):
    import numpy as np
    from i_SDOF import A_SDOF_span

    mass_column, sitff_y_super, stiff_x_column, stiff_y_column, damping, random_values = tmp_value

    # Stiffness y of super structure 
    sitff_y_super1 = sitff_y_super*random_values[0]
    sitff_y_super2 = sitff_y_super*random_values[1]
    sitff_y_super3 = sitff_y_super*random_values[2]
    sitff_y_super4 = sitff_y_super*random_values[3]

    # Stiffness y of column
    stiff_y_column1 = stiff_y_column*random_values[4]
    stiff_y_column2 = stiff_y_column*random_values[5]
    stiff_y_column3 = stiff_y_column*random_values[6]

    # Modal results
    mass_total = np.ones([16,])
    mass_total[0] = 17 + 3*mass_column
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    mass_total[12] = mass_total[12] + mass_column
    
    mass = np.diag(mass_total)
    

    stiff = np.array([[3*stiff_x_column, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ],
                      [0, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0 ],
                      [0, -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0 ],
                      [0, 0, -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
                      [0, 0, 0, -sitff_y_super1, sitff_y_super1+sitff_y_super2+stiff_y_column1, -sitff_y_super2, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
                      [0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0, 0, 0,0,0],
                      [0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0, 0,0,0],
                      [0, 0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, -sitff_y_super2, stiff_y_column2+sitff_y_super2+sitff_y_super3, -sitff_y_super3, 0, 0, 0, 0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0, 0, 0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0, 0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3,  stiff_y_column3+sitff_y_super3+sitff_y_super4,-sitff_y_super4, 0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4],

                      ])
        
    [natural_period, natural_angfreq, eigen_vect,  G, M_eff] = modal_analysis(stiff, mass) 

    Fancy_m = np.matmul(eigen_vect.T, np.matmul(mass, eigen_vect))
    
    a0 = (damping*2*natural_angfreq[0]*natural_angfreq[1]/
      (natural_angfreq[0]+natural_angfreq[1]))
    a1 = damping*2/(natural_angfreq[0]+natural_angfreq[1])
    
    xi = 1/2*(a0/natural_angfreq + a1*natural_angfreq)
    
    results_displ = []; results_accel = []
    s = np.ones([len(stiff),1])
    for ii in range(len(s)):
        # Each mode information
        period_n = 2*np.pi/natural_angfreq[ii]
        stiff_n = (2*np.pi/period_n)**2 # (2*np.pi/period_n)**2
        
        #xi_n = 2*damping_coef*np.sqrt(stiff/1) # 2xi sqrt(k/m)
        damping_n = 2*xi[ii]*np.sqrt(stiff_n/1)  # 2xi sqrt(k/m)
    
        # Run SDOF analysis (depends on the modal resopnse)
        if eigen_vect[0,ii] == 1:
            A_SDOF_span(stiff_n, damping_n, tmp1_1, tmp2)
        else:
            A_SDOF_span(stiff_n, damping_n, tmp1_2, tmp2)
            
        displ_SDOF = np.loadtxt('./Results_SDOF/NodeDisp.out')
        accel_SDOF = np.loadtxt('./Results_SDOF/NodeAccel.out')
        
        # Modal values
        gamma_n = np.matmul(eigen_vect[:,ii].T,np.matmul(mass,s))/Fancy_m[ii,ii]

        # Summation
        q_t = gamma_n*displ_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_displ.append(u_t)

        # Summation
        q_t = gamma_n*accel_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_accel.append(u_t)        
            
    return results_displ, results_accel 


def MDOF_5span_analysis(tmp_value, tmp1_1, tmp1_2, tmp2):    
    import numpy as np
    from i_SDOF import A_SDOF_span

    mass_column, sitff_y_super, stiff_x_column, stiff_y_column, damping, random_values = tmp_value

    # Stiffness y of super structure 
    sitff_y_super1 = sitff_y_super*random_values[0]
    sitff_y_super2 = sitff_y_super*random_values[1]
    sitff_y_super3 = sitff_y_super*random_values[2]
    sitff_y_super4 = sitff_y_super*random_values[3]
    sitff_y_super5 = sitff_y_super*random_values[4]

    # Stiffness y of column
    stiff_y_column1 = stiff_y_column*random_values[5]
    stiff_y_column2 = stiff_y_column*random_values[6]
    stiff_y_column3 = stiff_y_column*random_values[7]
    stiff_y_column4 = stiff_y_column*random_values[8]

    # Modal results
    mass_total = np.ones([20,])
    mass_total[0] = 21 + 4*mass_column
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    mass_total[12] = mass_total[12] + mass_column
    mass_total[16] = mass_total[16] + mass_column
    
    mass = np.diag(mass_total)


    stiff = np.array([[4*stiff_x_column, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0  , 0,0 ],
                      [0, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0  , 0,0,0,0 ],
                      [0,  -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0 , 0, 0, 0, 0, 0, 0, 0, 0,0,0,0 , 0,0,0,0 ],
                      [0, 0, -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0 , 0,0,0,0 ],
                      [0, 0, 0, -sitff_y_super1, sitff_y_super1+sitff_y_super2+stiff_y_column1, -sitff_y_super2, 0, 0,  0, 0, 0, 0, 0,0,0,0, 0,0,0,0 ],
                      [0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0, 0, 0,0,0, 0,0,0,0 ],
                      [0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0, 0,0,0, 0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0,0,0, 0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, -sitff_y_super2, stiff_y_column2+sitff_y_super2+sitff_y_super3, -sitff_y_super3, 0, 0, 0,0,0,0, 0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0, 0, 0,0,0,0, 0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0, 0,0,0,0, 0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0,0,0,0, 0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3,  stiff_y_column3+sitff_y_super3+sitff_y_super4,-sitff_y_super4, 0,0, 0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4, 0,0, 0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4, 0, 0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4, 0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, stiff_y_column4+sitff_y_super4+sitff_y_super5, -sitff_y_super5, 0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0, -sitff_y_super5, 2*sitff_y_super5, -sitff_y_super5, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0, -sitff_y_super5, 2*sitff_y_super5, -sitff_y_super5],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0, -sitff_y_super5, 2*sitff_y_super5]

                      ])

        
    [natural_period, natural_angfreq, eigen_vect,  G, M_eff] = modal_analysis(stiff, mass) 

    Fancy_m = np.matmul(eigen_vect.T, np.matmul(mass, eigen_vect))
    
    a0 = (damping*2*natural_angfreq[0]*natural_angfreq[1]/
      (natural_angfreq[0]+natural_angfreq[1]))
    a1 = damping*2/(natural_angfreq[0]+natural_angfreq[1])
    
    xi = 1/2*(a0/natural_angfreq + a1*natural_angfreq)
    
    results_displ = []; results_accel = []
    s = np.ones([len(stiff),1])
    for ii in range(len(s)):
        # Each mode information
        period_n = 2*np.pi/natural_angfreq[ii]
        stiff_n = (2*np.pi/period_n)**2 # (2*np.pi/period_n)**2
        
        #xi_n = 2*damping_coef*np.sqrt(stiff/1) # 2xi sqrt(k/m)
        damping_n = 2*xi[ii]*np.sqrt(stiff_n/1)  # 2xi sqrt(k/m)
    
        # Run SDOF analysis (depends on the modal resopnse)
        if eigen_vect[0,ii] == 1:
            A_SDOF_span(stiff_n, damping_n, tmp1_1, tmp2)
        else:
            A_SDOF_span(stiff_n, damping_n, tmp1_2, tmp2)
            
        displ_SDOF = np.loadtxt('./Results_SDOF/NodeDisp.out')
        accel_SDOF = np.loadtxt('./Results_SDOF/NodeAccel.out')        

        # Modal values
        gamma_n = np.matmul(eigen_vect[:,ii].T,np.matmul(mass,s))/Fancy_m[ii,ii]
        
        # Summation
        q_t = gamma_n*displ_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_displ.append(u_t)

        # Summation
        q_t = gamma_n*accel_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_accel.append(u_t)        
            
    return results_displ, results_accel                 


def MDOF_6span_analysis(tmp_value, tmp1_1, tmp1_2, tmp2):    
    import numpy as np
    from i_SDOF import A_SDOF_span

    mass_column, sitff_y_super, stiff_x_column, stiff_y_column, damping, random_values = tmp_value

    # Stiffness y of super structure 
    sitff_y_super1 = sitff_y_super*random_values[0]
    sitff_y_super2 = sitff_y_super*random_values[1]
    sitff_y_super3 = sitff_y_super*random_values[2]
    sitff_y_super4 = sitff_y_super*random_values[3]
    sitff_y_super5 = sitff_y_super*random_values[4]
    sitff_y_super6 = sitff_y_super*random_values[5]

    # Stiffness y of column
    stiff_y_column1 = stiff_y_column*random_values[6]
    stiff_y_column2 = stiff_y_column*random_values[7]
    stiff_y_column3 = stiff_y_column*random_values[8]
    stiff_y_column4 = stiff_y_column*random_values[9]
    stiff_y_column5 = stiff_y_column*random_values[10]

    # Modal results
    mass_total = np.ones([24,])
    mass_total[0] = 25 + 5*mass_column
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    mass_total[12] = mass_total[12] + mass_column
    mass_total[16] = mass_total[16] + mass_column
    mass_total[20] = mass_total[20] + mass_column
    
    mass = np.diag(mass_total)

    stiff = np.array([[5*stiff_x_column, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0 ],
                      [0, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0  , 0,0,0,0,0,0,0,0 ],
                      [0,  -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0 , 0, 0, 0, 0, 0, 0, 0, 0,0,0,0 , 0,0,0,0,0,0,0,0 ],
                      [0, 0, -sitff_y_super1, 2*sitff_y_super1, -sitff_y_super1, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0 , 0,0,0,0,0,0,0,0 ],
                      [0, 0, 0, -sitff_y_super1, sitff_y_super1+sitff_y_super2+stiff_y_column1, -sitff_y_super2, 0, 0,  0, 0, 0, 0, 0,0,0,0, 0,0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0, 0, 0,0,0, 0,0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0, 0,0,0, 0,0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, -sitff_y_super2, 2*sitff_y_super2, -sitff_y_super2, 0, 0, 0, 0, 0,0,0, 0,0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, -sitff_y_super2, stiff_y_column2+sitff_y_super2+sitff_y_super3, -sitff_y_super3, 0, 0, 0,0,0,0, 0,0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0, 0, 0,0,0,0, 0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0, 0,0,0,0, 0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3, 2*sitff_y_super3, -sitff_y_super3, 0,0,0,0, 0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super3,  stiff_y_column3+sitff_y_super3+sitff_y_super4,-sitff_y_super4, 0,0, 0,0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4, 0,0, 0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4, 0, 0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, 2*sitff_y_super4, -sitff_y_super4, 0,0,0,0,0,0,0 ],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super4, stiff_y_column4+sitff_y_super4+sitff_y_super5, -sitff_y_super5, 0,0,0,0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super5, 2*sitff_y_super5, -sitff_y_super5, 0,0,0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super5, 2*sitff_y_super5, -sitff_y_super5,0,0,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super5, 2*sitff_y_super5, -sitff_y_super5,0,0,0], #19
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super5, stiff_y_column5+sitff_y_super6+sitff_y_super5, -sitff_y_super6,0,0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super6, 2*sitff_y_super6, -sitff_y_super6, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super6, 2*sitff_y_super6, -sitff_y_super6],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sitff_y_super6, 2*sitff_y_super6]
                      ])

       
    [natural_period, natural_angfreq, eigen_vect,  G, M_eff] = modal_analysis(stiff, mass) 

    Fancy_m = np.matmul(eigen_vect.T, np.matmul(mass, eigen_vect))
    
    a0 = (damping*2*natural_angfreq[0]*natural_angfreq[1]/
      (natural_angfreq[0]+natural_angfreq[1]))
    a1 = damping*2/(natural_angfreq[0]+natural_angfreq[1])
    
    xi = 1/2*(a0/natural_angfreq + a1*natural_angfreq)
    
    results_displ = []; results_accel = []
    s = np.ones([len(stiff),1])
    for ii in range(len(s)):
        # Each mode information
        period_n = 2*np.pi/natural_angfreq[ii]
        stiff_n = (2*np.pi/period_n)**2 # (2*np.pi/period_n)**2
        
        #xi_n = 2*damping_coef*np.sqrt(stiff/1) # 2xi sqrt(k/m)
        damping_n = 2*xi[ii]*np.sqrt(stiff_n/1)  # 2xi sqrt(k/m)
    
        # Run SDOF analysis (depends on the modal resopnse)
        if eigen_vect[0,ii] == 1:
            A_SDOF_span(stiff_n, damping_n, tmp1_1, tmp2)
        else:
            A_SDOF_span(stiff_n, damping_n, tmp1_2, tmp2)
            
        displ_SDOF = np.loadtxt('./Results_SDOF/NodeDisp.out')
        accel_SDOF = np.loadtxt('./Results_SDOF/NodeAccel.out')
        
        # Modal values
        gamma_n = np.matmul(eigen_vect[:,ii].T,np.matmul(mass,s))/Fancy_m[ii,ii]
        
        # Summation
        q_t = gamma_n*displ_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_displ.append(u_t)

        # Summation
        q_t = gamma_n*accel_SDOF # Displacement; m --> mm
        u_t = np.matmul(eigen_vect[:,ii].reshape(len(Fancy_m),1),
                        q_t.reshape(1,len(q_t)))
        

        results_accel.append(u_t)        
            
    return results_displ, results_accel             








