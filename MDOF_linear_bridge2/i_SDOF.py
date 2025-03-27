"""
This code is to perform dynamic anlaysis for single degree of freedom structure
under an earthquake.

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""

def A_SDOF_span(stiff_n, damping_n, tmp1, tmp2):

    import numpy as np
    import os
    
    # Make OpenSees file
    file_name = 'Model_SDOF.tcl'
    f = open(file_name, 'w')
    f.write('uniaxialMaterial Elastic 10   %f  \n' %(stiff_n)); # stiffness
    f.write('uniaxialMaterial Viscous 11   %f  1.0  \n' %(damping_n)); # viscous damping
    f.write(" \n\n\n")
    f.close()            
        
    
    # Load time history acceleration
    gm1 = np.loadtxt(tmp1)
    gm2 = np.loadtxt(tmp2)
    
    scale_factor = 1
    timeInC = np.min([gm2, 0.001])
    
    # Ground motions
    file_name = 'DynAnalysis_SDOF.tcl'        
    f = open(file_name, 'w')    # Open file     
    f.write('set Factor             %f; \n' %(scale_factor))
    f.write('set dt                 %f; \n' %(gm2))
    f.write('set timeInc            %f; \n' %(timeInC))
    f.write('set Nsteps             %f; \n' %(int(len(gm1)*1.3)))
    f.write('set GMfile             %s; \n' %(tmp1))
    f.close()    
    
    # Records
    file_name = 'Records_SDOF.tcl'
    f = open(file_name, 'w')
    # Number of GM and Structure 
    f.write('recorder Node -file  ./Results_SDOF/NodeDisp.out    -node 10 -dof 1  disp;' );     
    f.write('recorder Node -file  ./Results_SDOF/NodeAccel.out   -node 10 -dof 1  -timeSeries $tsTag accel;' );         
    f.write(" \n\n\n")
    f.close()    
    
    # Run dynamic analysis
    os.system('OpenSees Main_Dynamic_OpenSees_SDOF.tcl')      

