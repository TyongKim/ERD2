#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:57:27 2023

@author: taeyongkim
"""

def A_SDOF_3span(stiff_n, damping_n, tmp1, tmp2):

    import numpy as np
    import os
    
    # Make OpenSees file
    file_name = 'Model_SDOF3.tcl'
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
    file_name = 'DynAnalysis_SDOF3.tcl'        
    f = open(file_name, 'w')    # Open file     
    f.write('set Factor             %f; \n' %(scale_factor))
    f.write('set dt                 %f; \n' %(gm2))
    f.write('set timeInc            %f; \n' %(timeInC))
    f.write('set Nsteps             %f; \n' %(int(len(gm1)*1.3)))
    f.write('set GMfile             %s; \n' %(tmp1))
    f.close()    
    
    # Records
    file_name = 'Records_SDOF3.tcl'
    f = open(file_name, 'w')
    # Number of GM and Structure 
    f.write('recorder Node -file  ./Results_SDOF/NodeDisp3.out    -node 10 -dof 1  disp;' ); # outer    
    f.write(" \n\n\n")
    f.close()    
    
    # Run dynamic analysis
    os.system('OpenSees Main_Dynamic_OpenSees_SDOF3.tcl')      

def A_SDOF_4span(stiff_n, damping_n, tmp1, tmp2):

    import numpy as np
    import os
    
    # Make OpenSees file
    file_name = 'Model_SDOF4.tcl'
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
    file_name = 'DynAnalysis_SDOF4.tcl'        
    f = open(file_name, 'w')    # Open file     
    f.write('set Factor             %f; \n' %(scale_factor))
    f.write('set dt                 %f; \n' %(gm2))
    f.write('set timeInc            %f; \n' %(timeInC))
    f.write('set Nsteps             %f; \n' %(int(len(gm1)*1.3)))
    f.write('set GMfile             %s; \n' %(tmp1))
    f.close()    
    
    # Records
    file_name = 'Records_SDOF4.tcl'
    f = open(file_name, 'w')
    # Number of GM and Structure 
    f.write('recorder Node -file  ./Results_SDOF/NodeDisp4.out    -node 10 -dof 1  disp;' ); # outer    
    f.write(" \n\n\n")
    f.close()    
    
    # Run dynamic analysis
    os.system('OpenSees Main_Dynamic_OpenSees_SDOF4.tcl')      

def A_SDOF_5span(stiff_n, damping_n, tmp1, tmp2):

    import numpy as np
    import os
    
    # Make OpenSees file
    file_name = 'Model_SDOF5.tcl'
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
    file_name = 'DynAnalysis_SDOF5.tcl'        
    f = open(file_name, 'w')    # Open file     
    f.write('set Factor             %f; \n' %(scale_factor))
    f.write('set dt                 %f; \n' %(gm2))
    f.write('set timeInc            %f; \n' %(timeInC))
    f.write('set Nsteps             %f; \n' %(int(len(gm1)*1.3)))
    f.write('set GMfile             %s; \n' %(tmp1))
    f.close()    
    
    # Records
    file_name = 'Records_SDOF5.tcl'
    f = open(file_name, 'w')
    # Number of GM and Structure 
    f.write('recorder Node -file  ./Results_SDOF/NodeDisp5.out    -node 10 -dof 1  disp;' ); # outer    
    f.write(" \n\n\n")
    f.close()    
    
    # Run dynamic analysis
    os.system('OpenSees Main_Dynamic_OpenSees_SDOF5.tcl')      
    

def A_SDOF_6span(stiff_n, damping_n, tmp1, tmp2):

    import numpy as np
    import os
    
    # Make OpenSees file
    file_name = 'Model_SDOF6.tcl'
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
    file_name = 'DynAnalysis_SDOF6.tcl'        
    f = open(file_name, 'w')    # Open file     
    f.write('set Factor             %f; \n' %(scale_factor))
    f.write('set dt                 %f; \n' %(gm2))
    f.write('set timeInc            %f; \n' %(timeInC))
    f.write('set Nsteps             %f; \n' %(int(len(gm1)*1.3)))
    f.write('set GMfile             %s; \n' %(tmp1))
    f.close()    
    
    # Records
    file_name = 'Records_SDOF6.tcl'
    f = open(file_name, 'w')
    # Number of GM and Structure 
    f.write('recorder Node -file  ./Results_SDOF/NodeDisp6.out    -node 10 -dof 1  disp;' ); # outer    
    f.write(" \n\n\n")
    f.close()    
    
    # Run dynamic analysis
    os.system('OpenSees Main_Dynamic_OpenSees_SDOF6.tcl')      
    