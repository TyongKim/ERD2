"""
This script performs dynamic anlaysis of MDOF systems under bi-directional ground motions.
OpenSeespy is employed.

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""



def OPS_3span(tmp_value2, tmp1_1, tmp1_2, tmp2):
    
    import openseespy.opensees  as ops
    import numpy as np

    [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]= tmp_value2

    # Modal anlaysis
    DOF = 15
    mass_total = np.ones([DOF-2,])
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    
    ops.wipe()
    ops.model('basic','-ndm',2,'-ndf',2)
    
    # Define node
    for ii in range(DOF):
        ops.node(ii+1, 0.0, 0.0)
    
    # Define mass
    for ii in range(DOF-2):
        if ii==0:
            ops.mass(ii+1, mass_total[ii], 0)
        elif ii== DOF-3:
            ops.mass(ii+1, mass_total[ii], 0)
        else:            
            ops.mass(ii+1, mass_total[ii], mass_total[ii])

    # Define equalDOF
    for ii in range(DOF-3):
        ops.equalDOF(1, ii+2, 1) # X constrained for the DOFs on the girder

    # Define boundary
    ops.fix(1,  0, 1)
    ops.fix(13, 0, 1)
    ops.fix(14, 1, 1)
    ops.fix(15, 1, 1)


    # Define material
    # Stiffness y of super structure 
    sitff_y_super1 = sitff_y_super*random_values[0]
    sitff_y_super2 = sitff_y_super*random_values[1]
    sitff_y_super3 = sitff_y_super*random_values[2]

    # Stiffness y of column
    stiff_y_column1 = stiff_y_column*random_values[3]
    stiff_y_column2 = stiff_y_column*random_values[4]



    ops.uniaxialMaterial('Elastic', 1, sitff_y_super1) 
    ops.uniaxialMaterial('Elastic', 2, sitff_y_super2) 
    ops.uniaxialMaterial('Elastic', 3, sitff_y_super3)

    ops.uniaxialMaterial('Elastic', 10, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 11, stiff_x_column) 

    ops.uniaxialMaterial('Elastic', 20, stiff_y_column1) 
    ops.uniaxialMaterial('Elastic', 21, stiff_y_column2) 
    
    
    
    # Define elements
    # Span 1
    for ii in range(4):
        ops.element("zeroLength", ii+1, ii+1, ii+2, '-mat', 1,'-dir', 2, '-doRayleigh', 1)
    # Span 2
    for ii in range(4):
        ops.element("zeroLength", ii+5, ii+5, ii+6, '-mat', 2,'-dir', 2, '-doRayleigh', 1)    
    # Span 3
    for ii in range(4):
        ops.element("zeroLength", ii+9, ii+9, ii+10, '-mat', 3,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 1
    ops.element("zeroLength", 100, 5, 14, '-mat', 10,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 101, 5, 14, '-mat', 20,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 2
    ops.element("zeroLength", 200, 9, 15, '-mat', 11,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 201, 9, 15, '-mat', 21,'-dir', 2, '-doRayleigh', 1)    
    
    
    # Damping
    # Rayleigh damping
    lambda_total = ops.eigen(3); 
    #ops.modalProperties('-print', '-file', 'ModalReport.txt', '-unorm')
    ops.wipeAnalysis()
    
    lambda1 = lambda_total[0]
    lambda2 = lambda_total[1]
    #lambda3 = lambda_total[2]
    omega1 = np.sqrt(lambda1)
    omega2 = np.sqrt(lambda2)
    #omega3 = np.sqrt(lambda3)
    period1 = 2*np.pi/omega1
    
    alphaM = 2*omega1*omega2*tmp_damping/(omega1+omega2)
    betaKcurr = 2*tmp_damping/(omega1+omega2)
    ops.rayleigh(alphaM, betaKcurr, 0, 0)
    
    # Eigen mode
    #ops.recorder('Node', '-file', 'Results/mode1.out', '-nodeRange', 1, 7, '-dof', 1, 2, 'eigen', 1)

    # Ground motion
    dt = float(np.loadtxt(tmp2))
    Nsteps = len(np.loadtxt(tmp1_1))
    timeInc = np.min([dt,0.001])
    scale_factor = 9.81   
    
    # Ground motions    
    tsTag1 = 1
    tsTag2 = 2
    ptTag1 = 1
    ptTag2 = 2
    
    ops.timeSeries('Path', tsTag1, '-dt', dt, '-filePath', tmp1_1, '-factor', scale_factor)
    ops.timeSeries('Path', tsTag2, '-dt', dt, '-filePath', tmp1_2, '-factor', scale_factor)
    ops.pattern('UniformExcitation', ptTag1, 1, '-accel', tsTag1)
    ops.pattern('UniformExcitation', ptTag2, 2, '-accel', tsTag2)

    # Record
    tmp_record_d = './Results_MDOF/DFree.out'     
    tmp_record_a1 = './Results_MDOF/AFree1_3.out' 
    tmp_record_a2 = './Results_MDOF/AFree2_3.out' 

    ops.recorder('Node', '-file', tmp_record_a1, '-time', '-node', 3, 7, 11, '-dof', 1,  '-timeSeries', tsTag1, 'accel')
    ops.recorder('Node', '-file', tmp_record_a2, '-time', '-node', 3, 7, 11, '-dof', 2,  '-timeSeries', tsTag2, 'accel')    
    ops.recorder('Node', '-file', tmp_record_d, '-time', '-node', 3, 7, 11, '-dof', 1,2, 'disp' )
    
    # Analysis parameters
    ops.wipeAnalysis()
    Tol = 1.0e-7;                     # convergence tolerance for test
    ops.constraints('Transformation')
    ops.numberer('Plain')
    ops.system('UmfPack')
    ops.test('NormDispIncr', Tol, 100)
    ops.algorithm('ModifiedNewton')
    
    NewmarkGamma = 0.5
    NewmarkBeta = 0.25
    ops.integrator('Newmark', NewmarkGamma, NewmarkBeta)
    ops.analysis('Transient')
    
    
    numSteps =  int(Nsteps*dt/timeInc)
    
    ok = ops.analyze(numSteps+int(period1*5/timeInc), timeInc)
    ops.wipe()
    
    # Post process
    results = np.loadtxt(tmp_record_d)
    results1 = np.loadtxt(tmp_record_a1)
    results2 = np.loadtxt(tmp_record_a2)
    
    return [results, results1, results2]

def OPS_4span(tmp_value2, tmp1_1, tmp1_2, tmp2):
    
    import openseespy.opensees  as ops
    import numpy as np

    [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]= tmp_value2

    # Modal anlaysis
    DOF = 20
    mass_total = np.ones([DOF-3,])
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    mass_total[12] = mass_total[12] + mass_column
    
    ops.wipe()
    ops.model('basic','-ndm',2,'-ndf',2)
    
    # Define node
    for ii in range(DOF):
        ops.node(ii+1, 0.0, 0.0)
    
    # Define mass
    for ii in range(DOF-3):
        if ii==0:
            ops.mass(ii+1, mass_total[ii], 0)
        elif ii== DOF-4:
            ops.mass(ii+1, mass_total[ii], 0)
        else:            
            ops.mass(ii+1, mass_total[ii], mass_total[ii])

    # Define equalDOF
    for ii in range(DOF-4):
        ops.equalDOF(1, ii+2, 1) # X constrained for the DOFs on the girder

    # Define boundary
    ops.fix(1,  0, 1)
    ops.fix(17, 0, 1)
    ops.fix(18, 1, 1)
    ops.fix(19, 1, 1)
    ops.fix(20, 1, 1)


    # Define material
    # Stiffness y of super structure 
    sitff_y_super1 = sitff_y_super*random_values[0]
    sitff_y_super2 = sitff_y_super*random_values[1]
    sitff_y_super3 = sitff_y_super*random_values[2]
    sitff_y_super4 = sitff_y_super*random_values[3]

    # Stiffness y of column
    stiff_y_column1 = stiff_y_column*random_values[4]
    stiff_y_column2 = stiff_y_column*random_values[5]
    stiff_y_column3 = stiff_y_column*random_values[6]
    

    ops.uniaxialMaterial('Elastic', 1, sitff_y_super1) 
    ops.uniaxialMaterial('Elastic', 2, sitff_y_super2) 
    ops.uniaxialMaterial('Elastic', 3, sitff_y_super3)
    ops.uniaxialMaterial('Elastic', 4, sitff_y_super4)
    
    ops.uniaxialMaterial('Elastic', 10, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 11, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 12, stiff_x_column) 

    ops.uniaxialMaterial('Elastic', 20, stiff_y_column1) 
    ops.uniaxialMaterial('Elastic', 21, stiff_y_column2) 
    ops.uniaxialMaterial('Elastic', 22, stiff_y_column3) 
    
    
    
    # Define elements
    # Span 1
    for ii in range(4):
        ops.element("zeroLength", ii+1, ii+1, ii+2, '-mat', 1,'-dir', 2, '-doRayleigh', 1)
    # Span 2
    for ii in range(4):
        ops.element("zeroLength", ii+5, ii+5, ii+6, '-mat', 2,'-dir', 2, '-doRayleigh', 1)    
    # Span 3
    for ii in range(4):
        ops.element("zeroLength", ii+9, ii+9, ii+10, '-mat', 3,'-dir', 2, '-doRayleigh', 1)    
    # Span 4
    for ii in range(4):
        ops.element("zeroLength", ii+13, ii+13, ii+14, '-mat', 4,'-dir', 2, '-doRayleigh', 1)    
        
    # Column 1
    ops.element("zeroLength", 100, 5, 18, '-mat', 10,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 101, 5, 18, '-mat', 20,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 2
    ops.element("zeroLength", 200, 9, 19, '-mat', 11,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 201, 9, 19, '-mat', 21,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 3
    ops.element("zeroLength", 300, 13, 20, '-mat', 12,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 301, 13, 20, '-mat', 22,'-dir', 2, '-doRayleigh', 1)        
    
    # Damping
    # Rayleigh damping
    lambda_total = ops.eigen(3); 
    #ops.modalProperties('-print', '-file', 'ModalReport.txt', '-unorm')
    ops.wipeAnalysis()
    
    lambda1 = lambda_total[0]
    lambda2 = lambda_total[1]
    #lambda3 = lambda_total[2]
    omega1 = np.sqrt(lambda1)
    omega2 = np.sqrt(lambda2)
    #omega3 = np.sqrt(lambda3)
    period1 = 2*np.pi/omega1
    
    alphaM = 2*omega1*omega2*tmp_damping/(omega1+omega2)
    betaKcurr = 2*tmp_damping/(omega1+omega2)
    ops.rayleigh(alphaM, betaKcurr, 0, 0)
    
    # Eigen mode
    #ops.recorder('Node', '-file', 'Results/mode1.out', '-nodeRange', 1, 7, '-dof', 1, 2, 'eigen', 1)

    # Ground motion
    dt = float(np.loadtxt(tmp2))
    Nsteps = len(np.loadtxt(tmp1_1))
    timeInc = np.min([dt,0.001])
    scale_factor = 9.81   
    
    # Ground motions    
    tsTag1 = 1
    tsTag2 = 2
    ptTag1 = 1
    ptTag2 = 2
    
    ops.timeSeries('Path', tsTag1, '-dt', dt, '-filePath', tmp1_1, '-factor', scale_factor)
    ops.timeSeries('Path', tsTag2, '-dt', dt, '-filePath', tmp1_2, '-factor', scale_factor)
    ops.pattern('UniformExcitation', ptTag1, 1, '-accel', tsTag1)
    ops.pattern('UniformExcitation', ptTag2, 2, '-accel', tsTag2)
    
    # Record
    # Central node of each span 
    tmp_record_d = './Results_MDOF/DFree_4span.out' 
    tmp_record_a1 = './Results_MDOF/AFree1_4.out' 
    tmp_record_a2 = './Results_MDOF/AFree2_4.out' 
    
    ops.recorder('Node', '-file', tmp_record_d, '-time', '-node', 3, 7, 11, 15, '-dof', 1,2, 'disp' )
    ops.recorder('Node', '-file', tmp_record_a1, '-time', '-node', 3, 7, 11, 15, '-dof', 1,  '-timeSeries', tsTag1, 'accel')
    ops.recorder('Node', '-file', tmp_record_a2, '-time', '-node', 3, 7, 11, 15, '-dof', 2,  '-timeSeries', tsTag2, 'accel')   
    
    
    # Analysis parameters
    ops.wipeAnalysis()
    Tol = 1.0e-7;                     # convergence tolerance for test
    ops.constraints('Transformation')
    ops.numberer('Plain')
    ops.system('UmfPack')
    ops.test('NormDispIncr', Tol, 100)
    ops.algorithm('ModifiedNewton')
    
    NewmarkGamma = 0.5
    NewmarkBeta = 0.25
    ops.integrator('Newmark', NewmarkGamma, NewmarkBeta)
    ops.analysis('Transient')
    
    
    numSteps =  int(Nsteps*dt/timeInc)
    
    ok = ops.analyze(numSteps+int(period1*5/timeInc), timeInc)
    ops.wipe()
    
    # Post process
    results = np.loadtxt(tmp_record_d)
    results1 = np.loadtxt(tmp_record_a1)
    results2 = np.loadtxt(tmp_record_a2)
    
    return [results, results1, results2]


def OPS_5span(tmp_value2, tmp1_1, tmp1_2, tmp2):
    
    import openseespy.opensees  as ops
    import numpy as np

    [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]= tmp_value2

    # Modal anlaysis
    DOF = 25
    mass_total = np.ones([DOF-4,])
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    mass_total[12] = mass_total[12] + mass_column
    mass_total[16] = mass_total[16] + mass_column
    
    ops.wipe()
    ops.model('basic','-ndm',2,'-ndf',2)
    
    # Define node
    for ii in range(DOF):
        ops.node(ii+1, 0.0, 0.0)
    
    # Define mass
    for ii in range(DOF-4):
        if ii==0:
            ops.mass(ii+1, mass_total[ii], 0)
        elif ii== DOF-5:
            ops.mass(ii+1, mass_total[ii], 0)
        else:            
            ops.mass(ii+1, mass_total[ii], mass_total[ii])

    # Define equalDOF
    for ii in range(DOF-5):
        ops.equalDOF(1, ii+2, 1) # X constrained for the DOFs on the girder

    # Define boundary
    ops.fix(1,  0, 1)
    ops.fix(21, 0, 1)
    ops.fix(22, 1, 1)
    ops.fix(23, 1, 1)
    ops.fix(24, 1, 1)
    ops.fix(25, 1, 1)


    # Define material
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

    

    ops.uniaxialMaterial('Elastic', 1, sitff_y_super1) 
    ops.uniaxialMaterial('Elastic', 2, sitff_y_super2) 
    ops.uniaxialMaterial('Elastic', 3, sitff_y_super3)
    ops.uniaxialMaterial('Elastic', 4, sitff_y_super4)
    ops.uniaxialMaterial('Elastic', 5, sitff_y_super5)
    
    ops.uniaxialMaterial('Elastic', 10, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 11, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 12, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 13, stiff_x_column) 

    ops.uniaxialMaterial('Elastic', 20, stiff_y_column1) 
    ops.uniaxialMaterial('Elastic', 21, stiff_y_column2) 
    ops.uniaxialMaterial('Elastic', 22, stiff_y_column3) 
    ops.uniaxialMaterial('Elastic', 23, stiff_y_column4) 
    
    
    
    # Define elements
    # Span 1
    for ii in range(4):
        ops.element("zeroLength", ii+1, ii+1, ii+2, '-mat', 1,'-dir', 2, '-doRayleigh', 1)
    # Span 2
    for ii in range(4):
        ops.element("zeroLength", ii+5, ii+5, ii+6, '-mat', 2,'-dir', 2, '-doRayleigh', 1)    
    # Span 3
    for ii in range(4):
        ops.element("zeroLength", ii+9, ii+9, ii+10, '-mat', 3,'-dir', 2, '-doRayleigh', 1)    
    # Span 4
    for ii in range(4):
        ops.element("zeroLength", ii+13, ii+13, ii+14, '-mat', 4,'-dir', 2, '-doRayleigh', 1)    
    # Span 4
    for ii in range(4):
        ops.element("zeroLength", ii+17, ii+17, ii+18, '-mat', 5,'-dir', 2, '-doRayleigh', 1)   
        
    # Column 1
    ops.element("zeroLength", 100, 5, 22, '-mat', 10,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 101, 5, 22, '-mat', 20,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 2
    ops.element("zeroLength", 200, 9, 23, '-mat', 11,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 201, 9, 23, '-mat', 21,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 3
    ops.element("zeroLength", 300, 13, 24, '-mat', 12,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 301, 13, 24, '-mat', 22,'-dir', 2, '-doRayleigh', 1)        
    
    # Column 4
    ops.element("zeroLength", 400, 17, 25, '-mat', 13,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 401, 17, 25, '-mat', 23,'-dir', 2, '-doRayleigh', 1)  
    
    # Damping
    # Rayleigh damping
    lambda_total = ops.eigen(3); 
    #ops.modalProperties('-print', '-file', 'ModalReport.txt', '-unorm')
    ops.wipeAnalysis()
    
    lambda1 = lambda_total[0]
    lambda2 = lambda_total[1]
    #lambda3 = lambda_total[2]
    omega1 = np.sqrt(lambda1)
    omega2 = np.sqrt(lambda2)
    #omega3 = np.sqrt(lambda3)
    period1 = 2*np.pi/omega1
    
    alphaM = 2*omega1*omega2*tmp_damping/(omega1+omega2)
    betaKcurr = 2*tmp_damping/(omega1+omega2)
    ops.rayleigh(alphaM, betaKcurr, 0, 0)
    
    # Eigen mode
    #ops.recorder('Node', '-file', 'Results/mode1.out', '-nodeRange', 1, 7, '-dof', 1, 2, 'eigen', 1)

    # Ground motion
    dt = float(np.loadtxt(tmp2))
    Nsteps = len(np.loadtxt(tmp1_1))
    timeInc = np.min([dt,0.001])
    scale_factor = 9.81   
    
    # Ground motions    
    tsTag1 = 1
    tsTag2 = 2
    ptTag1 = 1
    ptTag2 = 2
    
    ops.timeSeries('Path', tsTag1, '-dt', dt, '-filePath', tmp1_1, '-factor', scale_factor)
    ops.timeSeries('Path', tsTag2, '-dt', dt, '-filePath', tmp1_2, '-factor', scale_factor)
    ops.pattern('UniformExcitation', ptTag1, 1, '-accel', tsTag1)
    ops.pattern('UniformExcitation', ptTag2, 2, '-accel', tsTag2)
    
    # Record
    # Central node of each span 
    tmp_record_a1 = './Results_MDOF/AFree1_5.out' 
    tmp_record_a2 = './Results_MDOF/AFree2_5.out' 
    tmp_record_d = './Results_MDOF/DFree_5span.out' 

    ops.recorder('Node', '-file', tmp_record_d, '-time', '-node', 3, 7, 11, 15, 19, '-dof', 1,2, 'disp' )  
    ops.recorder('Node', '-file', tmp_record_a1, '-time', '-node', 3, 7, 11, 15, 19, '-dof', 1,  '-timeSeries', tsTag1, 'accel')
    ops.recorder('Node', '-file', tmp_record_a2, '-time', '-node', 3, 7, 11, 15, 19, '-dof', 2,  '-timeSeries', tsTag2, 'accel')   
    
    
    # Analysis parameters
    ops.wipeAnalysis()
    Tol = 1.0e-7;                     # convergence tolerance for test
    ops.constraints('Transformation')
    ops.numberer('Plain')
    ops.system('UmfPack')
    ops.test('NormDispIncr', Tol, 100)
    ops.algorithm('ModifiedNewton')
    
    NewmarkGamma = 0.5
    NewmarkBeta = 0.25
    ops.integrator('Newmark', NewmarkGamma, NewmarkBeta)
    ops.analysis('Transient')
    
    
    numSteps =  int(Nsteps*dt/timeInc)
    
    ok = ops.analyze(numSteps+int(period1*5/timeInc), timeInc)
    ops.wipe()
    
    # Post process
    results = np.loadtxt(tmp_record_d)
    results1 = np.loadtxt(tmp_record_a1)
    results2 = np.loadtxt(tmp_record_a2)
    
    return [results, results1, results2]

def OPS_6span(tmp_value2, tmp1_1, tmp1_2, tmp2):
    
    import openseespy.opensees  as ops
    import numpy as np

    [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]= tmp_value2

    # Modal anlaysis
    DOF = 30
    mass_total = np.ones([DOF-5,])
    mass_total[4] = mass_total[4] + mass_column
    mass_total[8] = mass_total[8] + mass_column
    mass_total[12] = mass_total[12] + mass_column
    mass_total[16] = mass_total[16] + mass_column
    mass_total[20] = mass_total[20] + mass_column
    
    ops.wipe()
    ops.model('basic','-ndm',2,'-ndf',2)
    
    # Define node
    for ii in range(DOF):
        ops.node(ii+1, 0.0, 0.0)
    
    # Define mass
    for ii in range(DOF-5):
        if ii==0:
            ops.mass(ii+1, mass_total[ii], 0)
        elif ii== DOF-6:
            ops.mass(ii+1, mass_total[ii], 0)
        else:            
            ops.mass(ii+1, mass_total[ii], mass_total[ii])

    # Define equalDOF
    for ii in range(DOF-6):
        ops.equalDOF(1, ii+2, 1) # X constrained for the DOFs on the girder

    # Define boundary
    ops.fix(1,  0, 1)
    ops.fix(25, 0, 1)
    ops.fix(26, 1, 1)
    ops.fix(27, 1, 1)
    ops.fix(28, 1, 1)
    ops.fix(29, 1, 1)
    ops.fix(30, 1, 1)

    # Define material
    # Stiffness y of super structure
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

    ops.uniaxialMaterial('Elastic', 1, sitff_y_super1) 
    ops.uniaxialMaterial('Elastic', 2, sitff_y_super2) 
    ops.uniaxialMaterial('Elastic', 3, sitff_y_super3)
    ops.uniaxialMaterial('Elastic', 4, sitff_y_super4)
    ops.uniaxialMaterial('Elastic', 5, sitff_y_super5)
    ops.uniaxialMaterial('Elastic', 6, sitff_y_super6)
    
    ops.uniaxialMaterial('Elastic', 10, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 11, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 12, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 13, stiff_x_column) 
    ops.uniaxialMaterial('Elastic', 14, stiff_x_column) 

    ops.uniaxialMaterial('Elastic', 20, stiff_y_column1) 
    ops.uniaxialMaterial('Elastic', 21, stiff_y_column2) 
    ops.uniaxialMaterial('Elastic', 22, stiff_y_column3) 
    ops.uniaxialMaterial('Elastic', 23, stiff_y_column4) 
    ops.uniaxialMaterial('Elastic', 24, stiff_y_column5) 
    
    
    
    # Define elements
    # Span 1
    for ii in range(4):
        ops.element("zeroLength", ii+1, ii+1, ii+2, '-mat', 1,'-dir', 2, '-doRayleigh', 1)
    # Span 2
    for ii in range(4):
        ops.element("zeroLength", ii+5, ii+5, ii+6, '-mat', 2,'-dir', 2, '-doRayleigh', 1)    
    # Span 3
    for ii in range(4):
        ops.element("zeroLength", ii+9, ii+9, ii+10, '-mat', 3,'-dir', 2, '-doRayleigh', 1)    
    # Span 4
    for ii in range(4):
        ops.element("zeroLength", ii+13, ii+13, ii+14, '-mat', 4,'-dir', 2, '-doRayleigh', 1)    
    # Span 5
    for ii in range(4):
        ops.element("zeroLength", ii+17, ii+17, ii+18, '-mat', 5,'-dir', 2, '-doRayleigh', 1)   
    # Span 6
    for ii in range(4):
        ops.element("zeroLength", ii+21, ii+21, ii+22, '-mat', 6,'-dir', 2, '-doRayleigh', 1)   
        
    # Column 1
    ops.element("zeroLength", 100, 5, 26, '-mat', 10,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 101, 5, 26, '-mat', 20,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 2
    ops.element("zeroLength", 200, 9, 27, '-mat', 11,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 201, 9, 27, '-mat', 21,'-dir', 2, '-doRayleigh', 1)    
    
    # Column 3
    ops.element("zeroLength", 300, 13, 28, '-mat', 12,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 301, 13, 28, '-mat', 22,'-dir', 2, '-doRayleigh', 1)        
    
    # Column 4
    ops.element("zeroLength", 400, 17, 29, '-mat', 13,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 401, 17, 29, '-mat', 23,'-dir', 2, '-doRayleigh', 1)  
    
    # Column 5
    ops.element("zeroLength", 500, 21, 30, '-mat', 14,'-dir', 1, '-doRayleigh', 1)    
    ops.element("zeroLength", 501, 21, 30, '-mat', 24,'-dir', 2, '-doRayleigh', 1)  
        
    # Damping
    # Rayleigh damping
    lambda_total = ops.eigen(3); 
    #ops.modalProperties('-print', '-file', 'ModalReport.txt', '-unorm')
    ops.wipeAnalysis()
    
    lambda1 = lambda_total[0]
    lambda2 = lambda_total[1]
    #lambda3 = lambda_total[2]
    omega1 = np.sqrt(lambda1)
    omega2 = np.sqrt(lambda2)
    #omega3 = np.sqrt(lambda3)
    period1 = 2*np.pi/omega1
    
    alphaM = 2*omega1*omega2*tmp_damping/(omega1+omega2)
    betaKcurr = 2*tmp_damping/(omega1+omega2)
    ops.rayleigh(alphaM, betaKcurr, 0, 0)
    
    # Eigen mode
    #ops.recorder('Node', '-file', 'Results/mode1.out', '-nodeRange', 1, 7, '-dof', 1, 2, 'eigen', 1)

    # Ground motion
    dt = float(np.loadtxt(tmp2))
    Nsteps = len(np.loadtxt(tmp1_1))
    timeInc = np.min([dt,0.001])
    scale_factor = 9.81   
    
    # Ground motions    
    tsTag1 = 1
    tsTag2 = 2
    ptTag1 = 1
    ptTag2 = 2
    
    ops.timeSeries('Path', tsTag1, '-dt', dt, '-filePath', tmp1_1, '-factor', scale_factor)
    ops.timeSeries('Path', tsTag2, '-dt', dt, '-filePath', tmp1_2, '-factor', scale_factor)
    ops.pattern('UniformExcitation', ptTag1, 1, '-accel', tsTag1)
    ops.pattern('UniformExcitation', ptTag2, 2, '-accel', tsTag2)

    # Record
    tmp_record_d = './Results_MDOF/DFree_6span.out' 
    tmp_record_a1 = './Results_MDOF/AFree1_6.out' 
    tmp_record_a2 = './Results_MDOF/AFree2_6.out' 

    ops.recorder('Node', '-file', tmp_record_d, '-time', '-node', 3, 7, 11, 15, 19, 23, '-dof', 1,2, 'disp' )
    ops.recorder('Node', '-file', tmp_record_a1, '-time', '-node', 3, 7, 11, 15, 19, 23, '-dof', 1,  '-timeSeries', tsTag1, 'accel')
    ops.recorder('Node', '-file', tmp_record_a2, '-time', '-node', 3, 7, 11, 15, 19, 23, '-dof', 2,  '-timeSeries', tsTag2, 'accel')   
    
    
    # Analysis parameters
    ops.wipeAnalysis()
    Tol = 1.0e-7;                     # convergence tolerance for test
    ops.constraints('Transformation')
    ops.numberer('Plain')
    ops.system('UmfPack')
    ops.test('NormDispIncr', Tol, 100)
    ops.algorithm('ModifiedNewton')
    
    NewmarkGamma = 0.5
    NewmarkBeta = 0.25
    ops.integrator('Newmark', NewmarkGamma, NewmarkBeta)
    ops.analysis('Transient')
    
    
    numSteps =  int(Nsteps*dt/timeInc)
    
    ok = ops.analyze(numSteps+int(period1*5/timeInc), timeInc)
    ops.wipe()
    
    # Post process
    results = np.loadtxt(tmp_record_d)
    results1 = np.loadtxt(tmp_record_a1)
    results2 = np.loadtxt(tmp_record_a2)
    
    return [results, results1, results2]