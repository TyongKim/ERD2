###############################################################################
# This code is developed by Prof. Taeyong Kim at Ajou University              #
# taeyongkim@ajou.ac.kr                                                       #
# Jan 20, 2024                                                                #
###############################################################################

# Import libraries
import numpy as np
import pandas as pd
import os
import openseespy.opensees as ops
print(np.__version__)
print(pd.__version__)
#%% Define OpenSeespy Model
def run_dynamic(stiff, period, damping_candi, GMfile, dtfile, idx_gm):
    
    # Make OpenSees file
    ops.wipe()
    ops.model('basic','-ndm',1,'-ndf',1)
    ops.node(1, 0.0)
    ops.fix(1, 1)
    
    # GM
    ###########################################################################
    scale_factor = 9.81 # g
    dt = np.loadtxt(dtfile)    
    
    ###########################################################################

    dt = float(dt)
    Nsteps = np.loadtxt(GMfile)
    Nsteps = len(Nsteps)
    timeInc = np.min([dt,period/100])

    tsTag = 1
    ptTag = 1

    ops.timeSeries('Path', tsTag, '-dt', dt, '-filePath', GMfile, '-factor', scale_factor)
    ops.pattern('UniformExcitation', ptTag, 1, '-accel', tsTag)
        
    # Make many SDOF systems for different damping values
    for ii in range(len(damping_candi)):
        
        # Calculate damping
        damping_n = 2*damping_candi[ii]*np.sqrt(stiff/1)
        
        # structural info
        ops.node(10*(ii+1), 0.0)
        ops.mass(10*(ii+1), 1.0)
        ops.uniaxialMaterial('Elastic', 10*(ii+1), stiff)
        ops.uniaxialMaterial('Viscous', 10*(ii+1)+1, damping_n, 1.0)
        
        ops.element("zeroLength", 10*(ii+1), 1, 10*(ii+1), '-mat', 10*(ii+1),'-dir', 1)
        ops.element("zeroLength", 10*(ii+1)+1, 1, 10*(ii+1), '-mat', 10*(ii+1)+1,'-dir', 1)
            
        # Output recorders    
        tmp_record_d = './Results_displ/DFree_%d_%f_%d.out' %(idx_gm, period, ii)

        ops.recorder('Node', '-file', tmp_record_d, '-time', '-node', 10*(ii+1), '-dof', 1, 'disp' )
        
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
    
    ok = ops.analyze(numSteps, timeInc)
    
    ops.wipe()
    #print('analysis done')
    
    # Post process
    result_list_sort = []
    for ii in range(len(damping_candi)):
        tmp_record_d = './Results_displ/DFree_%d_%f_%d.out' %(idx_gm, period, ii)
        displ = np.loadtxt(tmp_record_d)
        
        # times
        if ii ==0:
            result_list_sort.append(displ[:,0])
                  
        result_list_sort.append(displ[:,1])
        os.remove(tmp_record_d) # delete the file

    result_list_sort = np.asarray(result_list_sort)

    df0 = pd.DataFrame({'time': result_list_sort[0,:] })
    
    df = pd.DataFrame({'damping{}'.format(i): result_list_sort[i,:] 
                       for i in range(1,len(result_list_sort))})
    
    df1 = pd.concat([df0,df],  axis=1, ignore_index=False)
    
    df2 = df1.astype('float32', copy=False) # change data type
    
    return df2
    
#%% discretize SDOF systems (300 period steps, 50 damping coeff. steps)
dp = 0.015
period_candidate = np.arange(np.log(0.05)-6*dp,np.log(4)+dp,dp)
period_candi = np.exp(period_candidate)

del dp

ddm = 0.08
damping_candidate = np.arange(np.log(0.005),np.log(0.25)+ddm,ddm) # 0.5% to 20%
damping_candi = np.exp(damping_candidate)

del ddm
#%% Run dynamic anlaysis
for idx_gm in range(2): # NGA-West database Ground motions (2 ground motions are provided)

    GMfile = './Ground_motions/GM_%d' %(idx_gm)
    dtfile = './Ground_motions/time_%d' %(idx_gm)    

    for idx_pe in range(len(period_candi)):
                
        period_n = period_candi[idx_pe]
        stiff_n = (2*np.pi/period_n)**2 # (2*np.pi/period_n)**2
        
        df2 = run_dynamic(stiff_n, period_n, damping_candi, GMfile, dtfile, idx_gm)
            
        # Save storage
        tmp_direct_file = './Results_displ/GM_%d/period_%d.p' %(idx_gm,idx_pe)
        df2.to_pickle(tmp_direct_file)
        
          