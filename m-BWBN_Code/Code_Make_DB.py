"""
This script is used to construct a seismic demand database for 
various structural systems that exhibit degradation and pinching effects. 
The m-BWBN model is employed to generate hysteretic behaviors, and 
the Runge-Kutta method is used to run the dynamic analysis.

March 18, 2023

Written by Taeyong Kim at Ajou University.

Reference: Kim, T., Kwon, O., and Song, J. (2023) Deep learning-based seismic 
response prediction of hysteretic systems having degradation and pinching,
Earthquake Engineering and Structural Dynamics. 

"""

# Import libraries
import sqlite3
import numpy as np
from scipy.integrate import solve_ivp

#%% Construct database using Sqlite3, tables and queries
sddb_name = 'SDDB_mBWBN_ver1.0.db' # Name of the database

# Make database using SQLite3
con = sqlite3.connect(sddb_name) # connect to the database
cur = con.cursor()                           # instantiate a cursor obj

# Ground motion information table
# this table is related to the i-ERD net
GM_sql = """ CREATE TABLE Ground_motion (ID integer PRIMARY KEY,
                                         RSN integer NOT NULL,
                                         location text NOT NULL) """

cur.execute(GM_sql)

# Structural system table
# Parameters of the m-BWBN model are listed
SS_sql = """ CREATE TABLE Strcutral_system (ID integer PRIMARY KEY,
                                            \"Period(s)\" real NOT NULL,
                                            \"Stiffness(g/m)\" real NOT NULL,
                                            \"Yield_force(g)\" real NOT NULL,
                                            Post_yield_stiffness real NOT NULL,
                                            n real NOT NULL,
                                            deltaNu real NOT NULL,
                                            deltaEta real NOT NULL,
                                            zeta0 real NOT NULL,
                                            p real NOT NULL,
                                            q real NOT NULL,
                                            shi real NOT NULL,
                                            del_shi real NOT NULL,
                                            lambda real NOT NULL) """

cur.execute(SS_sql)

# Seismic demand database table
RA_sql =  """ CREATE TABLE Analysis_result (\"Displacement(m)\" real NOT NULL,
                                            \"Velocity(m/s)\" real NOT NULL,
                                            \"Acceleration(m/s^2)\" real NOT NULL,
                                            \"Residual_displ(m)\" real NOT NULL,
                                            \"Energy_input(Nm)\" real NOT NULL,
                                            \"Energy_viscous(Nm)\" real NOT NULL,
                                            \"Energy_elas_hys(Nm)\" real NOT NULL,
                                            \"Energy_plas_hys(Nm)\" real NOT NULL,
                                            \"Restoring_force(g)\" real NOT NULL,
                                            Ground_motion_id integer NOT NULL,
                                            Structural_system_id integer NOT NULL,
                                            FOREIGN KEY (Ground_motion_id) REFERENCES Ground_motion (ID),
                                            FOREIGN KEY (Structural_system_id) REFERENCES Strcutral_system (ID)) """

cur.execute(RA_sql)

#%% Define m-BWBN model   
# State-space model of the m-BWBN model
def m_BWBN_TK(t, y, gt, xgt, m, c, alpha, k0, Fy, n, deltaNu, deltaEta,
            pinch_q, pinch_p, pinch_zeta0, pinch_shi, pinch_delshi, pinch_lambda):

    gamma = 0.5
    beta = 0.5
    
    xgt2 = np.interp(t, gt, xgt) # external force
    
    y0, y1, y2, y3 = y # u, u_dot, z, epsilon
    
    # pinching effects
    zeta_1 = pinch_zeta0*(1-np.exp(-pinch_p*y3))
    zeta_2 = (pinch_shi+pinch_delshi*y3)*(pinch_lambda+zeta_1)
    
    
    z_u = ((1+deltaNu*y3)*(beta+gamma))**(-1/n)
    h = 1-zeta_1*np.exp(-(y2*np.sign(y1)-pinch_q*z_u)**2/zeta_2**2)    
    
    # First order derivative of the parameters
    # Left hand side of the state space model
    u_dot = y1
    u_dot_dot = -1/m*(c*y1 + (alpha*k0*y0+(1-alpha)*Fy*y2) + m*xgt2)
    z_dot = h/(1+deltaEta*y3)*(y1-(beta*np.abs(y1)*y2*np.abs(y2)**(n-1)+
                               gamma*y1*np.abs(y2)**n)*(1+deltaNu*y3))*k0/Fy
    e_dot = (1-alpha)*k0/Fy*y2*y1
    
    return [u_dot, u_dot_dot, z_dot, e_dot]     


#%% Discretize the parameters of m-BWBN model
# Please refer to table 1 of the reference

# Periods
str_period = np.array([0.05, 0.055, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 
                       0.23, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.5, 0.55, 0.60, 
                       0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0, 1.2, 1.5, 2.0, 
                       2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]) # Total of 40

# Stiffness
str_stiffness = 1/9.8/(str_period/(2*np.pi))**2 # Normalized stiffness by g

# Normalized yield strength
str_yield_force = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
                            1.0, 1.25, 1.5]) # 10

# Post yield stiffness ratio
str_post_yield = np.array([0.0, 0.05, 0.2, 0.5])  #4

# m-BWBN parameters
str_n = 3 # fix
str_del_Nu = np.array([0.0, 0.1, 0.36]) # 3
str_del_Eta = np.array([0.0, 0.1, 0.39]) # 3
str_pinch_q = 0.027 # fix
str_pinch_shi = 0.117 # fix
str_pinch_delshi = 0.003 # fix
str_pinchi = [[0, 0, 0.025],
              [0.58, 0.33, 0.012],
              [0.62, 0.22, 0.45],
              [0.65, 1.20, 0.015],
              [0.70, 1.38, 0.38],
              [0.80, 0.24, 0.019],
              [0.92, 0.34, 0.6],
              [0.87, 1.30, 0.025],
              [0.91, 1.36, 0.7]] # zeta0(0.7, 0.9), p (0.3,1.38), lambda (0.015, 0.4)

str_pinchi = np.asarray(str_pinchi)
g = 9.8#; % ground acceleration m/s2
m = 1 # mass
                                                       
# Make a list of all combinations
Index_structural = [] 
SDOF_R_ID = 0  
for iik in range(len(str_stiffness)):
    for iit in range(len(str_yield_force)):
        for iiy in range(len(str_post_yield)):
            for iikk in range(len(str_del_Nu)):
                for iitt in range(len(str_del_Eta)):
                    for iiyy in range(len(str_pinchi)):
                        Index_structural.append([iik,iit,iiy,iikk,iitt,iiyy,SDOF_R_ID])
                        
                        Results_SS = np.asarray([SDOF_R_ID, str_period[iik,],str_stiffness[iik], str_yield_force[iit],
                                                 str_post_yield[iiy], str_n, str_del_Nu[iikk], str_del_Eta[iitt], 
                                                 str_pinchi[iiyy,0], str_pinchi[iiyy,1], str_pinch_q,
                                                 str_pinch_shi, str_pinch_delshi, str_pinchi[iiyy,2]])
                        
                        # Save the structural information in the database
                        # Structural ID starts from 0
                        SS_sql = "INSERT INTO Strcutral_system (ID, \"Period(s)\", \"Stiffness(g/m)\", \"Yield_force(g)\", \
                                                                Post_yield_stiffness, n, deltaNu, deltaEta, \
                                                                zeta0, p, q, shi, del_shi, lambda) \
                                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"                
                        
                        cur.execute(SS_sql,Results_SS)
                         
                        SDOF_R_ID = SDOF_R_ID+1
        
Index_structural = np.asarray(Index_structural)      
#%% Structural analysis code

def loop_worker(jj):
    # for each index of the combination
    tmp_index_structural = Index_structural[jj,:]
    
    # Assign index of the parameter
    iik = tmp_index_structural[0]
    iit = tmp_index_structural[1]
    iiy = tmp_index_structural[2]
    iikk = tmp_index_structural[3]
    iitt = tmp_index_structural[4]
    iiyy = tmp_index_structural[5]
    SDOF_R_ID = float(tmp_index_structural[6])
    
    # Stiffness, yield force, post yield, and damping
    k0 = str_stiffness[iik]*g
    Fy = str_yield_force[iit]*g
    alpha = str_post_yield[iiy]
    c = 2*0.05*np.sqrt(m*k0) # 5% damping ratio is assumed
    
    # Other m-BWBN model parameters
    n = str_n
    deltaNu = str_del_Nu[iikk]
    deltaEta = str_del_Eta[iitt]
    pinch_zeta0 = str_pinchi[iiyy,0]
    pinch_p = str_pinchi[iiyy,1]
    pinch_q = str_pinch_q
    pinch_shi = str_pinch_shi
    pinch_delshi = str_pinch_delshi
    pinch_lambda = str_pinchi[iiyy,2]

    # Run analysis using Runge-Kutta
    sol = solve_ivp(lambda t, y:m_BWBN_TK(t, y, gt, xgt, m, c, alpha, k0, Fy, n, 
                                          deltaNu, deltaEta, pinch_q, pinch_p, 
                                          pinch_zeta0, pinch_shi, pinch_delshi, 
                                          pinch_lambda),
                   t_span=(0, gt[-1]), y0=[0,0,0,0], method="RK23" ,t_eval = gt, max_step=dt)

    # Post process
    solution_y = sol.y
    solution_acce = (-1/m*(c*solution_y[1,:] + (alpha*k0*solution_y[0,:]+
                           (1-alpha)*Fy*solution_y[2,:])) )
    
    # Responses (displacement, velocity, and acceleration)
    NodeDisp = solution_y.T[:,0]
    NodeVelo = solution_y.T[:,1]
    NodeAcce = solution_acce
    
    Force_restoring = alpha*k0*sol.y.T[:,0]+(1-alpha)*Fy*sol.y.T[:,2]
    Force_damping = c*NodeVelo              
        
    # Displacement increment to calcualte energy        
    #NodeDisp_Inc = np.zeros((num_step,))      
    #NodeDisp_Inc[1:-2,] = 0.5*(NodeDisp[2:-1,] - NodeDisp[0:-3,]) 
    
    # calculate energy parameters
    E_history_i  = np.zeros([num_step,])                    # Input energy history
    E_history_v  = np.zeros([num_step,])                    # Dissiated energy thorugh viscos damping
    E_history_he = np.zeros([num_step,])                    # Elastic hysteretic energy
    E_history_hp = np.zeros([num_step,])                    # Viscous hysteretic energy

    E_history_i  = -m*NodeAcce*NodeVelo*dt                 # Input energy: integ(mass*acce*disp)
    E_history_v  = Force_damping*NodeVelo*dt               # Dissipated energy (viscos damping)
    E_history_he = alpha*k0*sol.y.T[:,0]*NodeVelo*dt       # Dissipated energy (hysteretic damping)
    E_history_hp = (1-alpha)*Fy*sol.y.T[:,2]*NodeVelo*dt   # Dissipated energy (hysteretic damping)    
    
    E_history_i  = np.cumsum(E_history_i, axis=0)
    E_history_v  = np.cumsum(E_history_v, axis=0)  
    E_history_he = np.cumsum(E_history_he, axis=0)
    E_history_hp = np.cumsum(E_history_hp, axis=0)
    
    #Calculate recorded dataset
    max_acc = np.max(np.abs(NodeAcce))                      # acceleration, m/s^2
    max_vel = np.max(np.abs(NodeVelo))                      # velocity, m/s
    max_dis = np.max(np.abs(NodeDisp))                      # displacement, m
    res_dis = NodeDisp[-1]                                  # Residual displacement, m
    max_force_restoring = np.max(np.abs(Force_restoring))/g # restoring force, g
    max_E_i  = E_history_i[-1]                              # Total input Energy,m N*m
    max_E_v  = E_history_v[-1]                              # Total viscous Energy, N*m
    max_E_he = E_history_he[-1]                             # Total elastic hysteresis Energy, N*m
    max_E_hp = E_history_hp[-1]                             # Total plastic hysteresis Energy, N*m
    
    # Save the results    
    Results_RA = np.array([max_dis, max_vel, max_acc, res_dis, max_E_i, max_E_v, 
                           max_E_he, max_E_hp, max_force_restoring, ii, SDOF_R_ID])
    
    return Results_RA
    
#%% Save the value into database
# Query to save the structural responses
RA_sql = "INSERT INTO Analysis_result (\"Displacement(m)\", \"Velocity(m/s)\", \"Acceleration(m/s^2)\", \"Residual_displ(m)\", \
                                       \"Energy_input(Nm)\", \"Energy_viscous(Nm)\", \"Energy_elas_hys(Nm)\", \"Energy_plas_hys(Nm)\", \
                                       \"Restoring_force(g)\", Ground_motion_id, Structural_system_id) \
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)" 

# Query to save the ground motion information
# ID: index that used in this script
# RSN: Record sequence number that is matched with the flat file of the NGA-West database
# Location is the directory where the acceleration is saved in the local drive
GM_sql = "INSERT INTO Ground_motion (ID, RSN, location) \
                                     VALUES (?, ?, ?)"  

                                     
# Number of total ground motions
# Even though the author employ 1,499 ground motions from NGA West database
# due to the copyright issue, we only upload two aritificailly generated ground 
# motions.
# Users need to put their own ground motions
total_gm = 2 
for ii in range(0, total_gm):#len(GM_location)):
         
    # Load ground motions, should use user's ground motions
    tmp1 = './Ground_motions/GM_%d.txt' %(ii) # Acceleration in g
    tmp2 = './Ground_motions/dt_%d.txt' %(ii) # time step of the GM acceleration
    xgt = np.loadtxt(tmp1)
    dt = np.loadtxt(tmp2)
    
    xgt = xgt*g
    dt = float(dt)
    num_step = int(len(xgt))
    gt = np.arange(0,(num_step+2)*float(dt),float(dt))
    if len(gt) > num_step:
        gt = gt[0:num_step]
        
    # Save GM infromation in the database
    # Becasue we use the simulated GM, we set ID == RSN
    cur.execute(GM_sql, (int(ii), int(ii), tmp1))             
    
    Results_append = []
    for jj in range(0, 3): #len(Index_structural)): # For every strucutral system combination
        tmp_result = loop_worker(jj)
        Results_append.append(tmp_result)
        
    Results_append = np.asarray(Results_append)


    # Save the dataset to the database
    cur.executemany(RA_sql,Results_append)                
    con.commit()                                                                        
    

# Close database
con.close()
