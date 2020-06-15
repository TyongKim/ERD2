# Newmark's method: linear systems by Chopra 3rd edition pp.177
# m: mass
# k: stiffness
# xi: damping ratio
# GM: ground motion history
# dt_analy: time step
# gamma & beta are the parameter of Newmark's method
# dis: displacement response 
# vel: velocity response
# acc: acceleration response
def Newmark_TK(m, k, xi, GM, dt,dt_analy, gamma, beta):
    
    # Import a library
    import numpy as np
    
   
    temp_GM = GM
    num_data = len(temp_GM)
    tmp = np.arange(0,num_data)*dt
    tmp2 = np.arange(0,num_data*dt/dt_analy)*dt_analy
    GM = np.interp(tmp2,tmp,temp_GM)
    
    # Initial calculations
    c = 2*xi*np.sqrt(m*k)
    k_hat = k+gamma/beta/dt_analy*c+1/beta/dt_analy**2*m
    a = 1/beta/dt_analy*m+gamma/beta*c
    b = 1/2/beta*m+dt_analy*(gamma/2/beta-1)*c
    p = -m*GM
    
    # Assume initial acceleration,     
    results = np.zeros([len(GM),3])

    
    # We assume initial velocitiy and displacement are zero
    dis = 0
    vel = 0
    pi = 0      # Assume initial force is zero
    acc = (pi-c*vel-k*dis)/m

    
    # Calculations for each time step, i
    for i in range(len(GM)):
        dp = p[i]-pi
        dp_hat = dp + a*vel + b*acc
        
        ddis = dp_hat/k_hat
        dvel = gamma/beta/dt_analy*ddis - gamma/beta*vel +dt_analy*(1-gamma/2/beta)*acc
        dacc = 1/beta/(dt_analy*dt_analy)*ddis - 1/beta/dt_analy*vel -1/2/beta*acc
        
        dis = dis+ddis
        vel = vel+dvel
        acc = acc+dacc
        
        results[i,0] = dis
        results[i,1] = vel
        results[i,2] = acc + GM[i]
        pi = p[i]
        
    return results