"""
Created on Thu Mar 31 15:19:50 2022

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% m-BWBN model taking into account pinching and strength/stiffness        %
% degradations                                                            %
% Please see the input parameter carefully, especially sitffness and      %
% period.                                                                 %
% Author: Taeyong Kim University of Toronto                               %
% tyong.kim@mail.utoronto.ca, Mar 31, 2022                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input parameters of the m-Bouc-Wen model                                %
%                                                                         %
% 1. alpha      Post-yield stiffness ratio alpha = k_y/k_e with k_y the   %
%               post yield stiffness and k_e is yield stiffness.          %
% 2. Fy         Yield force                                               %
% 3. Period     Period of structrual system --> needs to be changed to    %
%               stiffness --> (1/g/(T0/(2*pi))**2)*g                      %
% 3. ko         ko is the elasttic stiffness fk_0 = F_y/u_y where F_y is  %
%               yield stiffness and fu_y the yield displacement.          %
% 4. beta       Bouc-Wen model coefficient(=0.5, fixed).                  %
% 5. gamma      Bouc-Wen model coefficient(=0.5, fixed).                  %
% 6. n          Softening parameter. Controls the transition from linear  %
%               to non-linear range (as n increases the transition becomes%
%               sharper n is usually grater or equal to 1).               %
% 7. delNu      Strength degradation parameter. With delta_nu = 0 no      %
%               strength degradation is included in the model.            %
% 8. delEta     Stiffness degradation parameter. With delta_eta = 0 no    %
%               stiffness degradation is included in the model.           %
% 9. pin_xi     Measure of total slip (always less than 1)                %
% 10.pin_p      Pinching slope (0 < pin_p < 1.38)                         %
% 11.pin_q      Pinching initiation (0.01 < pin_p < 0.43)                 %
% 12.pin_shi    Pinching magnitude (0.1 < shi < 0.85)                     %
% 13.pin_delshi Pinching rate (0< del_shi < 0.09)                         %
% 14.pin_lambda Pinching severity (0.01 < lambda < 0.8)                   %
% Details are shown in the 'summary.pdf' wirtten by Taeyong Kim           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def mBWBN_quasi_cyclic_python(parameters, DispTdT_total):
    
    import numpy as np

    g = 9.8; #% ground acceleration (m/s2)
    
    alpha = parameters[0] #(1);
    Fy = parameters[1] #(2);
    T0 = parameters[2] #(3);
    k0 = (1/g/(T0/(2*np.pi))**2)*g;
    beta = parameters[3] #(4);
    gamma = parameters[4] #(5);
    n = parameters[5] #(6);
    deltaNu = parameters[6] #(7);
    deltaEta = parameters[7] #(8);
    pin_xi = parameters[8] #(9);  % <1
    pin_p = parameters[9] #(10);
    pin_q = parameters[10] #(11);
    pin_shi = parameters[11] #(12);
    pin_delshi = parameters[12] #(13);
    pin_lambda = parameters[13] #(14);

    # Basic parameters
    maxIter = 5000;
    tolerance = 1E-12;
    startPoint = 0.01;
    DispT = 0.0;
    e_old = 0.0;
    z_old = 0.0;
    
    
    # Start using BWBN model (Equations are from the appendix of 'summary.pdf')
    e_Total = []; z_Total = []; force_Total = [];
    for ii in range(len(DispTdT_total)):

        DispTdT = DispTdT_total[ii];
        deltaDisp = DispTdT - DispT;

        # learning rate
        lr = 1;         taeyong = 1;
        # Perform Newton-Rhapson
        count = 0;         count_total = 0;         z_new = 1.0; 
        z_new_p = startPoint; z_eval = startPoint;

        while (np.abs(z_new_p - z_new) > tolerance) and (count < maxIter):

            # Step 1 
            e_new = e_old + (1 - alpha)*deltaDisp*k0/Fy*z_eval;

            nu_new = 1 + deltaNu*e_new;
            eta_new = 1 + deltaEta*e_new; 
            
            if nu_new<0:
                lr = lr*0.1;          
                count_total = count_total+count;
                count = 0;
                z_new = 1.0; z_new_p = startPoint; z_eval = startPoint;
                taeyong=taeyong+1;
                
                e_new = e_old + (1 - alpha)*deltaDisp*k0/Fy*z_eval;
                nu_new = 1 + deltaNu*e_new;
                eta_new = 1 + deltaEta*e_new;    
            
            a_1 = beta*np.sign(deltaDisp*z_eval)+ gamma;
            a_2 = (1-np.abs(z_eval)**n*a_1*nu_new)/eta_new;

            # Pinching effect
            pin_zx = (1/((beta+gamma)*nu_new))**(1/n);
            ching_xi1 = pin_xi*(1-np.exp(-pin_p*e_new));
            ching_xi2 = (pin_shi+e_new*pin_delshi)*(pin_lambda+ching_xi1);
            ching_h1 = 1 - ching_xi1*np.exp(-(z_eval*np.sign(deltaDisp)-pin_q*pin_zx)**2/ching_xi2**2);

            fz_new = z_eval - z_old - ching_h1*a_2*deltaDisp*k0/Fy;

            # Step 2: evaluate the deriviative
            # Evaluate function derivatives with respect to z_eval for the Newton-Rhapson scheme
            e_new_ = (1 - alpha)*k0/Fy*deltaDisp;
            nu_new_ = deltaNu*e_new_;
            eta_new_ = deltaEta*e_new_;
            pin_zx_ = -nu_new_*(beta+gamma)/n*((beta+gamma)*nu_new)**(-(n+1)/n);

            a_2_ = ((-eta_new_*(1-np.abs(z_eval)**(n)*a_1*nu_new) 
                   -eta_new*(n*n*np.abs(z_eval)**(n-1)*a_1*nu_new*np.sign(z_eval) 
                             +np.abs(z_eval)**(n)*a_1*nu_new_))/eta_new**2);

            # Pinching effect
            ching_xi1_ = pin_xi*pin_p*np.exp(-pin_p*e_new)*e_new_;
            ching_xi2_ = (pin_shi*ching_xi1_ + pin_lambda*pin_delshi*e_new_ +
                        pin_delshi*e_new_*ching_xi1 + pin_delshi*e_new*ching_xi1_);

            ching_h1_a3 = -np.exp(-(z_eval*np.sign(deltaDisp)-pin_q*pin_zx)**2/ching_xi2**2);
            ching_h1_a4 = (2*ching_xi1*(z_eval*np.sign(deltaDisp)-pin_q*pin_zx)
                          *(np.sign(deltaDisp)-pin_q*pin_zx_)/ching_xi2**2);
            ching_h1_a5 = 2*ching_xi1*(z_eval*np.sign(deltaDisp)-pin_q*pin_zx)**2/ching_xi2**3;
            ching_h1_ = ching_h1_a3*(ching_xi1_ - ching_h1_a4 + ching_xi2_*ching_h1_a5);

            fz_new_ = 1 - (ching_h1_*a_2+ching_h1*a_2_)*deltaDisp*k0/Fy;

            # Step 3: Perform a new step
            z_new = z_eval - lr*fz_new/fz_new_;

            # Step 4: Update the root
            z_new_p = z_eval;
            z_eval = z_new;

            count = count + 1;

            # Warning if there is no convergence
            if count == maxIter:
                print("WARNING: Could not find the root z, after maxIter")
                lr = lr*0.1;          
                count_total = count_total+count;
                count = 0;
                z_new = 1.0; z_new_p = startPoint; z_eval = startPoint;
                taeyong=taeyong+1;

        # Compute restoring force.
        force = alpha*k0*DispTdT + (1 - alpha)*Fy*z_eval;

        DispT = DispTdT;
        e_old = e_new;
        z_old = z_eval;

        force_Total.append(force)
        z_Total.append(z_eval)
        e_Total.append(e_new)

    return force_Total
