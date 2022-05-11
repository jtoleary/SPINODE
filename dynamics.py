###############################################################################
# Import required packages
import numpy as np
###############################################################################

###############################################################################
def stoch_dyn_CSA(states):
    
    """
    Function that simulates stochastic colloidal self-assembly dynamics
      
    states --> [xk, uk, xkw], shape = (3,1) or (3,)
    
    xk --> system state (C6)
    
    uk --> exogenous input (electric field voltage)
    
    xkw --> Gaussian white noise ~ N(0,1)
    
    """
    # Sampling time (s)
    dt = 1
    
    # Distribute states
    xk = states[0]
    uk = states[1]
    xkw = states[2]
    
    # Get diffusion coefficient
    g2 = 0.0045*np.exp(-(xk-2.1-0.75*uk)**2)+0.0005
    
    # Get drift coefficient
    #    F/KT = 10*(x-2.1-0.75*u)**2
    dFdx = 20*(xk-2.1-0.75*uk)
    dg2dx = -2*(xk-2.1-0.75*uk)*0.0045*np.exp(-(xk-2.1-0.75*uk)**2)
    g1 = dg2dx-g2*dFdx
    
    # Predict forward dynamics
    xkp1 = xk + g1*dt + np.sqrt(2*g2*dt)*xkw
    
    return [np.asarray([xkp1]), 
            np.asarray([g1]), 
            np.asarray([g2])]
###############################################################################
    
###############################################################################
def stoch_dyn_LVE(states):
    
    '''
    Function that simulates stochastic competitive Lotka-Volterra dynamics
    with coexistence equilbirum
    
    states = [xk, yk, xkw, ykw], shape = (4,1) or (4,)
    
    xk, yk --> species populations
    
    xkw, ykw --> independent Guassian white noise processes, ~ N(0,1)
    
    '''
    
    # Sampling time (s)
    dt = 0.01
    
    # Distribute states
    xk = states[0]
    yk = states[1]
    xkw = states[2]
    ykw = states[3]
    
    # Enter parameters
    k1 = 0.4
    k2 = 0.5
    xeq = 0.75
    yeq = 0.625
    d1 = 0.5
    d2 = 0.5
    
    # Get drift coefficients
    g1x = xk*(1 - xk - k1*yk)
    g1y = yk*(1 - yk - k2*xk)
    
    # Get diffusion coefficients
    g2x = 1/2*(d1*xk*(yk-yeq))**2
    g2y = 1/2*(d2*yk*(xk-xeq))**2
    
    # Predict forward dynamics
    xkp1 = xk + g1x*dt + np.sqrt(2*g2x*dt)*xkw
    ykp1 = yk + g1y*dt + np.sqrt(2*g2y*dt)*ykw
    
    return [np.asarray([[xkp1], [ykp1]]), 
            np.asarray([[g1x], [g1y]]), 
            np.asarray([[g2x], [g2y]])]
###############################################################################

###############################################################################
def stoch_dyn_SIR(states):
    
    '''
    Function that simulates stochastic Susceptible-Infectious-Recovered (SIR)
    dynamics
    
    states = [sk, ik, rk, skw, ikw, rkw], shape = (6,1) or (6,)
    
    sk, ik, rk --> susceptible, infectious, recovered populations
    
    skw, ikw, rkw --> independent Guassian white noise processes, ~ N(0,1)
    
    '''
    
    # Sampling time (s)
    dt = 1
    
    # Distribute states
    sk = states[0]
    ik = states[1]
    rk = states[2]
    skw = states[3]
    ikw = states[4]
    rkw = states[5]
    
    # Enter parameters
    b = 1
    d = 0.1
    k = 0.2
    alpha = 0.5
    gamma = 0.01
    mu = 0.05
    h = 2
    delta = 0.01
    sigma_1 = 0.2
    sigma_2 = 0.2
    sigma_3 = 0.1
    
    # Get nonlinear incidence rate
    g = (k*sk**h*ik)/(sk**h+alpha*ik**h)
    
    # Get drift coefficients
    g1s = b-d*sk-g+gamma*rk
    g1i = g-(d+mu+delta)*ik
    g1r = mu*ik-(d+gamma)*rk
    
    # Get diffusion coefficients
    g2s = 1/2*(sigma_1*sk)**2
    g2i = 1/2*(sigma_2*ik)**2
    g2r = 1/2*(sigma_3*rk)**2
    
    # Predict forward dynamics
    skp1 = sk + g1s*dt + np.sqrt(2*g2s*dt)*skw
    ikp1 = ik + g1i*dt + np.sqrt(2*g2i*dt)*ikw
    rkp1 = rk + g1r*dt + np.sqrt(2*g2r*dt)*rkw
    
    return [np.asarray([skp1, ikp1, rkp1]), 
            np.asarray([g1s, g1i, g1r]), 
            np.asarray([g2s, g2i, g2r]),
            np.asarray([g]),
            np.asarray([b-d*sk+gamma*rk]),
            np.asarray([(d+mu+delta)*ik])]
###############################################################################
