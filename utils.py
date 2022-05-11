###############################################################################
# Import required packages
import numpy as np
from scipy.linalg import sqrtm, block_diag
###############################################################################

###############################################################################
def standardize(X, mu, std):

    '''
    Function that standardizes X. 
    
    X --> shape = (N, num_features) or (N, num_features, num_features)
    
    mu/std --> mean/standard deviation for standardization, 
               shape = (num_features,) or (num_features, num_features)
    
    '''
    return (X-mu)/std
###############################################################################

###############################################################################
def un_standardize(X, mu, std):
    
    '''
    Function that un-standardizes X. 
    
    X --> shape = (N, num_features) or (N, num_features, num_features)
    
    mu/std --> mean/standard deviation for standardization, 
               shape = (num_features,) or (num_features, num_features)
    
    '''
    
    return (X*std)+mu
###############################################################################

###############################################################################
def find_mu_std(X):
    
    '''
    Function that finds mean and standard deviation of X. 
    
    X -->  shape = (N, num_features, 1) or (N, num_features, num_features)
    
    '''
    # Pre-allocate
    std = np.zeros((np.shape(X)[1],np.shape(X)[2]))
    mu = np.zeros((np.shape(X)[1],np.shape(X)[2]))
    
    # Find mu, std
    for i in range(0, np.shape(X)[1]):
        for j in range(0, np.shape(X)[2]):
            std[i,j] = np.std(X[:,i,j])
            mu[i,j] = np.mean(X[:,i,j])

    return np.float32(mu), np.float32(std)
###############################################################################
    
###############################################################################
def get_sigma(mean, covariance, nx, nu, nw):
    
    '''
    Function that creates the sigma point matrix for unscented transform (UT).
    
    Note that we create sigma point matrices whose columns are in the following
    order: [x, u, w]^T, where x is the state, u is an exogenous input, and w 
    is the noise. 
    
    Note this method only takes the mean and covariance into account, 
    although several recently reported UT methods use higher moments
    
    Finally note that this method is slightly simpler than the UT
    method shown in O'Leary et al., 2022. But in this case study, both methods
    perform similarly and this is easier to read.
    
    nx --> state dimension (int)
    
    nu --> exogenous input dimension (int)
    
    nw --> noise dimension (int) (assumed to be zero mean and unit variance)
    
    mean --> shape = (nx+nu, 1)
    
    covariance --> shape = (nx,nx)
         
    '''
    
    # Get total system size
    n = nx + nw
    
    # Choose scaling factor
    lam = 0
    
    # Get weights
    W = np.zeros((2*n+1,1))
    W[0] = lam/(n+lam)
    W[1:2*n+1,:] = 1/(2*(n+lam))*np.ones((2*n,1))
    
    # Normalize weights
    W = W/np.sum(W)
    
    # Combine covariances of states and process noise variables into n x n matrix
    SS = block_diag(covariance, np.identity(nw))
    
    # Take square root of this matrix (multiplied by (n+lam))
    S = sqrtm((n+lam)*SS)
    
    # Conatenate state mean with means of process noise variables
    concat_mean = np.concatenate((mean[0:nx,:], np.zeros((nw,1))), axis=0)
    
    # Initialize matrix of sigma points with mean values for every entry
    sigma = concat_mean*np.ones((n, 2*n+1))
    
    # Adjust sigma points based on covariance
    count = 0
    for i in range(1, n+1):
        sigma[:,i] = sigma[:,i] + S[:,count]
        count = count + 1
    
    count = 0
    for j in range(n+1,2*n+1):
        sigma[:,j] = sigma[:,j] - S[:,count]
        count = count + 1
    
    # Insert row(s) for exogenous input(s) (if exogenous inputs exist)
    if nu > 0:
        for i in range(0, nu):
            sigma = np.insert(sigma, nx+i, mean[nx+i,0]*np.ones(2*n+1), 0)

    return [np.float32(sigma), 
            np.float32(W)]
###############################################################################
