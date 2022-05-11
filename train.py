###############################################################################
# Do not write bytecode to maintain clean directories
import sys
sys.dont_write_bytecode = True

# Import required packages and core code
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import torch
from torchdiffeq import odeint_adjoint as odeint
#from torchdiffeq import odeint

import dynamics
from utils import standardize, un_standardize, find_mu_std, get_sigma
###############################################################################

###############################################################################
def prep_data(mean_i,
              cov_i,
              mean_f,
              cov_f,
              nx,
              nu,
              nw,
              path):
    
    '''
    Function that creates some of the data used to train the neural networks
    that approximate the hidden physics. Note that the "sigma" points come from
    the unscented transform function defined in "utils.py"
    
    mean_i --> mean at initial time, shape = (N, nx+nu, 1)
    
    cov_i --> covariance at initial time, shape = (N, nx, nx)
    
    cov_f --> covariance at final time, shape = (N, nx, nx)
    
    mean_f --> mean at final time, shape = (N, nx, 1)
    
    nx --> state dimension (int)
    
    nu --> exogenous input dimension (int)
    
    nw --> noise dimension (int)
    
    path --> path where data is saved (str)
    
    '''
    
    # Get total system size
    n = nx + nw
    
    # Get total number of samples
    N = len(mean_i)
    
    # Pre-allocate sigma point array
    sigma = np.zeros((N, n+nu, 2*n+1))
    
    # Pre-allocate weights array
    W = np.zeros((N, 2*n+1, 1))
    
    # Get sigma points
    for i in range(N):
        
        [sigma[i,:,:],
         W[i,:,:]] = get_sigma(mean_i[i,:,:], 
                               cov_i[i,:,:],
                               nx, 
                               nu, 
                               nw)
    
    # Create and shuffle indices of input data
    indices = np.array(range(N))
    random.shuffle(indices)
    
    # Input information for 80/10/10 train/val/test/split
    train_frac = 0.8
    val_test_frac = train_frac + (1-train_frac)/2
    
    # Get train/val/test indices
    train_indices = indices[0:int(train_frac*N)]
    val_indices = indices[int(train_frac*N):int(val_test_frac*N)]
    test_indices = indices[int(val_test_frac*N):]
    
    # Get train/val/test data
    sigma_train_raw = sigma[train_indices,:,:]
    mean_f_train_raw = mean_f[train_indices,:,:]
    cov_f_train = cov_f[train_indices,:,:]
    W_train = W[train_indices, :, :]
   
    sigma_val_raw = sigma[val_indices,:,:]
    mean_f_val_raw = mean_f[val_indices,:,:]
    cov_f_val = cov_f[val_indices,:,:]
    W_val = W[val_indices, :, :]
    
    sigma_test_raw = sigma[test_indices,:,:]
    mean_f_test_raw = mean_f[test_indices,:,:]
    cov_f_test = cov_f[test_indices,:,:]
    W_test = W[test_indices, :, :]
    
    # Get mean and standard deviation of training data for standardization
    # Note that rows of sigma point matrix that correspond to w variables have
    # pre-defined mu and variance of 0 and 1 respectively and therefore do not 
    # require standardization
    sigma_mu, sigma_std = find_mu_std(sigma_train_raw[:,0:nx+nu,0:1])
    
    # Standardize training/validation/test data and convert to float32
    mean_f_train = np.float32(standardize(mean_f_train_raw, 
                                          sigma_mu[0:nx], 
                                          sigma_std[0:nx]))
    
    mean_f_val = np.float32(standardize(mean_f_val_raw, 
                                        sigma_mu[0:nx], 
                                        sigma_std[0:nx]))
    
    mean_f_test = np.float32(standardize(mean_f_test_raw, 
                                         sigma_mu[0:nx], 
                                         sigma_std[0:nx]))
    
    # Pre-allocate (remember that w sigma points do not change)
    sigma_train = np.zeros(np.shape(sigma_train_raw))
    sigma_train[:, nx+nu:,:] = sigma_train_raw[:,nx+nu:,:]
    
    sigma_val = np.zeros(np.shape(sigma_val_raw))
    sigma_val[:, nx+nu:,:] = sigma_val_raw[:,nx+nu:,:]
    
    sigma_test = np.zeros(np.shape(sigma_test_raw))
    sigma_test[:, nx+nu:,:] = sigma_test_raw[:,nx+nu:,:]
    
    for j in range(2*n+1):
        sigma_train[:,0:nx+nu,j:j+1] = standardize(sigma_train_raw[:,0:nx+nu,j:j+1], 
                                               sigma_mu, 
                                               sigma_std)
        
        sigma_val[:,0:nx+nu,j:j+1] = standardize(sigma_val_raw[:,0:nx+nu,j:j+1], 
                                             sigma_mu, 
                                             sigma_std)
        
        sigma_test[:,0:nx+nu,j:j+1] = standardize(sigma_test_raw[:,0:nx+nu,j:j+1], 
                                              sigma_mu, 
                                              sigma_std)
        
    # Convert to float 32
    sigma_train = np.float32(sigma_train)
    sigma_val = np.float32(sigma_val)
    sigma_test = np.float32(sigma_test)
    sigma_mu = np.float32(sigma_mu)
    sigma_std = np.float32(sigma_std)
    
    W_train = np.float32(W_train)
    W_val = np.float32(W_val)
    W_test = np.float32(W_test)
    
    # Save data
    np.save(path + 'sigma_train.npy', sigma_train)
    np.save(path + 'sigma_val.npy', sigma_val)
    np.save(path + 'sigma_test.npy', sigma_test)
    np.save(path + 'mean_f_train.npy', mean_f_train)
    np.save(path + 'mean_f_val.npy', mean_f_val)
    np.save(path + 'mean_f_test.npy', mean_f_test)
    np.save(path + 'cov_f_train.npy', cov_f_train)
    np.save(path + 'cov_f_val.npy', cov_f_val)
    np.save(path + 'cov_f_test.npy', cov_f_test)
    np.save(path + 'W_train.npy', W_train)
    np.save(path + 'W_val.npy', W_val)
    np.save(path + 'W_test.npy', W_test)
    np.save(path + 'sigma_mu.npy', sigma_mu)
    np.save(path + 'sigma_std.npy', sigma_std)
    
    return [sigma_train, 
            sigma_val, 
            sigma_test, 
            mean_f_train,
            mean_f_val, 
            mean_f_test,
            cov_f_train,
            cov_f_val,
            cov_f_test,
            W_train,
            W_val,
            W_test,
            sigma_mu,
            sigma_std]
############################################################################### 

###############################################################################   
class NN(torch.nn.Module):
    
    '''
    Class for neural network where hidden layers have swish activation
    functions and the output layer has linear activation functions.
    
    input_dim --> number of input nodes (which has to match output dimension)
    (int)
    
    hidden_dim --> number of hidden nodes (int)
    
    num_hidden_layers --> number of hidden layers (int)
    
    '''

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 num_hidden_layers):
        
        # Initialize
        super(NN, self).__init__()
        
        # Create modules
        modules = []
        
        # Create first layer 
        modules.append(torch.nn.Linear(input_dim, hidden_dim))
        modules.append(torch.nn.SiLU())
        
        # Create hidden layers
        for _ in range(num_hidden_layers-1):
            modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
            modules.append(torch.nn.SiLU())
        
        # Create final layer
        modules.append(torch.nn.Linear(hidden_dim, input_dim))
        
        # Create net
        self.net = torch.nn.Sequential(*modules)
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, val=0)
        
    def forward(self, t, x):
        return self.net(x)
    
############################################################################### 
    
############################################################################### 
def train_g1(hidden_dim_g1, 
             num_hidden_layers_g1, 
             device,
             nx,
             nu,
             dt,
             num_epoch,
             sigma_train,
             sigma_val,
             mean_f_train,
             mean_f_val, 
             W_train,
             W_val,
             solver_g1,
             rtol_g1,
             atol_g1,
             path):
    
    '''
    Function that trains the neural network that approximates the drift
    coefficient, g1.
    
    hidden_dim_g1 --> number of hidden nodes in neural network (int)
    
    num_hidden_layers_g1 --> number of hidden layers in neural network (int)
    
    device --> 'cpu' or 'cuda'
    
    nx --> state dimension (int)
    
    nu --> exogenous input dimension (int)
    
    dt --> sampling time (int or float)
    
    num_epoch --> number of epochs uses to train neural network (int)
    
    sigma_train/val --> sigma points split into train/val sections
    
    mean_f_train/val --> predicted mean split into train/val sections
    
    W_train/val --> UT weights split into train/val sections
    
    solver_g1 --> type of ode solver, see
                  https://github.com/rtqichen/torchdiffeq for list of options
    
    rtol_g1/atol_g1 --> ode solver tolerances
    
    path --> path to which model is saved
    
    '''
    
    # Get input dimension
    input_dim = nx + nu
    
    # Instantiate model
    model = NN(input_dim, 
               hidden_dim_g1, 
               num_hidden_layers_g1)
    
    # Move model to GPU (if available)
    model = model.to(device)
    
    # Convert data to torch tensors and move to GPU (if GPU is available)
    sigma_train = torch.tensor(sigma_train).to(device)
    sigma_val = torch.tensor(sigma_val).to(device)
    
    mean_f_train = torch.tensor(mean_f_train).to(device)
    mean_f_val = torch.tensor(mean_f_val).to(device)
    
    W_train = torch.tensor(W_train).to(device)
    W_val = torch.tensor(W_val).to(device)
    
    # Get timespan for solving ODE
    t = torch.tensor(np.float32(np.array([0.0, float(dt)]))).to(device)
    
    # Set training parameters
    batch_size = 32
    lr = 0.001
    
    # Set loss function
    criterion = torch.nn.MSELoss()
    
    # Choose Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           'min')
    # Train neural network
    for epoch in range(num_epoch):
        
        # Record time per epoch
        start = time.time()
        
        # Put model in "train mode"
        model.train()
        
        # Randomize training data
        permutation = torch.randperm(sigma_train.size()[0])
        
        train_loss = 0
        num_batches = 0
        
        for i in range(0, sigma_train.size()[0], batch_size):
            
            # Zero the gradients
            optimizer.zero_grad()
        
            # Get batch indices and batches
            indices = permutation[i:i+batch_size]
            
            [W_batch,
             batch_in, 
             batch_target] = [W_train[indices],
                              sigma_train[indices], 
                              mean_f_train[indices]]
            
            # Get forward pass
            outputs = 0
            for k in range(0, np.shape(W_batch)[1]):
                outputs = outputs + W_batch[:,k,:]*odeint(model, 
                                                          batch_in[:,0:nx+nu,k], 
                                                          t, 
                                                          method = solver_g1, 
                                                          rtol=rtol_g1, 
                                                          atol=atol_g1)
            
            # Calculate loss
            loss = criterion(outputs[1,:,0:nx], 
                             batch_target[:,:,0])
            
            # Calculate gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Add to train loss
            train_loss += loss
            
            # Record number of tested batches
            num_batches +=1
        
        train_loss = train_loss/num_batches
        
        # Put model in "val" mode
        model.eval()
        
        # Calculate validation loss
        with torch.no_grad():
            
            # Get forward pass
            val_outputs = 0
            for j in range(0, np.shape(W_val)[1]):
                val_outputs = val_outputs + W_val[:,j,:]*odeint(model, 
                                                           sigma_val[:,0:nx+nu,j], 
                                                           t, 
                                                           method = solver_g1, 
                                                           rtol=rtol_g1, 
                                                           atol=atol_g1)
            
            # Calculate loss
            val_loss = criterion(val_outputs[1,:,0:nx], 
                                 mean_f_val[:,:,0])
            
            # Update scheduler
            scheduler.step(val_loss)
        
        # Record end time
        end= time.time()
        
        # Print losses and time per epoch
        print('Epoch {:04d} | Train Loss {:.9f} | Val Loss {:.9f}'.format(epoch, train_loss, val_loss))
        print (end-start)
        
    # Save model
    torch.save(model, path + 'g1.pt')
    
    return model
############################################################################### 
    
############################################################################### 
def NN_int(model, 
           device, 
           method,
           rtol,
           atol,
           states, 
           sigma_mu, 
           sigma_std, 
           nx,
           dt):
    
    '''
    Function that integrates neural network
    
    model --> trained neural network
    
    device --> 'cpu' or 'cuda'
    
    solver --> type of ode solver, see https://github.com/rtqichen/torchdiffeq
                for list of options
    
    rtol/atol --> ode solver tolerances
    
    states --> STANDARDIZED shape = (N, nx+nu)
    
    sigma_mu/std --> standardization scaling factors --> shape (nx+nu,)
    
    nx --> state dimension (int)
    
    dt --> measurement time (int or float)
    
    '''
    # Create placeholder
    t = torch.tensor(np.float32(np.array([0.0, float(dt)]))).to(device)
    
    # Put model in "evaluation" mode
    model.eval()
    
    # Create torch tensor out of states and move to device
    states = torch.tensor(np.float32(states)).to(device)
    
    # Put model in eval mode
    with torch.no_grad():
        output = odeint(model, 
                        states, 
                        t, 
                        method = method, 
                        rtol=rtol, 
                        atol=atol)
        output = output[1,:,0:nx]
        
    # Unstandardize
    output = un_standardize(output.cpu().detach().numpy().T, sigma_mu[0:nx,:], sigma_std[0:nx,:])
        
    # Unscale output and return
    return output.T
###############################################################################  
   
###############################################################################
def calc_g2_target(sigma, 
                   mean_f,
                   cov_f,
                   nx,
                   nu, 
                   nw,
                   dt,
                   sigma_mu,
                   sigma_std,
                   W,
                   model_g1,
                   device,
                   method_g1,
                   rtol_g1,
                   atol_g1):
       
    '''
    Function that creates target data used to train neural network that
    approximates g2. Essentially, the covariance prediction from UT can be
    re-arranged such that the g2(x_mean, u) can directly predict a target. This
    function creates that target.
    
    sigma_train/val/test --> STANDARDIZED sigma points
                             shape --> (N, nx+nu+nw, 2*n+1)
    
    mean_f_train/val/test--> STANDARDIZED means at final times
                             shape --> (N, nx, 1)
    
    cov_f_train/val/ test --> UN-STANDARDIZED covariances at final time,
                              shape --> (N, nx, nx)
    
    nx --> state dimension (int)
    
    nu --> exogenous input dimension (int)
    
    nw --> noise dimension (int)
    
    dt --> measurement time (int or float)
    
    sigma_mu/std --> sigma point standardization scaling factors 
                     shape --> (nx+nu,)
    
    W --> UT weights
    
    device --> 'cpu' or 'gpu'
    
    model_g1 --> trained neural network that represents the drift coefficient
    
    method_g1 --> type of ode solver. See
                  https://github.com/rtqichen/torchdiffeq for list of options
    
    rtol_g1/atol_g1 --> ode solver tolerances
    
    '''

    # Unstandardize mean_f
    mean_f = un_standardize(mean_f, sigma_mu[0:nx], sigma_std[0:nx])
    
    # Get un-standardized mean initial 
    mean_i = sigma[:, 0:nx, 0:1]
    
    mean_i = un_standardize(mean_i, sigma_mu[0:nx], sigma_std[0:nx])
    
    # Get total system size
    n = nx + nw
    
    # Get total number of train/val.test samples
    N = len(sigma)
    
    # Pre-allocate
    targets = np.zeros((N, nx))
    
    for i in range(N):
        
        # Get propagated sigma points (assuming g2=0)
        chi = np.zeros((nx, 2*n+1))
        for j in range(0, 2*n+1):
            chi[:,j] = NN_int(model_g1, 
                              device, 
                              method_g1,
                              rtol_g1,
                              atol_g1,
                              sigma[i, 0:nx+nu, j].reshape(1,nx+nu), 
                              sigma_mu, 
                              sigma_std, 
                              nx,
                              dt).flatten()
            
        # Subtract mean 
        chi_minus_mean = chi - mean_f[i,:]
        
        # "Zero out" entries that have non-zero g2 contributions
        for k in range(0, nw):
            chi_minus_mean[k,1+nx+k] = 0
            chi_minus_mean[k,1+nx+n+k] = 0
            
        # Get first part of variance contribution
        cov_pt_1 = np.matmul(W[i].T*(chi_minus_mean),
                             np.transpose((chi_minus_mean)))
            
        # Calculate remaining contribution to covariance
        cov_pt_2 = np.eye(nx)
        for l in range(0, nx):
            cov_pt_2[l,l] = W[i,2*nx+1,0]*(2*(chi[l,0])**2 + 2*mean_f[i,l,0]**2 - 4*chi[l,0]*mean_f[i,l,0])
            
        # Get target
        target = np.array(np.diag(cov_f[i,:,:] - cov_pt_1 - cov_pt_2))
        
        # Adjust target for isolated g2
        for m in range(0, nw):
           sigma_w = sigma[i, nx+nu+m, 1+nx+m]
           target[m] = target[m]/4/W[i,nx+1,0]/sigma_w**2 + mean_i[i,m,0]
           
        # Record
        targets[i,:] = target
           
    return targets.reshape(N, nx, 1)
###############################################################################  
        
###############################################################################     
def calc_g2_targets(sigma_train,
                    sigma_val,
                    sigma_test,
                    mean_f_train,
                    mean_f_val,
                    mean_f_test,
                    cov_f_train,
                    cov_f_val,
                    cov_f_test,
                    nx,
                    nu, 
                    nw,
                    dt,
                    sigma_mu,
                    sigma_std,
                    W_train,
                    W_val,
                    W_test,
                    model_g1,
                    device,
                    solver_g1,
                    rtol_g1,
                    atol_g1,
                    path):   
    
    '''
    Function that just runs "g2_target_calc" for training/validation/testing
    data sets and then standardizes the g2 target and saves the data. See 
    "g2_target_calc" for more information.
    
    '''
    
    g2_target_train_raw = calc_g2_target(sigma_train, 
                                         mean_f_train,
                                         cov_f_train,
                                         nx,
                                         nu, 
                                         nw,
                                         dt,
                                         sigma_mu,
                                         sigma_std,
                                         W_train,
                                         model_g1,
                                         device,
                                         solver_g1,
                                         rtol_g1,
                                         atol_g1)
    
    g2_target_val_raw = calc_g2_target(sigma_val, 
                                       mean_f_val,
                                       cov_f_val,
                                       nx,
                                       nu, 
                                       nw,
                                       dt,
                                       sigma_mu,
                                       sigma_std,
                                       W_val,
                                       model_g1,
                                       device,
                                       solver_g1,
                                       rtol_g1,
                                       atol_g1)
    
    g2_target_test_raw = calc_g2_target(sigma_test, 
                                        mean_f_test,
                                        cov_f_test,
                                        nx,
                                        nu, 
                                        nw,
                                        dt,
                                        sigma_mu,
                                        sigma_std,
                                        W_test,
                                        model_g1,
                                        device,
                                        solver_g1,
                                        rtol_g1,
                                        atol_g1)
    
    # Standardize training/validation/test data and convert to float32
    g2_target_train = np.float32(standardize(g2_target_train_raw, 
                                             sigma_mu[0:nx], 
                                             sigma_std[0:nx]))
    
    g2_target_val = np.float32(standardize(g2_target_val_raw, 
                                           sigma_mu[0:nx], 
                                           sigma_std[0:nx]))
    
    g2_target_test = np.float32(standardize(g2_target_test_raw, 
                                            sigma_mu[0:nx], 
                                            sigma_std[0:nx]))
    
    # Save
    np.save(path + 'g2_target_train.npy', g2_target_train)
    np.save(path + 'g2_target_val.npy', g2_target_val)
    np.save(path + 'g2_target_test.npy', g2_target_test)
    
    return [g2_target_train,
            g2_target_val,
            g2_target_test]
############################################################################### 
    
############################################################################### 
def train_g2(hidden_dim_g2, 
             num_hidden_layers_g2, 
             device,
             nx,
             nu,
             dt,
             num_epoch,
             sigma_train,
             sigma_val,
             g2_target_train,
             g2_target_val, 
             solver_g2,
             rtol_g2,
             atol_g2,
             path):
    
    '''
    Function that trains the neural network that approximates the diffusion
    coefficient, g2
    
    hidden_dim --> number of hidden nodes in neural network (int)
    
    num_hidden_layers --> number of hidden layers in neural network (int)
    
    device --> 'cpu' or 'cuda'
    
    nx --> state dimension (int)
    
    nu --> exogenous input dimension (int)
    
    dt --> measurement time (int or float)
    
    num_epoch --> number of epochs uses to train neural network (int)
    
    sigma_train/val --> sigma points split into train/val sections
    
    g2_target_train/val --> g2 target split into train/val sections
    
    solver_g2 --> type of ode solver. See
                  https://github.com/rtqichen/torchdiffeq for list of options
    
    rtol_g2/atol_g2 --> ode solver tolerances
    
    path --> path to save model in
    
    '''
    
    # Get input dimension
    input_dim = nx + nu
    
    # Instantiate model
    model = NN(input_dim, 
               hidden_dim_g2, 
               num_hidden_layers_g2)
    
    # Move model to GPU (if available)
    model = model.to(device)
    
    # Convert data to torch tensors and move to GPU (if GPU is available)
    sigma_train = torch.tensor(sigma_train).to(device)
    sigma_val = torch.tensor(sigma_val).to(device)
    
    g2_target_train = torch.tensor(g2_target_train).to(device)
    g2_target_val = torch.tensor(g2_target_val).to(device)
    
    # Get timespan for solving ODE
    t = torch.tensor(np.float32(np.array([0.0, float(dt)]))).to(device)
    
    # Set training parameters
    batch_size = 32
    lr = 0.001
    
    # Set loss function
    criterion = torch.nn.MSELoss()
    
    # Choose Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           'min')
    # Train neural network
    for epoch in range(num_epoch):
        
        # Record time per epoch
        start = time.time()
        
        # Put model in "train mode"
        model.train()
        
        # Randomize training data
        permutation = torch.randperm(sigma_train.size()[0])
        
        train_loss = 0
        num_batches = 0
        
        for i in range(0, sigma_train.size()[0], batch_size):
            
            # Zero the gradients
            optimizer.zero_grad()
        
            # Get batch indices and batches
            indices = permutation[i:i+batch_size]
            [batch_in, 
             batch_target] = [sigma_train[indices], 
                              g2_target_train[indices]]
            
            # Get forward pass
            outputs = odeint(model, 
                             batch_in[:,0:nx+nu,0], 
                             t, 
                             method = solver_g2, 
                             rtol=rtol_g2, 
                             atol=atol_g2)
          
            # Calculate loss
            loss = criterion(outputs[1,:,0:nx], 
                             batch_target[:,:,0])
            
            # Calculate gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Add to train loss
            train_loss += loss
            
            # Record number of tested batches
            num_batches +=1
        
        train_loss = train_loss/num_batches
        
        # Put model in "val" mode
        model.eval()
        
        # Calculate validation loss
        with torch.no_grad():
            
            # Get forward pass
            val_outputs = odeint(model, 
                                 sigma_val[:,0:nx+nu,0], 
                                 t, 
                                 method = solver_g2, 
                                 rtol=rtol_g2, 
                                 atol=atol_g2)
            
            # Calculate loss
            val_loss = criterion(val_outputs[1,:,0:nx], 
                                 g2_target_val[:,:,0])
            
            # Update scheduler
            scheduler.step(val_loss)
        
        # Record end time
        end= time.time()
        
        # Print losses and time per epoch
        print('Epoch {:04d} | Train Loss {:.9f} | Val Loss {:.9f}'.format(epoch, train_loss, val_loss))
        print (end-start)
        
    # Save model
    torch.save(model, path + 'g2.pt')
    
    return model
###############################################################################
           
############################################################################### 
def NN_eval(model, 
            device, 
            states, 
            sigma_mu, 
            sigma_std, 
            nx,
            dt):
    
    '''
    Function that evaluates neural network
    
    model --> trained neural network
    
    device --> 'cpu' or 'cuda'
    
    states --> UNSTANDARDIZED shape (N, nx+nu)
    
    sigma_mu/std --> standardization scaling factors --> shape (nx+nu,)
    
    nx --> state dimension (int)
    
    dt --> measurement time (int or float)
    
    '''
    # Create placeholder
    t = torch.tensor(np.float32(np.array([0.0, float(dt)]))).to(device)
    
    # Put model in "evaluation" mode
    model.eval()
    
    # Scale states
    states_scaled = standardize(states.T, sigma_mu, sigma_std).T
    
    # Create torch tensor out of states and move to device
    states_scaled = torch.tensor(np.float32(states_scaled)).to(device)
    
    # Put model in eval mode
    with torch.no_grad():
        output = model.forward(t, states_scaled)
        
    # Unscale output and return
    output = output[:,0:nx].cpu().detach().numpy().T*sigma_std[0:nx,:]
    output = output.T
    
    return output
###############################################################################

###############################################################################
def CSA_state_space(numx, numu):
    
    '''
    Function that creates x/u state space for visualizing reconstruction of
    CSA hidden physics
    
    numx --> number of points on x-axis
    numu --> number of points on u-axis
    
    '''
    
    # Minimum and maximum state values
    x_min = 0.1
    x_max = 5.0
    
    # Choose some number of state values for plotting. We choose 1000 in the
    # paper
    x = np.linspace(x_min, x_max, numx)
    
    # Minimum and maximum exogenous input values
    u_min = 0.6
    u_max = 3.9
    
    # Choose some number of inputs for plotting. We choose 8 in the paper
    u = np.linspace(u_min, u_max, numu)
    
    return [x,
            u]
###############################################################################   

###############################################################################
def reconstruct_CSA(hp,
                    model, 
                    device,
                    sigma_mu, 
                    sigma_std, 
                    nx,
                    nu,
                    dt,
                    path):
    
    '''
    Function that reconstructs hidden physics for CSA system
    
    hp --> 'g1' or 'g2' if drift or diffusion coefficient
    
    model --> trained neural network that approximates hidden physics
    
    device --> 'cuda' or 'cpu'
    
    nx --> state dimension (int)
    
    nu --> exogenous input dimension (int)
    
    dt --> measurement time (int or float)
    
    sigma_mu/std --> sigma point standardization scaling factors, 
                     shape = (nx+nu,1)
    
    path --> path where plot and rmse is saved
    
    '''
    
    # Get state space for plotting
    numu_plot = 8
    numx_plot = 1000
    [x_plot, 
     u_plot] = CSA_state_space(numx_plot, numu_plot)
    
    # Pre-allocate
    g_pred = np.zeros((len(u_plot), len(x_plot)))
    g_true = np.zeros((len(u_plot), len(x_plot)))
    
    # Create plots
    for j in range(0, len(u_plot)):
        
        for i in range(0, len(x_plot)):
            
            # Get state
            states = np.asarray([x_plot[i], u_plot[j], 0])
            
            # Get true prediction
            if hp == 'g1':
                _, g_true_ind, _ = dynamics.stoch_dyn_CSA(states)
                
            elif hp == 'g2':
                 _, _, g_true_ind = dynamics.stoch_dyn_CSA(states)
                
            g_true[j,i] = g_true_ind
      
            # Get NN prediction
            g_pred_ind = NN_eval(model, 
                                 device, 
                                 states[0:nx+nu].reshape(1,nx+nu), 
                                 sigma_mu, 
                                 sigma_std, 
                                 nx,
                                 dt)[0]
            
            g_pred[j,i] = g_pred_ind
            
    # Plot
    plt.clf()
    plt.figure(1)
    for j in range(numu_plot):
        if j == 0:
            plt.plot(x_plot, g_pred[j,:], color="red", linewidth=2, label="SPINODE")
            plt.plot(x_plot, g_true[j,:], color = "black", linestyle=":", linewidth=3, label="True")
        else:
            plt.plot(x_plot, g_pred[j,:], color="red", linewidth=2, label=None)
            plt.plot(x_plot, g_true[j,:], color = "black", linestyle=":", linewidth=3, label=None)
    plt.legend(fontsize=16)
    plt.xlabel("$x$", fontsize=20)
    if hp == 'g1':
        plt.ylabel("$g_1(x,u)$", fontsize=20)
    elif hp == 'g2':
        plt.ylabel("$g_2(x,u)$", fontsize=20)
    plt.tight_layout()
    if hp == 'g1':
         plt.savefig(path + "/g1_CSA.png")
    elif hp == 'g2':
        plt.savefig(path + "/g2_CSA.png")
        
    # Get rmse
    numu_rmse = 100
    numx_rmse = 100
    [x_rmse, 
     u_rmse] = CSA_state_space(numx_rmse, numu_rmse)
    
    # Pre-allocate
    g_pred = np.zeros((len(u_rmse), len(x_plot)))
    g_true = np.zeros((len(u_rmse), len(x_plot)))
    
    # Create plots
    for j in range(0, len(u_rmse)):
        
        for i in range(0, len(x_rmse)):
            
            # Get state
            states = np.asarray([x_rmse[i], u_rmse[j], 0])
            
            # Get true prediction
            if hp == 'g1':
                _, g_true_ind, _ = dynamics.stoch_dyn_CSA(states)
                
            elif hp == 'g2':
                 _, _, g_true_ind = dynamics.stoch_dyn_CSA(states)
                
            g_true[j,i] = g_true_ind
      
            # Get NN prediction
            g_pred_ind = NN_eval(model, 
                                 device, 
                                 states[0:nx+nu].reshape(1,nx+nu), 
                                 sigma_mu, 
                                 sigma_std, 
                                 nx,
                                 dt)[0]
            
            g_pred[j,i] = g_pred_ind
        
        
    # Calc rmse
    rmse = (np.mean((g_pred - g_true)**2))**0.5
        
    # Save rmse
    if hp == 'g1':
        np.save(path + "/rmse_g1.npy", rmse)
    elif hp == 'g2':
        np.save(path + "/rmse_g2.npy", rmse)
        
    return rmse
############################################################################### 



