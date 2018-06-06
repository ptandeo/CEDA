import numpy as np
from algos.utils import RMSE, inv_svd
from numpy.linalg import inv
from tqdm import tqdm
from algos.EM_EKS import _EKS, _likelihood

def _adaptive_covariance_inflation_EKF(Nx, No, T, xb, B, lambda_init, sigma2_R_init, Yo, f, jacF, h, jacH, tau):
  Xa = np.zeros((Nx, T+1))
  Xf = np.zeros((Nx, T))
  Pa = np.zeros((Nx, Nx, T+1))
  Pf = np.zeros((Nx, Nx, T))
  F_all = np.zeros((Nx, Nx, T))
  H_all = np.zeros((No, Nx, T))
  
  K_all=np.zeros((Nx, No, T))
  d_all=np.zeros((No, T))
  lambda_adapt=np.zeros((T+1))
  sigma2_R_adapt=np.zeros((T+1))

  x = xb; Xa[:,0] = x
  P = B; Pa[:,:,0] = P
  lambda_adapt[0]=lambda_init
  sigma2_R_adapt[0]=sigma2_R_init
  lambda_ = lambda_init
  sigma2_R = sigma2_R_init
  
  for t in range(T):
        
    # Linearization
    F = jacF(x)
    H = jacH(x)
    F_all[:,:,t] = F
    H_all[:,:,t] = H
    
    # Forecast
    x = f(x)
    P = lambda_*F.dot(P).dot(F.T) # + Q # no model perturbations but multiplicative inflation
    P = .5*(P + P.T)
    Pf[:,:,t]=P; Xf[:,t]=x;
    
    # Update
    if not np.isnan(Yo[0,t]):
      d = Yo[:,t] - h(x)
      S = H.dot(P).dot(H.T) + sigma2_R*np.eye(No,No)
      K = P.dot(H.T).dot(inv(S))
      P = (np.eye(Nx) - K.dot(H)).dot(P)
      x = x + K.dot(d)
      K_all[:,:,t]=K
      d_all[:,t]=d
      Pa[:,:,t+1]=P; Xa[:,t+1]=x; # T+1 ou t ???

      d_of=Yo[:,t]-h(f(Xa[:,t]))
      d_oa=Yo[:,t]-h(Xa[:,t+1])
      
      sigma2_R_tmp=(d_oa.T.dot(d_of))/No # Li et al. 2009, Eq. (6)
      sigma2_R_adapt[t+1]=sigma2_R_adapt[t]+(sigma2_R_tmp-sigma2_R_adapt[t])/tau

      #lambda_tmp=(d_of.T.dot(d_of)-sigma2_R_adapt[t+1])/(np.trace(H.dot(Pf[:,:,t]).dot(H).T)) # Li et al. 2009, Eq. (4)
      #lambda_tmp=(np.trace(np.multiply(d_of.dot(d_of.T),1/sigma2_R_adapt[t+1]*np.eye(No,No)))))/(np.trace(np.multiply(H.dot(Pf[:,:,t]).dot(H).T,1/sigma2_R_adapt[t+1]*np.eye(No,No)))) # Miyoshi et al. 2011, Eq. (8)
      #lambda_tmp=np.sqrt((d_of.T.dot(d_of)-sigma2_R_adapt[t+1])/np.trace(H.dot(Pf[:,:,t]).dot(H).T)) # Yin and Zhang 2015, Eq. (5)

      lambda_tmp=(d_of.T.dot(d_of)-sigma2_R_adapt[t+1]*No)/np.trace(H.dot(Pf[:,:,t]/lambda_).dot(H).T) # PIERRE (cf. demo review)
      lambda_adapt[t+1]=lambda_adapt[t]+(lambda_tmp-lambda_adapt[t])/tau
    
    else:
      K_all[:,:,t]=K
      d_all[:,t]=d
      Pa[:,:,t+1]=P; Xa[:,t+1]=x; # t+1 ou t ???
      lambda_adapt[t+1]=lambda_adapt[t]
      sigma2_R_adapt[t+1]=sigma2_R_adapt[t]
    lambda_=lambda_adapt[t+1]
    sigma2_R=sigma2_R_adapt[t+1]
    
  return Xa, Pa, Xf, Pf, H_all, lambda_adapt, sigma2_R_adapt

def CI_EKF(params):
  xb       = params['initial_background_state']
  B        = params['initial_background_covariance']
  lmbda    = params['initial_multiplicative_inflation']
  sigma2_R = params['initial_observation_noise_variance']
  f        = params['model_dynamics']
  jacF     = params['model_jacobian']
  h        = params['observation_operator']
  jacH     = params['observation_jacobian']
  Yo       = params['observations']
  Xt       = params['true_state']
  Nx       = params['state_size']
  No       = params['observation_size']
  T        = params['temporal_window_size']
  tau      = params['adaptive_parameter']
  
  Xa, Pa, Xf, Pf, H, lambda_adapt, sigma2_R_adapt = _adaptive_covariance_inflation_EKF(Nx, No, T, xb, B, lmbda, sigma2_R, Yo, f, jacF, h, jacH, tau)
  
  loglik = _likelihood(Xf, Pf, Yo, sigma2_R_adapt[T]*np.eye(No,No), H)
  rmse = RMSE(Xa - Xt)

  res = {
          'filtered_states'                      : Xa,
          'adaptive_multiplicative_inflation'    : lambda_adapt,
          'adaptive_observation_noise_variance'  : sigma2_R_adapt, 
          'loglikelihood'                        : loglik,
          'RMSE'                                 : rmse, 
          'params'                               : params
        }
  return res
