import numpy as np
from algos.utils import RMSE, inv_svd
from numpy.linalg import inv
from tqdm import tqdm
from algos.EM_EKS import _EKS, _likelihood

def _bayesian_inference_EKF(Nx, No, T, xb, B, sigma2_Q_init, sigma2_R_init, Yo, f, jacF, h, jacH, tau):
  Xa = np.zeros((Nx, T+1))
  Xf = np.zeros((Nx, T))
  Pa = np.zeros((Nx, Nx, T+1))
  Pf = np.zeros((Nx, Nx, T))
  F_all = np.zeros((Nx, Nx, T))
  H_all = np.zeros((No, Nx, T))
  
  K_all=np.zeros((Nx, No, T))
  d_all=np.zeros((No, T))
  sigma2_Q_adapt=np.zeros((T+1))
  sigma2_R_adapt=np.zeros((T+1))

  x = xb; Xa[:,0] = x
  P = B; Pa[:,:,0] = P
  sigma2_Q_adapt[0]=sigma2_Q_init
  sigma2_R_adapt[0]=sigma2_R_init
  sigma2_Q = sigma2_Q_init
  sigma2_R = sigma2_R_init
  
  for t in range(T):
        
    # Linearization
    F = jacF(x)
    H = jacH(x)
    F_all[:,:,t] = F
    H_all[:,:,t] = H
    
    # Forecast
    x = f(x)
    P = F.dot(P).dot(F.T) + sigma2_Q*np.eye(Nx,Nx) # STEP 1 of Stroud et al. 2017
    P = .5*(P + P.T)
    Pf[:,:,t]=P; Xf[:,t]=x;
    
    # Update
    if not np.isnan(Yo[0,t]):
      d = Yo[:,t] - h(x)
      S = H.dot(P).dot(H.T) + sigma2_R*np.eye(No,No)
      
      lik=_likelihood(Xf[:,t], Pf[:,:,t], Yo[:,t], sigma2_R*np.eye(No,No), H) # STEP 2 of Stroud et al. 2017
      
      mean_theta, cov_theta # STEP 3 of Stroud et al. 2017
      
      #random.multivariate_normal(mean_theta,cov_theta,NB_PARTICULES) # STEP 4 of Stroud et al. 2017
        
        
    
      K = P.dot(H.T).dot(inv(S))
      P = (np.eye(Nx) - K.dot(H)).dot(P)
      x = x + K.dot(d)
      K_all[:,:,t]=K
      d_all[:,t]=d
      Pa[:,:,t+1]=P; Xa[:,t+1]=x; # T+1 ou t ???

      d_of=Yo[:,t]-h(f(Xa[:,t]))
      d_oa=Yo[:,t]-h(Xa[:,t+1])
      
      sigma2_R_tmp=1
      sigma2_R_adapt[t+1]=sigma2_R_adapt[t]+(sigma2_R_tmp-sigma2_R_adapt[t])/tau

      sigma2_Q_tmp=1
      sigma2_Q_adapt[t+1]=sigma2_Q_adapt[t]+(sigma2_Q_tmp-sigma2_Q_adapt[t])/tau
    
    else:
      K_all[:,:,t]=K
      d_all[:,t]=d
      Pa[:,:,t+1]=P; Xa[:,t+1]=x; # T+1 ou t ???
      sigma2_Q_adapt[t+1]=sigma2_Q_adapt[t]
      sigma2_R_adapt[t+1]=sigma2_R_adapt[t]
    sigma2_Q=sigma2_Q_adapt[t+1]
    sigma2_R=sigma2_R_adapt[t+1]
    
  return Xa, Pa, Xf, Pf, H_all, sigma2_Q_adapt, sigma2_R_adapt

def BI_EKF(params):
  xb       = params['initial_background_state']
  B        = params['initial_background_covariance']
  sigma2_Q = params['initial_model_noise_variance']
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
  
  Xa, Pa, Xf, Pf, H, sigma2_Q_adapt, sigma2_R_adapt = _bayesian_inference_EKF(Nx, No, T, xb, B, sigma2_Q, sigma2_R, Yo, f, jacF, h, jacH, tau)
  #Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H = _EKS(Nx, No, T, xb, B, Q_adapt[:,:,T], sigma2_R_adapt*np.eye(No,No), Yo, f, jacF, h, jacH, alpha)
  loglik = _likelihood(Xf, Pf, Yo, sigma2_R_adapt[T]*np.eye(No,No), H)
  rmse = RMSE(Xa - Xt)

  res = {
          'filtered_states'                      : Xa,
          #'smoothed_states'                      : Xs,
          'adaptive_model_noise_variance'        : sigma2_Q_adapt,
          'adaptive_observation_noise_variance'  : sigma2_R_adapt, 
          'loglikelihood'                        : loglik,
          'RMSE'                                 : rmse, 
          'params'                               : params
        }
  return res