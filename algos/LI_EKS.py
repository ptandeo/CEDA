import numpy as np
from algos.utils import RMSE, inv_svd
from numpy.linalg import inv
from tqdm import tqdm
from algos.EM_EKS import _EKS, _likelihood

def _adaptive_EKF(Nx, No, T, xb, B, Q_init, R_init, Yo, f, jacF, h, jacH, alpha, tau): # PIERRE (tau)
  Xa = np.zeros((Nx, T+1))
  Xf = np.zeros((Nx, T))
  Pa = np.zeros((Nx, Nx, T+1))
  Pf = np.zeros((Nx, Nx, T))
  F_all = np.zeros((Nx, Nx, T))
  H_all = np.zeros((No, Nx, T))
  
  K_all=np.zeros((Nx, No, T))
  d_all=np.zeros((No, T))
  Q_adapt=np.zeros((Nx, Nx, T+1))
  R_adapt=np.zeros((No, No, T+1))

  x = xb; Xa[:,0] = x
  P = B; Pa[:,:,0] = P
  Q_adapt[:,:,0]=Q_init
  R_adapt[:,:,0]=R_init
  Q = Q_init
  R = R_init
  
  for t in range(T):
        
    # Linearization
    F = jacF(x)
    H = jacH(x)
    F_all[:,:,t] = F
    H_all[:,:,t] = H
    
    # Forecast
    x = f(x)
    P = F.dot(P).dot(F.T) + Q
    P = .5*(P + P.T)
    Pf[:,:,t]=P; Xf[:,t]=x;
    
    # Update
    if not np.isnan(Yo[0,t]):
      d = Yo[:,t] - h(x)
      S = H.dot(P).dot(H.T) + R/alpha
      K = P.dot(H.T).dot(inv(S))
      P = (np.eye(Nx) - K.dot(H)).dot(P)
      x = x + K.dot(d)
      K_all[:,:,t]=K
      d_all[:,t]=d
      Pa[:,:,t+1]=P; Xa[:,t+1]=x; # T+1 ou t ???
      if t==0:
        Q_adapt[:,:,t+1]=Q_adapt[:,:,t]
        R_adapt[:,:,t+1]=R_adapt[:,:,t]
      else:
        Pe=inv(F_all[:,:,t]).dot(inv(H_all[:,:,t])).dot(np.outer(d_all[:,t],d_all[:,t-1])).dot(inv(H_all[:,:,t-1].T))+K_all[:,:,t-1].dot(np.outer(d_all[:,t-1],d_all[:,t-1])).dot(inv(H_all[:,:,t-1].T))
        Qe=Pe-F_all[:,:,t-1].dot(Pa[:,:,t-1]).dot(F_all[:,:,t-1].T)
        Re=np.outer(d_all[:,t-1],d_all[:,t-1])-H_all[:,:,t-1].dot(Pf[:,:,t-1]).dot(H_all[:,:,t-1].T)
        Q_adapt[:,:,t+1]=Q_adapt[:,:,t]+(Qe-Q_adapt[:,:,t])/tau
        R_adapt[:,:,t+1]=R_adapt[:,:,t]+(Re-R_adapt[:,:,t])/tau
    else:
      K_all[:,:,t]=K
      d_all[:,t]=d
      Pa[:,:,t+1]=P; Xa[:,t+1]=x; # T+1 ou t ???
      Q_adapt[:,:,t+1]=Q_adapt[:,:,t]
      R_adapt[:,:,t+1]=R_adapt[:,:,t]
    Q=Q_adapt[:,:,t+1]
    R=R_adapt[:,:,t+1]
    
  return Xa, Pa, Xf, Pf, H_all, Q_adapt, R_adapt

def LI_EKF(params):
  xb    = params['initial_background_state']
  B     = params['initial_background_covariance']
  Q     = params['initial_model_noise_covariance']
  R     = params['initial_observation_noise_covariance']
  f     = params['model_dynamics']
  jacF  = params['model_jacobian']
  h     = params['observation_operator']
  jacH  = params['observation_jacobian']
  Yo    = params['observations']
  Xt    = params['true_state']
  Nx    = params['state_size']
  No    = params['observation_size']
  T     = params['temporal_window_size']
  alpha = params['inflation_factor']
  tau   = params['adaptive_parameter']
  structQ = params['model_noise_covariance_structure']
  if structQ == 'const':
    baseQ = params['model_noise_covariance_matrix_template']
  else:
    baseQ = None    
  
  Xa, Pa, Xf, Pf, H, Q_adapt, R_adapt = _adaptive_EKF(Nx, No, T, xb, B, Q, R, Yo, f, jacF, h, jacH, alpha, tau)
  #Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H = _EKS(Nx, No, T, xb, B, np.nanmedian(Q_adapt,2), np.nanmedian(R_adapt,2), Yo, f, jacF, h, jacH, alpha) # Q_adapt[:,:,T] and R_adapt[:,:,T]

  loglik = _likelihood(Xf, Pf, Yo, R_adapt[:,:,T], H) # np.nanmedian(R_adapt,2)
  rmse = RMSE(Xa - Xt)

  res = {
          'filtered_states'                      : Xa,
          'LI_model_noise_covariance'            : Q_adapt,
          'LI_observation_noise_covariance'      : R_adapt, 
          'loglikelihood'                        : loglik,
          'RMSE'                                 : rmse, 
          'params'                               : params
        }
  return res

def LI_EKS(params):
  xb    = params['initial_background_state']
  B     = params['initial_background_covariance']
  Q     = params['initial_model_noise_covariance']
  R     = params['initial_observation_noise_covariance']
  f     = params['model_dynamics']
  jacF  = params['model_jacobian']
  h     = params['observation_operator']
  jacH  = params['observation_jacobian']
  Yo    = params['observations']
  nIter = params['nb_iterations']
  Xt    = params['true_state']
  Nx    = params['state_size']
  No    = params['observation_size']
  T     = params['temporal_window_size']
  alpha = params['inflation_factor']
  tau   = params['adaptive_parameter']
  structQ = params['model_noise_covariance_structure']
  if structQ == 'const':
    baseQ = params['model_noise_covariance_matrix_template']
  else:
    baseQ = None

  loglik = np.zeros(nIter)
  rmse = np.zeros(nIter)

  Q_all  = np.zeros(np.r_[Q.shape, nIter+1])
  R_all  = np.zeros(np.r_[R.shape, nIter+1])

  Xs_all = np.zeros([Nx, T+1, nIter])

  Q_all[:,:,0] = Q
  R_all[:,:,0] = R

  for k in tqdm(range(nIter)):
    
    Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H = _EKS(Nx, No, T, xb, B, Q, R, Yo, f, jacF, h, jacH, alpha)
    loglik[k] = _likelihood(Xf, Pf, Yo, R, H)
    rmse[k] = RMSE(Xs - Xt)
    
    # adaptive-EKF
    Xa, Pa, Xf, Pf, H, Q_adapt, R_adapt = _adaptive_EKF(Nx, No, T, xb, B, Q, R, Yo, f, jacF, h, jacH, alpha, tau)    
    Q = np.nanmedian(Q_adapt,2) # Q = Q_adapt[:,:,T]
    R = np.nanmedian(R_adapt,2) # R = R_adapt[:,:,T]
    
    Xs_all[...,k] = Xs
    Q_all[:,:,k+1] = Q
    R_all[:,:,k+1] = R

  res = {
          'smoothed_states'                                : Xs_all,
          'LI_model_noise_covariance'                      : Q_all,
          'LI_observation_noise_covariance'                : R_all,
          'loglikelihood'                                  : loglik,
          'RMSE'                                           : rmse, 
          'params'                                         : params
        }
  return res
