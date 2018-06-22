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
    P = lambda_*np.array([F]).dot(P).dot(F.T) # + Q # no model perturbations but multiplicative inflation ### MODIFS PIERRE: array([F])
    P = .5*(P + P.T)
    P = np.array([P])
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
      Pa[:,:,t+1]=P; Xa[:,t+1]=x;

      d_of=Yo[:,t]-h(f(Xa[:,t]))
      d_oa=Yo[:,t]-h(Xa[:,t+1])
      
      # Li et al. 2009, Eq. (6)
      sigma2_R_tmp=(d_oa.T.dot(d_of))/No
      sigma2_R_adapt[t+1]=sigma2_R_adapt[t]+(sigma2_R_tmp-sigma2_R_adapt[t])/tau

      # Li et al. 2009, Eq. (4)      
      #lambda_tmp=(d_of.T.dot(d_of)-sigma2_R_adapt[t+1])/(np.trace(H.dot(Pf[:,:,t]).dot(H).T))
      
      # Miyoshi et al. 2011, Eq. (8)
      #lambda_tmp=(np.trace(np.multiply(d_of.dot(d_of.T),1/sigma2_R_adapt[t+1]*np.eye(No,No))))/(np.trace(np.multiply(H.dot(Pf[:,:,t]).dot(H).T,1/sigma2_R_adapt[t+1]*np.eye(No,No))))
      
      # Yin and Zhang 2015, Eq. (5)
      #lambda_tmp=np.sqrt((d_of.T.dot(d_of)-sigma2_R_adapt[t+1])/np.trace(H.dot(Pf[:,:,t]).dot(H).T)) 

      # Tandeo version (see demo in the CEDA review paper)
      lambda_tmp=(d_of.T.dot(d_of)-sigma2_R_adapt[t+1]*No)/np.trace(H.dot(Pf[:,:,t]/lambda_).dot(H).T)
      lambda_adapt[t+1]=lambda_adapt[t]+(lambda_tmp-lambda_adapt[t])/tau
    
    else:
      K_all[:,:,t]=K
      d_all[:,t]=d
      Pa[:,:,t+1]=P; Xa[:,t+1]=x;
      lambda_adapt[t+1]=lambda_adapt[t]
      sigma2_R_adapt[t+1]=sigma2_R_adapt[t]

    lambda_=lambda_adapt[t+1]
    sigma2_R=sigma2_R_adapt[t+1]
    
  return Xa, Pa, Xf, Pf, H_all, lambda_adapt, sigma2_R_adapt, F_all, K_all

def _adaptive_covariance_inflation_EKS(Nx, No, T, xb, B, lambda_init, sigma2_R_init, Yo, f, jacF, h, jacH, tau):
  Xa, Pa, Xf, Pf, H, lambda_adapt, sigma2_R_adapt, F_all, K_all = _adaptive_covariance_inflation_EKF(Nx, No, T, xb, B, lambda_init, sigma2_R_init, Yo, f, jacF, h, jacH, tau)

  F = F_all
  Kf_final = K_all[:,:,T-1]

  Xs = np.zeros((Nx, T+1))
  Ps = np.zeros((Nx, Nx, T+1))
  K_all = np.zeros((Nx, Nx, T))

  x = Xa[:,-1]; Xs[:,-1] = x
  P = Pa[:,:,-1]; Ps[:,:,-1] = P

  # PIERRE: pb for small t because no perturvation on xf thus det(Pf) close to 0
  #Pf[:,:,1] = Pf[:,:,2]
  #Pf[:,:,0] = Pf[:,:,2]

  for t in range(T-1, -1, -1):

    K = Pa[:,:,t].dot(F[:,:,t].T).dot(inv(Pf[:,:,t])) # PIERRE: inv_svd() OR inv()???
    x = Xa[:,t] + K.dot(x - Xf[:,t])
    P = Pa[:,:,t] - K.dot(Pf[:,:,t] - P).dot(K.T)

    Ps[:,:,t]=P; Xs[:,t]=x; K_all[:,:,t]=K

  Ps_lag = np.zeros((Nx, Nx, T))
  Ps_lag[:,:,-1] = ((np.eye(Nx)-Kf_final.dot(H[:,:,-1]))
                  .dot(F[:,:,-1]).dot(Pa[:,:,-2]))
  for t in range(T-2, -1, -1):
    Ps_lag[:,:,t] += Pa[:,:,t+1].dot(K_all[:,:,t].T)
    Ps_lag[:,:,t] += (K_all[:,:,t+1]
                   .dot(Ps_lag[:,:,t+1] - F[:,:,t+1].dot(Pa[:,:,t+1]))
                   .dot(K_all[:,:,t].T))

  return Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H

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
  
  Xa, Pa, Xf, Pf, H, lambda_adapt, sigma2_R_adapt, F_all, K_all = _adaptive_covariance_inflation_EKF(Nx, No, T, xb, B, lmbda, sigma2_R, Yo, f, jacF, h, jacH, tau)
  
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

def CI_EKS(params):
  xb       = params['initial_background_state']
  B        = params['initial_background_covariance']
  lmbda    = params['initial_multiplicative_inflation']
  sigma2_R = params['initial_observation_noise_variance']
  f        = params['model_dynamics']
  jacF     = params['model_jacobian']
  h        = params['observation_operator']
  jacH     = params['observation_jacobian']
  Yo       = params['observations']
  nIter    = params['nb_iterations']
  Xt       = params['true_state']
  Nx       = params['state_size']
  No       = params['observation_size']
  T        = params['temporal_window_size']
  tau      = params['adaptive_parameter']

  loglik = np.zeros(nIter)
  rmse = np.zeros(nIter)

  lmbda_all  = np.zeros(np.r_[nIter+1])
  sigma2_R_all  = np.zeros(np.r_[nIter+1])

  Xs_all = np.zeros([Nx, T+1, nIter])

  lmbda_all[0] = lmbda
  sigma2_R_all[0] = sigma2_R

  for k in tqdm(range(nIter)):
    
    # adaptive-covariance-inflation-EKS
    Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H = _adaptive_covariance_inflation_EKS(Nx, No, T, xb, B, lmbda, sigma2_R, Yo, f, jacF, h, jacH, tau)
    loglik[k] = _likelihood(Xf, Pf, Yo, sigma2_R*np.eye(No,No), H)
    rmse[k] = RMSE(Xs - Xt)
    
    # adaptive background state
    xb = Xs[:,0]
    B = Ps[:,:,0]

    # adaptive-covariance-inflation-EKF
    Xa, Pa, Xf, Pf, H, lmbda_adapt, sigma2_R_adapt, F_all, K_all = _adaptive_covariance_inflation_EKF(Nx, No, T, xb, B, lmbda, sigma2_R, Yo, f, jacF, h, jacH, tau)
    lmbda = np.nanmedian(lmbda_adapt) # lmbda = lmbda_adapt[T]
    sigma2_R = np.nanmedian(sigma2_R_adapt) # sigma2_R = sigma2_R_adapt[T]
    
    Xs_all[...,k] = Xs
    lmbda_all[k+1] = lmbda
    sigma2_R_all[k+1] = sigma2_R

  res = {
          'smoothed_states'                      : Xs_all, # PIERRE: WHY DO WE KEEP ALL THE STATES (KEEP ONLY THE LAST ONE?)
          'adaptive_multiplicative_inflation'    : lmbda_all,
          'adaptive_observation_noise_variance'  : sigma2_R_all, 
          'loglikelihood'                        : loglik,
          'RMSE'                                 : rmse, 
          'params'                               : params
        }

  return res
