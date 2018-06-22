import numpy as np
from algos.utils import RMSE, inv_svd, cov_prob
from numpy.linalg import inv
from tqdm import tqdm

def _likelihood(Xf, Pf, Yo, R, H):
  T = Xf.shape[1]
  l = 0
  for t in range(T):
    if not np.isnan(Yo[0,t]):
      sig = H[:,:,t].dot(Pf[:,:,t]).dot(H[:,:,t].T) + R
      innov = Yo[:,t] - H[:,:,t].dot(Xf[:,t])
      
      #l -= .5 * np.log(np.linalg.det(sig))
    
      sign, l_tmp = np.linalg.slogdet(sig)
      l -= .5 * sign * l_tmp
              
      #try:
      #  import warnings
      #  warnings.filterwarnings('error')
      #  l -= .5 * np.log(np.linalg.det(sig))
      #except:
      #  import pdb; pdb.set_trace()

      l -= .5 * innov.T.dot(np.linalg.solve(sig, innov))
  return l

def _EKF(Nx, No, T, xb, B, Q, R, Yo, f, jacF, h, jacH, alpha):
  Xa = np.zeros((Nx, T+1))
  Xf = np.zeros((Nx, T))
  Pa = np.zeros((Nx, Nx, T+1))
  Pf = np.zeros((Nx, Nx, T))
  F_all = np.zeros((Nx, Nx, T))
  H_all = np.zeros((No, Nx, T))
  Kf_all = np.zeros((Nx, No, T))

  x = xb; Xa[:,0] = x
  P = B; Pa[:,:,0] = P
  for t in range(T):
    # Forecast
    F = jacF(x)
    x = f(x)
    P = F.dot(P).dot(F.T) + Q
    P = .5*(P + P.T)

    Pf[:,:,t]=P; Xf[:,t]=x; F_all[:,:,t]=F

    # Update
    if not np.isnan(Yo[0,t]):
      H = jacH(x)
      d = Yo[:,t] - h(x)
      S = H.dot(P).dot(H.T) + R/alpha
      K = P.dot(H.T).dot(inv(S))
      P = (np.eye(Nx) - K.dot(H)).dot(P)
      x = x + K.dot(d)

    Pa[:,:,t+1]=P; Xa[:,t+1]=x; H_all[:,:,t]=H; Kf_all[:,:,t]=K

  return Xa, Pa, Xf, Pf, F_all, H_all, Kf_all

def _EKS(Nx, No, T, xb, B, Q, R, Yo, f, jacF, h, jacH, alpha):
  Xa, Pa, Xf, Pf, F, H, Kf = _EKF(Nx, No, T, xb, B, Q, R, Yo, f,
                                         jacF, h, jacH, alpha)

  Xs = np.zeros((Nx, T+1))
  Ps = np.zeros((Nx, Nx, T+1))
  K_all = np.zeros((Nx, Nx, T))

  x = Xa[:,-1]; Xs[:,-1] = x
  P = Pa[:,:,-1]; Ps[:,:,-1] = P
  for t in range(T-1, -1, -1):
    K = Pa[:,:,t].dot(F[:,:,t].T).dot(inv(Pf[:,:,t]))
    x = Xa[:,t] + K.dot(x - Xf[:,t])
    P = Pa[:,:,t] - K.dot(Pf[:,:,t] - P).dot(K.T) 

    Ps[:,:,t]=P; Xs[:,t]=x; K_all[:,:,t]=K

  # Dreano et al. 2017, Eq. (30)
  #Ps_lag = np.zeros((Nx, Nx, T))
  #Ps_lag[:,:,-1] = ((np.eye(Nx)-Kf[:,:,-1].dot(H[:,:,-1])).dot(F[:,:,-1]).dot(Pa[:,:,-2]))
  #for t in range(T-2, -1, -1):
  #  Ps_lag[:,:,t] += Pa[:,:,t+1].dot(K_all[:,:,t].T)
  #  Ps_lag[:,:,t] += (K_all[:,:,t+1]
  #                 .dot(Ps_lag[:,:,t+1] - F[:,:,t+1].dot(Pa[:,:,t+1]))
  #                 .dot(K_all[:,:,t].T))

  # Tandeo PhD dissertation, Eq. (2.22), seems to be equivalent to Dreano formula
  #Ps_lag = np.zeros((Nx, Nx, T))
  #Ps_lag[:,:,-1] = ((np.eye(Nx)-Kf[:,:,-1].dot(H[:,:,-1])).dot(F[:,:,-1]).dot(Pa[:,:,-2]))
  #for t in range(T-2, -1, -1):
  #  Sigma = (np.eye(Nx) - Kf[:,:,t].dot(H[:,:,t])).dot(F[:,:,t]).dot(Pa[:,:,t-1])
  #  Ps_lag[:,:,t] = Sigma + (Ps[:,:,t] - Pa[:,:,t]).dot(inv(Pa[:,:,t])).dot(Sigma)

  # pykalman
  Ps_lag = np.zeros((Nx, Nx, T))
  #Ps_lag[:,:,-1] = ((np.eye(Nx)-Kf[:,:,-1].dot(H[:,:,-1])).dot(F[:,:,-1]).dot(Pa[:,:,-2]))
  for t in range(1,T):
    Ps_lag[:,:,t] = Ps[:,:,t].dot(K_all[:,:,t-1].T)

  return Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H

def _maximize(Xs, Ps, Ps_lag, Yo, h, jacH, f, jacF, structQ, baseQ=None):
  No = Yo.shape[0]  
  T = Yo.shape[1]
  Nx = Xs.shape[0]

  xb = Xs[:,0]
  B = Ps[:,:,0]
  R = 0
  nobs = 0
  sumSig = 0

  # Dreano et al. 2017, Eq. (34)
  for t in range(T):
    if not np.isnan(Yo[0,t]):
      nobs += 1
      H = jacH(Xs[:,t+1])
      R += np.outer(Yo[:,t] - h(Xs[:,t+1]), Yo[:,t] - h(Xs[:,t+1]))
      R += H.dot(Ps[:,:,t+1]).dot(H.T)
  R = .5*(R + R.T)
  R /= nobs

  # for Shumway 1982
  mat_A = 0
  mat_B = 0
  mat_C = 0

  for t in range(T):

    # Dreano et al. 2017, Eq. (33)
    F = jacF(Xs[:,t+1])
    sumSig += Ps[:,:,t+1]
    sumSig += np.outer(Xs[:,t+1]-f(Xs[:,t]), Xs[:,t+1]-f(Xs[:,t])) # CAUTION: error in Dreano equations
    sumSig += F.dot(Ps[:,:,t]).dot(F.T)
    sumSig -= Ps_lag[:,:,t].dot(F.T) + F.dot(Ps_lag[:,:,t].T) # CAUTION: transpose at the end (error in Dreano equations)
    sumSig = .5*(sumSig + sumSig.T)

    # Shumway 1982, Eq. (13), not working
    #mat_A += Ps[:,:,t] + np.outer(Xs[:,t], Xs[:,t])
    #mat_B += Ps_lag[:,:,t] + np.outer(Xs[:,t+1], Xs[:,t])
    #mat_C += Ps[:,:,t+1] + np.outer(Xs[:,t+1], Xs[:,t+1])
  #sumSig = mat_C - mat_B.dot(inv(mat_A)).dot(mat_B.T) # CAUTION: only for Shumway solution

  if structQ == 'full':
    Q = sumSig/T
  elif structQ == 'diag':
    Q = np.diag(np.diag(sumSig))/T
  elif structQ == 'const':
    alpha = np.trace(inv_svd(baseQ).dot(sumSig)) / (T*Nx)
    Q = alpha*baseQ
    beta = np.trace(inv_svd(baseQ).dot(R)) / (No) ### MODIF PIERRE ###
    R = beta*baseQ ### MODIF PIERRE ###
 
  return xb, B, Q, R

def EKS(params):
  Yo = params['observations']
  xb = params['background_state']
  B  = params['background_covariance']
  Q  = params['model_noise_covariance']
  R  = params['observation_noise_covariance']
  f  = params['model_dynamics']
  jacF = params['model_jacobian']
  h  = params['observation_operator']
  jacH = params['observation_jacobian']
  Nx = params['state_size']
  No = params['observation_size']
  T  = params['temporal_window_size']
  Xt = params['true_state']
  alpha = params['inflation_factor']

  Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H = _EKS(Nx, No, T, xb, B, Q, R, Yo, f,
                                           jacF, h, jacH, alpha)
  l = _likelihood(Xf, Pf, Yo, R, H)
  cov_p = cov_prob(Xs, Ps, Xt)
  
  res = {
          'smoothed_states'            : Xs,
          'smoothed_covariances'       : Ps,
          'smoothed_lagged_covariances': Ps_lag,
          'analysis_states'            : Xa,
          'analysis_covariance'        : Pa,
          'forecast_states'            : Xf,
          'forecast_covariance'        : Pf,
          'RMSE'                       : RMSE(Xs - Xt),
          'params'                     : params,
          'loglikelihood'              : l,
          'cov_prob'                   : cov_p
        }
  return res

def EM_EKS(params):
  xb    = params['initial_background_state']
  B     = params['initial_background_covariance']
  Q     = params['initial_model_noise_covariance']
  R     = params['initial_observation_noise_covariance']
  f     = params['model_dynamics']
  jacF  = params['model_jacobian']
  h     = params['observation_operator']
  jacH  = params['observation_jacobian']
  Yo    = params['observations']
  nIter = params['nb_EM_iterations']
  Xt    = params['true_state']
  Nx    = params['state_size']
  No    = params['observation_size']
  T     = params['temporal_window_size']
  alpha = params['inflation_factor']
  estimateQ  = params['is_model_noise_covariance_estimated']
  estimateR  = params['is_observation_noise_covariance_estimated']
  estimateX0 = params['is_background_estimated']
  structQ = params['model_noise_covariance_structure']
  if structQ == 'const':
    baseQ = params['model_noise_covariance_matrix_template']
  else:
    baseQ = None

  loglik = np.zeros(nIter)
  rmse_em = np.zeros(nIter)
  cov_prob_em = np.zeros(nIter)

  Q_all  = np.zeros(np.r_[Q.shape, nIter+1])
  R_all  = np.zeros(np.r_[R.shape, nIter+1])
  B_all  = np.zeros(np.r_[B.shape, nIter+1])
  xb_all = np.zeros(np.r_[xb.shape, nIter+1])

  Xs_all = np.zeros([Nx, T+1, nIter])

  Q_all[:,:,0] = Q
  R_all[:,:,0] = R
  xb_all[:,0]  = xb
  B_all[:,:,0] = B

  for k in tqdm(range(nIter)):

    # E-step
    Xs, Ps, Ps_lag, Xa, Pa, Xf, Pf, H = _EKS(Nx, No, T, xb, B, Q, R, Yo, f, jacF,
                                             h, jacH, alpha)   
    
    loglik[k] = _likelihood(Xf, Pf, Yo, R, H)
    rmse_em[k] = RMSE(Xs - Xt)
    Xs_all[...,k] = Xs
    cov_prob_em[k] = cov_prob(Xs, Ps, Xt)

    # M-step
    xb_new, B_new, Q_new, R_new = _maximize(Xs, Ps, Ps_lag, Yo, h, jacH, f, jacF,
                            structQ=structQ, baseQ=baseQ)
    if estimateQ:
      Q = Q_new
    if estimateR:
      R = R_new
    if estimateX0:
      xb = xb_new
      B = B_new

    Q_all[:,:,k+1] = Q
    R_all[:,:,k+1] = R
    xb_all[:,k+1] = xb
    B_all[:,:,k+1] = B

  res = {
          'smoothed_states'                : Xs_all,
          'EM_background_state'            : xb_all,
          'EM_background_covariance'       : B_all,
          'EM_model_noise_covariance'      : Q_all,
          'EM_observation_noise_covariance': R_all, 
          'loglikelihood'                  : loglik,
          'RMSE'                           : rmse_em,
          'cov_prob'                       : cov_prob_em,
          'params'                         : params
        }
  return res
