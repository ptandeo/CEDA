import numpy as np
from multiprocessing import Pool
from numpy.linalg import inv
from scipy.integrate import ode
import numpy as np
from algos.utils import RMSE, inv_svd, sqrt_svd
from tqdm import tqdm

def _maximize(X, obs, H, f, structQ='full', baseQ=None):
  Nx, Ne, T = X.shape
  No = obs.shape[0]

  xb = np.mean(X[:,:,0], 1)
  B = np.cov(X[:,:,0])

  sumSig = np.zeros((Nx, Ne, T-1))
  for t in range(T-1):
    sumSig[...,t] = X[...,t+1] - f(X[...,t])
  sumSig = np.reshape(sumSig, (Nx, (T-1)*Ne))
  sumSig = sumSig.dot(sumSig.T) / Ne
  if structQ == 'full':
    Q = sumSig/(T-1)
  elif structQ == 'diag':
    Q = np.diag(np.diag(sumSig))/T
  elif structQ == 'const':
    alpha = np.trace(inv_svd(baseQ).dot(sumSig)) / ((T-1)*Nx)
    Q = alpha*baseQ

  W = np.zeros([No, Ne, T-1])
  nobs = 0
  for t in range(T-1):
    if not np.isnan(obs[0,t]):
      nobs += 1
      W[:,:,t] = np.tile(obs[:,t], (Ne, 1)).T - H.dot(X[:,:,t+1])
  W = np.reshape(W, (No, (T-1)*Ne))
  R = W.dot(W.T) / (nobs*Ne)

  return xb, B, Q, R

def _likelihood(Xf, obs, H, R):
  T = Xf.shape[2]

  x = np.mean(Xf, 1)

  l = 0
  for t in range(T):
    if not np.isnan(obs[0,t]):
      innov = obs[:, t] - H.dot(x[:, t])
      Y = H.dot(Xf[:, :, t])

      sig = np.cov(Y) + R
      l -= .5 * np.log(np.linalg.det(sig)) 
      l -= .5 * innov.T.dot(inv(sig)).dot(innov)
  return l

def _EnKF(Nx, T, No, xb, B, Q, R, Ne, alpha, f, H, obs, prng):
  sqQ = sqrt_svd(Q)
  sqR = sqrt_svd(R)
  sqB = sqrt_svd(B)

  Xa = np.zeros([Nx, Ne, T+1])
  Xf = np.zeros([Nx, Ne, T])

  # Initialize ensemble
  for i in range(Ne):
    Xa[:,i,0] = xb + sqB.dot(prng.normal(size=Nx))

  for t in range(T):
    # Forecast
    # for i in range(Ne):
    #   Xf[:,i,t] = f(Xa[:,i,t]) + sqQ.dot(prng.normal(size=Nx))
    Xf[:,:,t] = f(Xa[:,:,t]) + sqQ.dot(prng.normal(size=(Nx, Ne)))
    Y = H.dot(Xf[:,:,t]) + sqR.dot(prng.normal(size=(No, Ne)))

    # Update
    if np.isnan(obs[0,t]):
      Xa[:,:,t+1] = Xf[:,:,t]
    else:
      Pfxx = np.cov(Xf[:,:,t])
      K = Pfxx.dot(H.T).dot(inv_svd(H.dot(Pfxx).dot(H.T) + R/alpha))
      innov = np.tile(obs[:,t], (Ne, 1)).T - Y
      Xa[:,:,t+1] = Xf[:,:,t] + K.dot(innov)
#      for i in range(Ne):
#        innov = obs[:,t] - Y[:,i]
#        Xa[:,i,t+1] = Xf[:,i,t] + K.dot(innov)

  return Xa, Xf

def _EnKS(Nx, Ne, T, H, R, Yo, Xt, No, xb, B, Q, alpha, f, prng):
  Xa, Xf = _EnKF(Nx, T, No, xb, B, Q, R, Ne, alpha, f, H, Yo, prng)

  Xs = np.zeros([Nx, Ne, T+1])
  Xs[:,:,-1] = Xa[:,:,-1]
  for t in range(T-1,-1,-1):
    Paf = np.cov(Xa[:,:,t], Xf[:,:,t])[:Nx, Nx:] ### MODIF PIERRE ###
    Pff = np.cov(Xf[:,:,t])
    try:
      K = Paf.dot(inv(Pff))
    except:
      K = Paf.dot(Pff**(-1)) ### MODIF PIERRE ###
    Xs[:,:,t] = Xa[:,:,t] + K.dot(Xs[:,:,t+1] - Xf[:,:,t])
   # for i in range(Ne):
   #   Xs[:,i, t] = Xa[:,i,t] + K.dot(Xs[:,i,t+1] - Xf[:,i,t])

  return Xs, Xa, Xf

def EnKS(params, prng):
  Nx = params['state_size']
  Ne = params['nb_particles']
  T  = params['temporal_window_size']
  H  = params['observation_matrix']
  R  = params['observation_noise_covariance']
  Yo = params['observations']
  Xt = params['true_state']
  No = params['observation_size']
  xb = params['background_state']
  B  = params['background_covariance']
  Q  = params['model_noise_covariance']
  alpha = params['inflation_factor']
  f  = params['model_dynamics']

  Xs, Xa, Xf = _EnKS(Nx, Ne, T, H, R, Yo, Xt, No, xb, B, Q, alpha, f, prng)

  res = {
          'smoothed_ensemble': Xs,
          'analysis_ensemble': Xa,
          'forecast_ensemble': Xf,
          'loglikelihood'    : _likelihood(Xf, Yo, H, R),
          'RMSE'             : RMSE(Xt - Xs.mean(1)),
          'params'           : params
         }
  return res

def EM_EnKS(params, prng):
  xb      = params['initial_background_state']
  B       = params['initial_background_covariance']
  Q       = params['initial_model_noise_covariance']
  R       = params['initial_observation_noise_covariance']
  f       = params['model_dynamics']
  H       = params['observation_matrix']
  Yo      = params['observations']
  Ne      = params['nb_particles']
  nIter   = params['nb_EM_iterations']
  Xt      = params['true_state']
  alpha   = params['inflation_factor']
  Nx      = params['state_size']
  T       = params['temporal_window_size']
  No      = params['observation_size']
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

  Q_all  = np.zeros(np.r_[Q.shape,  nIter+1])
  R_all  = np.zeros(np.r_[R.shape,  nIter+1])
  B_all  = np.zeros(np.r_[B.shape,  nIter+1])
  xb_all = np.zeros(np.r_[xb.shape, nIter+1])

  Q_all[:,:,0] = Q
  R_all[:,:,0] = R
  xb_all[:,0]  = xb
  B_all[:,:,0] = B

  for k in tqdm(range(nIter)):

    # E-step
    Xs, Xa, Xf = _EnKS(Nx, Ne, T, H, R, Yo, Xt, No, xb, B, Q, alpha, f, prng)
    loglik[k] = _likelihood(Xf, Yo, H, R)
    rmse_em[k] = RMSE(Xt - Xs.mean(1))

    # M-step
    xb_new, B_new, Q_new, R_new = _maximize(Xs, Yo, H, f, structQ=structQ, baseQ=baseQ)
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
          'smoothed_ensemble'              : Xs,
          'EM_background_state'            : xb_all,
          'EM_background_covariance'       : B_all,
          'EM_model_noise_covariance'      : Q_all,
          'EM_observation_noise_covariance': R_all,
          'loglikelihood'                  : loglik,
          'RMSE'                           : rmse_em,
          'params'                         : params
        }
  return res
