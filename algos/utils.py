import numpy as np

def mat_approx_svd(A):
   "Returns the matrix approximation (using nb components) by SVD"
   nb = 5
   U, s, V = np.linalg.svd(A, full_matrices=True)
   S = np.diag(s)
   A_approx = np.dot(U[:,0:nb], np.dot(S[0:nb,0:nb], V[0:nb,:]))
   return A_approx

def inv_svd(A):
    "Returns the inverse matrix by SVD"
    U, s, V = np.linalg.svd(A, full_matrices=True)
    invs = 1./s
    n = np.size(s)
    invS = np.zeros((n,n))
    invS[:n, :n] = np.diag(invs)
    invA=np.dot(V.T,np.dot(invS, U.T))
    return invA

def sqrt_svd(A):
   "Returns the square root matrix by SVD"
   U, s, V = np.linalg.svd(A)#, full_matrices=True)
   sqrts = np.sqrt(s)
   n = np.size(s)
   sqrtS = np.zeros((n,n))
   sqrtS[:n, :n] = np.diag(sqrts)
   sqrtA=np.dot(V.T,np.dot(sqrtS, U.T))
   return sqrtA

def climat_background(X_true):
  "Returns a climatology (mean and covariance)"
  xb = np.mean(X_true,1)
  B  = np.cov(X_true)
  return xb, B

def RMSE(E):
  "Returns the Root Mean Squared Error"
  return np.sqrt(np.mean(E**2))

def cov_prob(Xs, Ps, X_true):
  "Returns the number of true state in the 95% confidence interval"
  n, T = np.shape(X_true)
  cov_prob = 0
  for i_n in range(n):
    cov_prob += sum((np.squeeze(Xs[i_n,:]) - 1.96 * np.sqrt(np.squeeze(Ps[i_n,i_n,:])) <= X_true[i_n,:]) & (np.squeeze(Xs[i_n,:]) + 1.96 * np.sqrt(np.squeeze(Ps[i_n,i_n,:])) >= X_true[i_n,:])) / T
  return cov_prob

def gen_truth(f, x0, T, Q, prng):
  sqQ = sqrt_svd(Q)
  Nx = x0.size
  Xt = np.zeros((Nx, T+1))
  Xt[:,0] = x0
  for k in range(T):
    Xt[:,k+1] = f(Xt[:,k]) + sqQ.dot(prng.normal(size=Nx))
  return Xt

def gen_truth_stochastic(f, x0, T, Q, subdivs, prng):
  sqQ = sqrt_svd(Q)
  Nx = x0.size
  Xt = np.zeros((Nx, T+1))
  Xt[:,0] = x0
  tmp = x0
  for k in range(T):
    for t in range(subdivs):
      tmp = f(tmp) + 1./np.sqrt(subdivs)*sqQ.dot(prng.normal(size=Nx))
    Xt[:,k+1] = tmp
  return Xt

def gen_truth_parameter(f, x0, T, beta_min, beta_max, prng):
  Nx = x0.size
  Xt = np.zeros((Nx, T+1))
  Xt[:,0] = x0
  for k in range(T):
    f_tmp = lambda x: f(x, prng.uniform(beta_min, beta_max))
    Xt[:,k+1] = f_tmp(Xt[:,k])
  return Xt

def gen_truth_multiplicative(f, x0, T, Q, prng):
  sqQ = sqrt_svd(Q)
  Nx = x0.size
  Xt = np.zeros((Nx, T+1))
  Xt[:,0] = x0
  for k in range(T):
    Xt[:, k+1] = f(Xt[:,k]) * np.exp(sqQ.dot(prng.normal(size=Nx)))
  return Xt

def gen_obs(h, Xt, R, nb_assim, prng):
  sqR = sqrt_svd(R)
  No = sqR.shape[0]
  T = Xt.shape[1] -1
  Yo = np.zeros((No, T))
  Yo[:] = np.nan
  for k in range(T):
    if k%nb_assim == 0:
      Yo[:,k] = h(Xt[:,k+1]) + sqR.dot(prng.normal(size=No))
  return Yo
