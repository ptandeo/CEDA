from scipy.integrate import ode
import numpy as np

def l63_f(t, y, sigma, rho, beta):
  dy = np.zeros(3)

  dy[0] = sigma*(y[1] - y[0])
  dy[1] = y[0]*(rho - y[2]) - y[1]
  dy[2] = y[0]*y[1] - beta*y[2]

  return dy

def l63_jac(y, dt, sigma, rho, beta):
  J = np.zeros((3,3))
  J[0,0] = -sigma
  J[1,0] = rho - y[2]
  J[2,0] = y[1]
  J[0,1] = sigma
  J[1,1] = -1
  J[2,1] = y[0]
  J[0,2] = 0
  J[1,2] = -y[0]
  J[2,2] = -beta
  return dt*J + np.eye(3)

def l63_predict(y0, dT, sigma, rho, beta):
  #r = ode(l63_f, l63_jac).set_integrator('dop853')
  #r.set_initial_value(y0, 0).set_f_params(sigma, rho, beta)
  #return r.integrate(r.t+dT)
  X1 = np.copy(y0)
  k1 = np.zeros(X1.shape)
  k1[0] = sigma*(X1[1] - X1[0])
  k1[1] = X1[0]*(rho-X1[2]) - X1[1]
  k1[2] = X1[0]*X1[1] - beta*X1[2]

  X2 = np.copy(y0+k1/2*dT)
  k2 = np.zeros(y0.shape)
  k2[0] = sigma*(X2[1] - X2[0])
  k2[1] = X2[0]*(rho-X2[2]) - X2[1]
  k2[2] = X2[0]*X2[1] - beta*X2[2]   

  X3 = np.copy(y0+k2/2*dT)
  k3 = np.zeros(y0.shape)
  k3[0] = sigma*(X3[1] - X3[0])
  k3[1] = X3[0]*(rho-X3[2]) - X3[1]
  k3[2] = X3[0]*X3[1] - beta*X3[2]

  X4 = np.copy(y0+k3*dT)
  k4 = np.zeros(y0.shape)
  k4[0] = sigma*(X4[1] - X4[0])
  k4[1] = X4[0]*(rho-X4[2]) - X4[1]
  k4[2] = X4[0]*X4[1] - beta*X4[2]

  return y0 + dT/6.*(k1+2*k2+2*k3+k4)
