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
  r = ode(l63_f, l63_jac).set_integrator('dopri5')
  r.set_initial_value(y0, 0).set_f_params(sigma, rho, beta)
  return r.integrate(r.t+dT)