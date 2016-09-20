from scipy.integrate import ode
import numpy as np

def l96_f(t, y, F):
  J = np.shape(y)[0]
  #dy = np.zeros(J)
  dy=y*0

  if y.ndim==1:
    #single integration
    for j in range(J-1):
      dy[j] = (y[j+1] - y[j-2]) * y[j-1] - y[j] + F
    dy[-1] = (y[0] - y[-3]) * y[-2] - y[-1] + F
  else:
    #ensemble integration
    disp('ensemble')
    disp('y.ndim')
    for j in range(J-1):
      dy[j,:] = (y[j+1,:] - y[j-2,:]) * y[j-1,:] - y[j,:] + F
    dy[-1,:] = (y[0,:] - y[-3,:]) * y[-2,:] - y[-1,:] + F

  return(dy)

def l96_predict(y0, dT, F):
  r = ode(l96_f).set_integrator('dopri5')
  r.set_initial_value(y0, 0).set_f_params(F)
  return (r.integrate(r.t + dT))
