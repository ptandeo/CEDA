#!/usr/bin/python
"""
Lorenz-96 model. Wrapper to fortran L96 code
"""

import numpy as np
import models.l96_for as tfor

class M:

    def __init__(self,dtcy=0.05,force=8,nx=40):
        "Lorenz-96 parameters"
        self.kt=25
        self.dtcy=dtcy
        self.dt=dtcy/self.kt # integration time step
        self.nx=nx
        self.xpar=np.zeros(nx)+force

    def integ(self,xold):
        "Time integration of Lorenz-96 (single and ensemble)"
        if xold.ndim==1:
            #single integration
            x=tfor.l96.tinteg1scl(self.kt,xold,self.xpar,self.dt)
        else:
            #ensemble integration
            x=xold*0
            for i in range(np.shape(xold)[1]):
                 x[:,i]=tfor.l96.tinteg1scl(self.kt,xold[:,i],self.xpar,self.dt)
        return x
