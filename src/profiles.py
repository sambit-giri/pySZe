import numpy as np 

try:
	from baryonification.profiles import *
except:
	print("Install baryonification package.")

def cvir_fct(mvir,z):
    """
    Concentrations form Dutton+Maccio (2014)
    c200 (200 times RHOC)
    Assumes PLANCK coismology
    """
    A = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
    B = -0.101 + 0.026*z
    return 10.0**A*(mvir/1.0e12)**(B)


def M200_fct(M500,c):
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return 2.0/5.0/y0**3.0*M500
