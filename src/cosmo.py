import numpy as np 
from scipy.integrate import quad, simps, cumtrapz

def comoving_dist(z,par):
	c  = 3e8/1000 #km/s
	H0 = 100*par.cosmo.h0 #km s^-1 Mpc^-1
	fn = lambda x: 1/np.sqrt(par.cosmo.Om*(1+x)**3+(1-par.cosmo.Om))
	dc = (c/H0)*quad(fn, 0, z)[0]*par.cosmo.h0 # Mpc/h
	return dc 

def angular_dist(z,par):
	dc = comoving_dist(z,par)
	return dc/(1+z)