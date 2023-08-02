import numpy as np

import scipy.interpolate as interpolate
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps, cumtrapz

from astropy.cosmology import Planck15
from astropy import units, constants 

from glob import glob
import pickle
from tqdm import tqdm

def get_fit_dTtSZ_theta(Mv, z, nu, z_min=None, z_max=None, theta_nbin=1000, l_nbin=1000, 
							func_model_Pe=None, Pe_bin=None, r_bin=None, cosmo=Planck15):

	# Mc, mu, Tej, eta_star, eta_cga = theta_bar

	if z_max is None: z_max = z+0.15
	if z_min is None: z_min = z-0.15

	# if func_model_ne is None: func_model_ne = model_ne

	# assert func_model_ne is not None
	# ne = func_model_ne(theta_bar)

	fit_Pe = None
	if func_model_Pe is None:
		fit_Pe = lambda r: 10**interp1d(np.log10(r_bin.to('Mpc').value) if type(r_bin)==units.quantity.Quantity else np.log10(r_bin), #np.log10(rbin/h), 
	                                np.log10(Pe_bin.value),
	                                fill_value='extrapolate'
	                               )(np.log10(r.to('Mpc').value))*Pe_bin.unit
	else:
		fit_Pe = func_model_Pe
	assert fit_Pe is not None

	l_min, l_max = cosmo.comoving_distance(z_min), cosmo.comoving_distance(z_max)
	# print(l_min, l_max, l_max-l_min)

	# theta_nbin = 1000 #256
	theta_bins = 10**np.linspace(-2,np.log10(20),theta_nbin)*units.arcmin

	theta_bins_2_r = cosmo.comoving_distance(z)*theta_bins.to('radian')/units.rad

	# l_nbin = 1000
	l_bins = np.linspace(-15,15,l_nbin)*units.Mpc

	y_theta = np.zeros(theta_bins_2_r.shape)

	for i in range(theta_bins_2_r.shape[0]):
	    thetr = theta_bins_2_r[i]
	    ll  = np.sqrt(l_bins**2+thetr**2)
	    Pel = fit_Pe(ll)
	    IPe = ((constants.sigma_T/constants.m_e/constants.c**2)*simps(Pel.value, l_bins.value)*Pel.unit*l_bins.unit).to('')
	    y_theta[i] = IPe.value

	def fv(nu):
		x = (constants.h*nu/constants.k_B/cosmo.Tcmb(z)).to('').value
		return x/np.tanh(x/2) - 4

	fit_dTtSZ_div_Tcmb_theta = lambda thet: 10**interp1d(np.log10(theta_bins.to('arcmin').value), 
                                np.log10(y_theta),
                                fill_value='extrapolate'
                               )(np.log10(thet.to('arcmin').value))*fv(nu) 

	fit_dTtSZ_theta = lambda thet: fit_dTtSZ_div_Tcmb_theta(thet)*cosmo.Tcmb(z)
	return fit_dTtSZ_theta, theta_bins

