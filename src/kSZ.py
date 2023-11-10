import numpy as np
import warnings

import scipy.interpolate as interpolate
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps, cumtrapz

from astropy.cosmology import Planck15
from astropy import units, constants 

from glob import glob
import pickle
from tqdm import tqdm

# try:
# 	import baryonification as bfc
# except:
# 	print("Install baryonification package.")

from .cosmo import * 
from .profiles import *

def get_fit_dTkSZ_theta(Mv, z, z_min=None, z_max=None, theta_nbin=1000, l_nbin=1000, 
							func_model_ne=None, ne_bin=None, r_bin=None, cosmo=Planck15):

	# Mc, mu, Tej, eta_star, eta_cga = theta_bar

	if z_max is None: z_max = z+0.15
	if z_min is None: z_min = z-0.15

	# if func_model_ne is None: func_model_ne = model_ne

	# assert func_model_ne is not None
	# ne = func_model_ne(theta_bar)

	fit_ne = None
	if func_model_ne is None:
		fit_ne = lambda r: 10**interp1d(np.log10(r_bin.to('Mpc').value) if type(r_bin)==units.quantity.Quantity else np.log10(r_bin), #np.log10(rbin/h), 
	                                np.log10(ne_bin.value),
	                                fill_value='extrapolate'
	                               )(np.log10(r.to('Mpc').value))*ne_bin.unit
	else:
		fit_ne = func_model_ne
	assert fit_ne is not None

	l_min, l_max = cosmo.comoving_distance(z_min), cosmo.comoving_distance(z_max)
	# print(l_min, l_max, l_max-l_min)

	# theta_nbin = 1000 #256
	theta_bins = 10**np.linspace(-2,np.log10(20),theta_nbin)*units.arcmin

	theta_bins_2_r = cosmo.comoving_distance(z)*theta_bins.to('radian')/units.rad

	# l_nbin = 1000
	l_bins = np.linspace(-15,15,l_nbin)*units.Mpc

	tau_theta = np.zeros(theta_bins_2_r.shape)

	for i in range(theta_bins_2_r.shape[0]):
		thetr = theta_bins_2_r[i]
		ll  = np.sqrt(l_bins**2+thetr**2)
		nel = fit_ne(ll)
		Ine = (constants.sigma_T*simps(nel.value, l_bins.value)*nel.unit*l_bins.unit).to('')
		tau_theta[i] = Ine.value

	fit_dTkSZ_div_Tcmb_theta = lambda thet: 10**interp1d(np.log10(theta_bins.to('arcmin').value), 
                                np.log10(tau_theta),
                                fill_value='extrapolate'
                               )(np.log10(thet.to('arcmin').value))*1.06e-3

	fit_dTkSZ_theta = lambda thet: fit_dTkSZ_div_Tcmb_theta(thet)*cosmo.Tcmb(z)
	return fit_dTkSZ_theta, theta_bins


class deltaTkSZ:
	'''
	A class to slove the differential kinetic-SZ temperature. 
	'''
	def __init__(self, par):
		self.initialise(par)

	def initialise(self, par):
		self.par = par 

		N_rbin = 100
		rmax = 49.0
		rmin = 0.0005
		rbin = np.logspace(np.log10(rmin),np.log10(rmax),N_rbin,base=10)

		vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(self.par.files.cosmofct, usecols=(0,1,2,3), unpack=True)
		bias_tck = splrep(vc_m, vc_bias, s=0)
		corr_tck = splrep(vc_r, vc_corr, s=0)
		cosmo_corr = splev(rbin,corr_tck)

		#2-halo term
		# bin_r, bin_m, bin_bias, bin_corr = bfc.cosmo(self.par)
		bin_r, bin_m, bin_bias, bin_corr = cosmo(self.par)

		self.rbin = rbin
		self.bias_tck = bias_tck
		self.cosmo_corr = cosmo_corr

		self.Xh = 0.76
		self.mp = 8.348e-58*self.par.cosmo.h0   # Msun/h
		self.mpc_to_cm = 3.086e24
		self.m_to_cm = 100
		self.sig_T   = 6.65e-29 * self.m_to_cm**2 * self.par.cosmo.h0**2 # cm^2/h^2
		self.arcmins_to_rad = 1./60*np.pi/180

		self.Tcmb_muK = 2.725*(1+self.par.cosmo.z) * 1e6 # muK
		self.dang = angular_dist(self.par.cosmo.z,self.par)

	def density_to_ne(self, dens_gas_NFW):
		ne_NFW = (self.Xh+1)/2*dens_gas_NFW/self.mp / self.mpc_to_cm**3 * self.par.cosmo.h0**3 # 1/cm^3
		ne_gas_tck = splrep(np.log10(self.rbin), np.log10(ne_NFW))
		ne_gas_fun = lambda r: 10**splev(np.log10(r), ne_gas_tck)		
		return ne_gas_fun

	def ne_gas_profile(self, Mv):
		if self.par.code.verbose: print('Assuming the gas to follow the NFW profile.')
		cv = cvir_fct(Mv,self.par.cosmo.z)
		cosmo_bias = splev(Mv, self.bias_tck)
		# fE, dE, mE = bfc.profiles(self.rbin,Mv,cv,self.cosmo_corr,cosmo_bias,self.par)
		fE, dE, mE = profiles(self.rbin,Mv,cv,self.cosmo_corr,cosmo_bias,self.par)

		dens = dE['HGA'] #+ dE['BG'] # h^2 Msun/Mpc^3
		dens_gas = dens*fE['HGA']  # h^2 Msun/Mpc^3
		# dens_gas_tck = splrep(np.log10(rbin), np.log10(dens_gas_NFW))

		return self.density_to_ne(dens_gas)

	def ne_NFW_profile(self, Mv):
		cv = cvir_fct(Mv,self.par.cosmo.z)
		cosmo_bias = splev(Mv, self.bias_tck)
		# fE, dE, mE = bfc.profiles(self.rbin,Mv,cv,self.cosmo_corr,cosmo_bias,self.par)
		fE, dE, mE = profiles(self.rbin,Mv,cv,self.cosmo_corr,cosmo_bias,self.par)

		dens_NFW = dE['NFW'] #+ dE['BG'] # h^2 Msun/Mpc^3
		dens_gas_NFW = dens_NFW*self.par.cosmo.Ob/self.par.cosmo.Om  # h^2 Msun/Mpc^3
		# dens_gas_tck = splrep(np.log10(rbin), np.log10(dens_gas_NFW))

		return self.density_to_ne(dens_gas_NFW)

	def tau_gal(self, ne_gas_fun):
		if not self.par.code.verbose: warnings.filterwarnings("ignore") # Suppress warnings
		theta_arcmins = 10**np.linspace(-2,1,100) #np.linspace(0.001,10,100)
		tau_arcmins   = np.zeros_like(theta_arcmins)
		for i in tqdm(range(theta_arcmins.size), disable=not self.par.code.verbose): 
			th = theta_arcmins[i]*self.arcmins_to_rad
			fn = lambda x: ne_gas_fun(x)/np.sqrt(x**2-self.dang**2*th**2)*x
			tau_arcmins[i] = 2*self.sig_T*quad(fn, np.abs(self.dang*th), 30)[0] * self.mpc_to_cm 
		self.tau_arcmins   = tau_arcmins / self.par.cosmo.h0**3
		self.theta_arcmins = theta_arcmins
		return self.tau_arcmins, self.theta_arcmins

	def dTkSZ_map(self, ne_gas_fun, **kwargs):
		vpec_by_c = kwargs.get('vpec_by_c', 1.06e-3)
		tau_arcmins, theta_arcmins = self.tau_gal(ne_gas_fun)
		dTkSZ_arcmins = tau_arcmins*vpec_by_c*self.Tcmb_muK
		dTkSZ_arcmins_tck = splrep(np.log10(theta_arcmins), np.log10(dTkSZ_arcmins))
		dTkSZ_arcmins_fun = lambda x: 10**splev(np.log10(x), dTkSZ_arcmins_tck)
		self.theta_arcmins = theta_arcmins
		self.dTkSZ_arcmins = dTkSZ_arcmins
		return dTkSZ_arcmins_fun

	def run(self, Mv, NFW=False, **kwargs):
		if NFW: ne_gas_fun = self.ne_NFW_profile(Mv)
		else: ne_gas_fun = self.ne_gas_profile(Mv)
		self.dTkSZ_arcmins_fun = self.dTkSZ_map(ne_gas_fun, **kwargs)



