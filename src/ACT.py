"""
Atacama Cosmology Telescope:
It contains modules to mimic the telescope filters and other systematics.
"""

import numpy as np 

import scipy.interpolate as interpolate
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps, cumtrapz

from astropy.cosmology import Planck15
from astropy import units 

from .kSZ import get_fit_dTkSZ_theta
from .tSZ import get_fit_dTtSZ_theta

def W_thetad(theta, thetad):
	'''
	CAP filter
	'''
	# print(theta.size)
	if theta.size>1:
		out = np.zeros(theta.shape)
		out[theta<=np.sqrt(2)*thetad] = -1
		out[theta<thetad] = 1
		return out
	if theta<thetad: return 1
	if thetad<=theta and theta<=np.sqrt(2)*thetad: return -1
	return 0


def B_f90(theta):
	'''
	Beam profiles: f90
	'''
	G_fwhm  = 2.1*units.arcmin
	G_sigma = G_fwhm/2.355 #(2*np.sqrt(2*np.log(2)))
	return np.exp(-theta**2/G_sigma**2/2)

def B_f150(theta):
	'''
	Beam profiles: f150
	'''
	G_fwhm  = 1.3*units.arcmin
	G_sigma = G_fwhm/2.355 #(2*np.sqrt(2*np.log(2)))
	return np.exp(-theta**2/G_sigma**2/2)


def AP_f90(thetad, fit_dTkSZ_theta, theta_bins, z, n_bins=256, cosmo=Planck15):
	Bf90 = B_f90(theta_bins)
	dTkSZ_theta_f90  = np.convolve(fit_dTkSZ_theta(theta_bins),Bf90/Bf90.sum(),mode='same').to('microKelvin')

	fit_dTkSZ_theta_f90 = lambda thet: 10**interp1d(np.log10(theta_bins.to('arcmin').value), 
												np.log10(dTkSZ_theta_f90.value),
												fill_value='extrapolate'
											)(np.log10(thet.to('arcmin').value))*dTkSZ_theta_f90.unit

	xr = np.linspace(-20, 20, n_bins)
	yr = np.linspace(-20, 20, n_bins)
	xxr, yyr = np.meshgrid(xr, yr, sparse=True)
	rri = np.sqrt(xxr**2 + yyr**2)*units.Mpc
	thetai = (rri/cosmo.comoving_distance(z)*units.rad).to('arcmin')
	# print(thetai.shape)

	theta_ = (thetai[1:,1:]+thetai[1:,:-1]+thetai[:-1,1:]+thetai[:-1,:-1])/4
	I = fit_dTkSZ_theta_f90(theta_)
	dtheta0 = np.abs(thetai[1:,1:]-thetai[:-1,1:])
	dtheta1 = np.abs(thetai[1:,1:]-thetai[1:,:-1])
	stp = I*W_thetad(theta_, thetad)*dtheta0*dtheta1
	out = stp.sum()

	return out

def AP_f150(thetad, fit_dTkSZ_theta, theta_bins, z, n_bins=256, cosmo=Planck15):
	Bf150 = B_f150(theta_bins)
	dTkSZ_theta_f150 = np.convolve(fit_dTkSZ_theta(theta_bins),Bf150/Bf150.sum(),mode='same').to('microKelvin')

	fit_dTkSZ_theta_f150 = lambda thet: 10**interp1d(np.log10(theta_bins.to('arcmin').value), 
												np.log10(dTkSZ_theta_f150.value),
												fill_value='extrapolate'
											)(np.log10(thet.to('arcmin').value))*dTkSZ_theta_f150.unit

	xr = np.linspace(-20, 20, n_bins)
	yr = np.linspace(-20, 20, n_bins)
	xxr, yyr = np.meshgrid(xr, yr, sparse=True)
	rri = np.sqrt(xxr**2 + yyr**2)*units.Mpc
	thetai = (rri/cosmo.comoving_distance(z)*units.rad).to('arcmin')
	# print(thetai.shape)

	theta_ = (thetai[1:,1:]+thetai[1:,:-1]+thetai[:-1,1:]+thetai[:-1,:-1])/4
	I = fit_dTkSZ_theta_f150(theta_)
	dtheta0 = np.abs(thetai[1:,1:]-thetai[:-1,1:])
	dtheta1 = np.abs(thetai[1:,1:]-thetai[1:,:-1])
	stp = I*W_thetad(theta_, thetad)*dtheta0*dtheta1
	out = stp.sum()

	return out


def get_TkSZ_f90_thetad(Mv, z, z_min=None, z_max=None, theta_nbin=1000, l_nbin=1000, n_bins=256,
						func_model_ne=None, cosmo=Planck15):
	fit_dTkSZ_theta, theta_bins = get_fit_dTkSZ_theta(Mv, z, z_min=z_min, z_max=z_max, 
											theta_nbin=theta_nbin, l_nbin=l_nbin,
											func_model_ne=func_model_ne, cosmo=cosmo, 
											ne_bin=ne_bin, r_bin=r_bin)

	thetad_bins = np.linspace(1,6,9)*units.arcmin

	TkSZ_f90_thetad = np.array([])
	for ii in thetad_bins:
		bla = AP_f90(ii, fit_dTkSZ_theta, theta_bins, z, n_bins=n_bins, cosmo=cosmo)
		TkSZ_f90_thetad = np.append(TkSZ_f90_thetad,bla.value)
	TkSZ_f90_thetad = TkSZ_f90_thetad*bla.unit

	return TkSZ_f90_thetad, thetad_bins

def get_TkSZ_f150_thetad(Mv, z, z_min=None, z_max=None, theta_nbin=1000, l_nbin=1000, n_bins=256,
						func_model_ne=None, cosmo=Planck15):
	fit_dTkSZ_theta, theta_bins = get_fit_dTkSZ_theta(Mv, z, z_min=z_min, z_max=z_max, 
											theta_nbin=theta_nbin, l_nbin=l_nbin,
											func_model_ne=func_model_ne, cosmo=cosmo, 
											ne_bin=ne_bin, r_bin=r_bin)
	
	thetad_bins = np.linspace(1,6,9)*units.arcmin

	TkSZ_f150_thetad = np.array([])
	for ii in thetad_bins:
		bla = AP_f150(ii, fit_dTkSZ_theta, theta_bins, z, n_bins=n_bins, cosmo=cosmo)
		TkSZ_f150_thetad = np.append(TkSZ_f150_thetad,bla.value)
	TkSZ_f150_thetad = TkSZ_f150_thetad*bla.unit

	return TkSZ_f150_thetad, thetad_bins


def get_TkSZ_thetad(Mv, z, z_min=None, z_max=None, theta_nbin=1000, l_nbin=1000, n_bins=256,
						func_model_ne=None, ne_bin=None, r_bin=None, cosmo=Planck15):
	fit_dTkSZ_theta, theta_bins = get_fit_dTkSZ_theta(Mv, z, z_min=z_min, z_max=z_max, 
											theta_nbin=theta_nbin, l_nbin=l_nbin,
											func_model_ne=func_model_ne, cosmo=cosmo, 
											ne_bin=ne_bin, r_bin=r_bin)
	
	thetad_bins = np.linspace(1,6,9)*units.arcmin

	TkSZ_f150_thetad = np.array([])
	TkSZ_f90_thetad = np.array([])

	for ii in thetad_bins:
		bla = AP_f90(ii, fit_dTkSZ_theta, theta_bins, z, n_bins=n_bins, cosmo=cosmo)
		TkSZ_f90_thetad = np.append(TkSZ_f90_thetad,bla.value)
		bla = AP_f150(ii, fit_dTkSZ_theta, theta_bins, z, n_bins=n_bins, cosmo=cosmo)
		TkSZ_f150_thetad = np.append(TkSZ_f150_thetad,bla.value)

	TkSZ_f90_thetad  = TkSZ_f90_thetad*bla.unit
	TkSZ_f150_thetad = TkSZ_f150_thetad*bla.unit

	return TkSZ_f90_thetad, TkSZ_f150_thetad, thetad_bins



def get_TtSZ_thetad(Mv, z, z_min=None, z_max=None, theta_nbin=1000, l_nbin=1000, n_bins=256,
						func_model_Pe=None, Pe_bin=None, r_bin=None, cosmo=Planck15):
	fit_dTtSZ_theta_90, theta_bins_90 = get_fit_dTtSZ_theta(Mv, z, 90*units.GHz,
											z_min=z_min, z_max=z_max, 
											theta_nbin=theta_nbin, l_nbin=l_nbin,
											func_model_Pe=func_model_Pe, cosmo=cosmo, 
											Pe_bin=Pe_bin, r_bin=r_bin)

	fit_dTtSZ_theta_150, theta_bins_150 = get_fit_dTtSZ_theta(Mv, z, 150*units.GHz,
											z_min=z_min, z_max=z_max, 
											theta_nbin=theta_nbin, l_nbin=l_nbin,
											func_model_Pe=func_model_Pe, cosmo=cosmo, 
											Pe_bin=Pe_bin, r_bin=r_bin)

	# print('90GHz, theta_bins', theta_bins_90)
	# print('fit_dTtSZ_theta:', fit_dTtSZ_theta_90(theta_bins_90))
	# print('150GHz, theta_bins', theta_bins_150)
	# print('fit_dTtSZ_theta:', fit_dTtSZ_theta_150(theta_bins_150))
	
	thetad_bins = np.linspace(1,6,9)*units.arcmin

	TtSZ_f150_thetad = np.array([])
	TtSZ_f90_thetad = np.array([])

	for ii in thetad_bins:
		bla = -AP_f90(ii, lambda x: -fit_dTtSZ_theta_90(x), theta_bins_90, z, n_bins=n_bins, cosmo=cosmo)
		TtSZ_f90_thetad = np.append(TtSZ_f90_thetad,bla.value)
		bla = -AP_f150(ii, lambda x: -fit_dTtSZ_theta_150(x), theta_bins_150, z, n_bins=n_bins, cosmo=cosmo)
		TtSZ_f150_thetad = np.append(TtSZ_f150_thetad,bla.value)

	TtSZ_f90_thetad  = TtSZ_f90_thetad*bla.unit
	TtSZ_f150_thetad = TtSZ_f150_thetad*bla.unit

	return TtSZ_f90_thetad, TtSZ_f150_thetad, thetad_bins

