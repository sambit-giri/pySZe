import numpy as np
import matplotlib.pyplot as plt

import baryonification as bfc
from tqdm import tqdm
import pickle 

from scipy.interpolate import splrep, splev, griddata, interp1d
from astropy.cosmology import Planck15 as cosmo
from astropy import units, constants 

import MockObs
from MockObs import ACT

# COSMO PARAMETERS ################################################

Deltavir = 200.0      #overdensity with respect to rhoc
Ob       = cosmo.Ob0  #0.049      #0.0455 #0.049 #0.049
Om       = cosmo.Om0  #0.304      #0.272 #0.304
rhoc     = cosmo.critical_density0.to('solMass/Mpc^3').value #2.7755e11 
h        = cosmo.h #0.6814     #0.704 #0.6814

#proton number density in 1/cm^3
Msun_Mpc3_to_1_cm3 = (1.189e57/2.939e+73)
fproton = Ob/Om/2.0
fbar = Ob/Om

print("Cosmic baryon fraction = ", Ob/Om)

# PARAMS #########################################################

#initialise parameters
par = bfc.par()

par.files.transfct = "CDM_PLANCK_tk.dat"
par.files.cosmofct = "cosmofct.dat"
par.cosmo.z = 0.0
par.baryon.thco = 0.1

#2-halo term
bin_r, bin_m, bin_bias, bin_corr = bfc.cosmo(par)


# BINNING ########################################################

#radius bins
N_rbin = 100
rmax = 49.0
rmin = 0.0005
rbin = np.logspace(np.log10(rmin),np.log10(rmax),N_rbin,base=10)

# FUNCTIONS ######################################################

#Arnaud (2005) M-T relation
def M200_T_Arnaud(T):
    Ad = 5.34e14
    al = 1.71
    return Ad*(T/5.0)**al

#Arnaud 2007
def M500_T_Arnaud(T):
    Ad = 3.802e14
    al = 1.71
    return Ad*(T/5.0)**al


#Lieu et al 2016 (Eq.11)
def M500_T_XXL(T):
    logT = log10(T)
    a = 13.57
    b = 1.67
    logM = a+b*logT
    return 10.0**logM

#approx (c not quite right)
def M200_fct(M500,c):
    f = lambda y: log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return 2.0/5.0/y0**3.0*M500

def MARNAUD_ov_MXXL(T):
    Ad = 3.84e14
    al = 1.71
    MARNAUD = Ad*(T/5.0)**al
    a = 13.57
    b = 1.67
    MXXL = 10.0**(a+b*log10(T))
    return MARNAUD/MXXL

def cvir_fct(mvir):
    """
    Concentrations form Dutton+Maccio (2014)
    c200 (200 times RHOC)
    Assumes PLANCK coismology
    """
    A = 1.025
    B = 0.097
    return 10.0**A*(mvir/1.0e12)**(-B)

def r500_fct(r200,c):
    """
    From r200 to r500 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*r200

def M500_fct(M200,c):
    """
    From M200 to M500 assuming a NFW profiles
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return 5.0/2.0*M200*y0**3.0

def beta_of_thej(thej,a,b,c):
    return a*(np.log(thej/b))**c


def Rdelta_to_Mdelta(Rdelta, delta, z, rho_cr=None, cosmo=cosmo):
    if rho_cr is None: rho_cr = cosmo.critical_density(z)
    Mdelta = (4*np.pi/3)*Rdelta**3*rho_cr*delta
    return Mdelta.to('solMass')

def Mdelta_to_Rdelta(Mdelta, delta, z, rho_cr=None, cosmo=cosmo):
    if rho_cr is None: rho_cr = cosmo.critical_density(z)
    Rdelta = np.cbrt(3*Mdelta/4/np.pi/rho_cr/delta)
    return Rdelta.to('Mpc')


####################################################################
# READ COSMO.DAT
####################################################################

vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(par.files.cosmofct, usecols=(0,1,2,3), unpack=True)
bias_tck = splrep(vc_m, vc_bias, s=0)
corr_tck = splrep(vc_r, vc_corr, s=0)

cosmo_corr = splev(rbin,corr_tck)


# kSZ

# Model gas profile

def rho_2h(Mv):
    Mc, mu, Tej, eta_star, eta_cga = 10**15.3, 0.4, 6, 0.32, 0.6
    cv = cvir_fct(Mv)
    cosmo_bias = splev(Mv, bias_tck)
    fE, dE, mE = bfc.profiles(rbin,Mv,cv,Mc,mu,Tej,cosmo_corr,cosmo_bias,par)
    frgasE = (fbar - fE['CGA'] - fE['SGA'])/fbar
    # rhoE = dE['HGA']*frgasE*model_SZ.fproton*model_SZ.Msun_Mpc3_to_1_cm3
    # rhoE = (dE['HGA']+A2h*dE['BG'])*frgasE*Msun_Mpc3_to_1_cm3*fproton
    rhoE = dE['BG']*frgasE*Msun_Mpc3_to_1_cm3*fproton
    rhoE = (rhoE/units.cm**3*constants.m_p).to('g/cm^3') 
    return rhoE, rbin

def model_rho_GNFW(z=0.55):
    # From MCMC
    log10rho0 = 2.8
    x_ck  = 0.6
    b_k   = 2.6
    A_k2h = 1.1

    # Fixed in model
    g_k = -0.2
    a_k = 1

    rho_cr = cosmo.critical_density(z)
    M200 = 3e13*units.solMass
    R200 = Mdelta_to_Rdelta(M200, 200, z, rho_cr=rho_cr).to('Mpc')

    x = 10**np.linspace(-2,1.5)
    r = x*R200
    rho_GNFW = 10**log10rho0*(x/x_ck)**g_k*(1+(x/x_ck)**a_k)**(-(b_k-g_k)/a_k)

    rho2h, r2h = rho_2h(M200.value)
    fit_rho2h  = lambda r: 10**interp1d(np.log10(r2h/h), np.log10(rho2h.value), 
                                        fill_value='extrapolate'
                                        )(np.log10(r.to('Mpc').value))*rho2h.unit

    # rho_gas = (rho_GNFW*rho_cr+A_k2h*fit_rho2h(r))*cosmo.Ob(z)/cosmo.Om(z)
    rho_gas = (rho_GNFW*rho_cr)*cosmo.Ob(z)/cosmo.Om(z)+A_k2h*fit_rho2h(r)*(1+z)**3
    return rho_gas, r, x

def model_ne_GNFW(z=0.55):
    rho_gas, rr, xx = model_rho_GNFW(z=z)
    X_H  = 0.76
    ne   = (rho_gas*(X_H+1)/2/constants.m_p).to('1/cm^3')
    return ne, rr


# Schaan+2020

dataSchaan2020Fig7a = pickle.load(open('Schaan2020Fig7a_data.pkl', 'rb'))
cdist2angle = dataSchaan2020Fig7a['cdist2angle']

def plot_TkSZ_f90(data=None, ax=None):
	if ax is None:
		fig, ax = plt.subplots(figsize=(12,8))

	xx, yy = dataSchaan2020Fig7a['f90']['r_cdist'], dataSchaan2020Fig7a['f90']['mean']
	yu = np.array(dataSchaan2020Fig7a['f90']['mean+'])-np.array(dataSchaan2020Fig7a['f90']['mean'])
	yd = np.array(dataSchaan2020Fig7a['f90']['mean'])-np.array(dataSchaan2020Fig7a['f90']['mean-'])
	ax.errorbar(cdist2angle(xx), yy, yerr=[yu,yd], c='m', ms=15, ls=' ', alpha=0.99, label='90GHz', marker='.')

	if data is not None:
		thetad_bins, TkSZ_thetad = data['thetad_bins'], data['TkSZ_thetad']
		cr = 'g' if 'color' not in data.keys() else data['color']
		ls = ':' if 'ls' not in data.keys() else data['ls']
		lw = 5 if 'lw' not in data.keys() else data['lw']
		label = None if 'label' not in data.keys() else data['label']
		ax.plot(thetad_bins, TkSZ_thetad, c=cr, ls=ls, lw=lw, label=label)

	ax.legend()
	ax.set_yscale('log')
	#ax.yaxis.set_major_formatter(ScalarFormatter())
	return ax

def plot_TkSZ_f150(data=None, ax=None):
	if ax is None:
		fig, ax = plt.subplots(figsize=(12,8))

	xx, yy = dataSchaan2020Fig7a['f150']['r_cdist'], dataSchaan2020Fig7a['f150']['mean']
	yu = np.array(dataSchaan2020Fig7a['f150']['mean+'])-np.array(dataSchaan2020Fig7a['f150']['mean'])
	yd = np.array(dataSchaan2020Fig7a['f150']['mean'])-np.array(dataSchaan2020Fig7a['f150']['mean-'])
	ax.errorbar(cdist2angle(xx), yy, yerr=[yu,yd], c='b', ms=15, ls=' ', alpha=0.99, label='150GHz', marker='.')

	if data is not None:
		thetad_bins, TkSZ_thetad = data['thetad_bins'], data['TkSZ_thetad']
		cr = 'g' if 'color' not in data.keys() else data['color']
		ls = ':' if 'ls' not in data.keys() else data['ls']
		lw = 5 if 'lw' not in data.keys() else data['lw']
		label = None if 'label' not in data.keys() else data['label']
		ax.plot(thetad_bins, TkSZ_thetad, c=cr, ls=ls, lw=lw, label=label)

	ax.legend()
	ax.set_yscale('log')
	#ax.yaxis.set_major_formatter(ScalarFormatter())
	return ax


z = 0.55
z_min, z_max = 0.4, 0.7


Mv = 3e13  # CMASS halo mass in Amodeo+2020
Mc, mu, Tej, eta_star, eta_cga = 10**15.3, 0.4, 6, 0.32, 0.6
theta_bar = [Mc, mu, Tej, eta_star, eta_cga]

# ne_bin, r_bin = model_ne(Mc, mu, Tej, eta_star, eta_cga, Mv, A2h=1), rbin/h*units.Mpc
ne_bin, r_bin = model_ne_GNFW(z=0.55)

TkSZ_f90_thetad, TkSZ_f150_thetad, thetad_bins = ACT.get_TkSZ_thetad(Mv, z, 
                                                    z_min=z_min, z_max=z_max, 
                                                    n_bins=256, theta_nbin=1000*2, l_nbin=1000,
                                                    func_model_ne=None, cosmo=cosmo,
                                                    ne_bin=ne_bin, r_bin=r_bin
                                                )

fig, ax = plt.subplots(figsize=(12,8))
ax = plot_TkSZ_f90(ax=ax)
ax = plot_TkSZ_f150(ax=ax)
ax.loglog(thetad_bins, TkSZ_f90_thetad, c='k', ls='-.', lw=5, label='90GHz model')
ax.loglog(thetad_bins, TkSZ_f150_thetad, c='g', ls=':', lw=5, label='150GHz model')
ax.set_yscale('symlog')
ax.set_xscale('linear')
ax.legend()
plt.show()



# tSZ

def model_Pth_GNFW(z=0.55):
    # From MCMC
    P0    = 2.0
    x_ct  = 0.8
    b_t   = 2.6
    A_t2h = 0.7

    # Fixed in model
    g_t = -0.3
    a_t = 1

    rho_cr = cosmo.critical_density(z)
    M200 = 3e13*units.solMass
    R200 = Mdelta_to_Rdelta(M200, 200, z, rho_cr=rho_cr).to('Mpc')

    x = 10**np.linspace(-2,1.5)
    r = x*R200
    P_GNFW = P0*(x/x_ct)**g_t*(1+(x/x_ct)**a_t)**(-b_t)

    # rho2h, r2h = rho_2h(M200.value)
    # fit_rho2h  = lambda r: 10**interp1d(np.log10(r2h/h), np.log10(rho2h.value), 
    #                                     fill_value='extrapolate'
    #                                     )(np.log10(r.to('Mpc').value))*rho2h.unit

    P200 = constants.G*M200*200*rho_cr*cosmo.Ob(z)/cosmo.Om(z)/(2*R200) 
    Pth  = P_GNFW*P200
    # Pth = (rho_GNFW*rho_cr)*cosmo.Ob(z)/cosmo.Om(z)+A_k2h*fit_rho2h(r)*(1+z)**3
    return Pth, r, x

def model_Pe_GNFW(z=0.55):
    Pth, rr, xx = model_Pth_GNFW(z=z)
    X_H  = 0.76
    Pe   = (Pth*(2+2*X_H)/(3+5*X_H)).to('erg/cm^3')
    return Pe, rr


z = 0.55
z_min, z_max = 0.4, 0.7


Mv = 3e13  # CMASS halo mass in Amodeo+2020
Mc, mu, Tej, eta_star, eta_cga = 10**15.3, 0.4, 6, 0.32, 0.6
theta_bar = [Mc, mu, Tej, eta_star, eta_cga]

# Pe_bin, r_bin = model_Pe(Mc, mu, Tej, eta_star, eta_cga, Mv, A2h=1), rbin/h*units.Mpc
Pe_bin, r_bin = model_Pe_GNFW(z=0.55)

TtSZ_f90_thetad, TtSZ_f150_thetad, thetad_bins = ACT.get_TtSZ_thetad(Mv, z, 
                                                    z_min=z_min, z_max=z_max, 
                                                    n_bins=256, theta_nbin=1000*2, l_nbin=1000,
                                                    func_model_Pe=None, cosmo=cosmo,
                                                    Pe_bin=Pe_bin, r_bin=r_bin
                                                )

print('TtSZ_f90_thetad:', TtSZ_f90_thetad)

fig, ax = plt.subplots(figsize=(12,8))
# ax = plot_TtSZ_f90()
# ax = plot_TtSZ_f150(ax=ax)
ax.plot(thetad_bins, TtSZ_f90_thetad, c='k', ls='-.', lw=5, label='90GHz model')
ax.plot(thetad_bins, TtSZ_f150_thetad, c='g', ls=':', lw=5, label='150GHz model')
ax.set_yscale('symlog')
ax.legend()
plt.show()
