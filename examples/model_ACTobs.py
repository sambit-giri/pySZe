import numpy as np
import matplotlib.pyplot as plt

import baryonification as bfc
from tqdm import tqdm
import pickle 

from scipy.interpolate import splrep, splev, griddata
from scipy.optimize import fsolve
from astropy.cosmology import Planck15 as cosmo
from astropy import units, constants 

import pySZe

from matplotlib import rcParams, ticker, cm
rcParams['font.family'] = 'sans-serif'
#rcParams['font.family'] = 'small-caps'
#rcParams['text.usetex'] = True
rcParams['axes.labelsize']  = 13
rcParams['font.size']       = 13 
rcParams['axes.linewidth']  = 1.6


par = bfc.par()

# COSMO PARAMETERS ################################################

Deltavir     = 200.0      #overdensity with respect to rhoc
par.cosmo.Ob = 0.044
par.cosmo.Om = 0.25
par.cosmo.h0 = 0.7
rhoc         = 2.776e11

#proton number density in 1/cm^3
Msun_Mpc3_to_1_cm3 = (1.189e57/2.939e+73)
fproton = par.cosmo.Ob/par.cosmo.Om/2.0
fbar = par.cosmo.Ob/par.cosmo.Om

print("Cosmic baryon fraction = ", fbar)

# PARAMETERS #######################################################

par.files.transfct = "CDM_PLANCK_tk.dat"
par.files.cosmofct = "cosmofct.dat"
par.cosmo.z = 0.55
par.baryon.thco = 0.1
par.code.eps = 1

Mv = 3e13*par.cosmo.h0 

TkSZmod = pySZe.deltaTkSZ(par)
TkSZmod.run(Mv, NFW=True)
dTkSZ_arcmins_fun = TkSZmod.dTkSZ_arcmins_fun
ACTmod = pySZe.ACT(par)
ACTmod.TkSZ(dTkSZ_arcmins_fun)

fig, ax = plt.subplots(figsize=(8,6))
ACTmod.plot_Schaan2020(ax=ax)

TkSZmod = pySZe.deltaTkSZ(par)
TkSZmod.run(Mv, NFW=False)
dTkSZ_arcmins_fun = TkSZmod.dTkSZ_arcmins_fun
ACTmod = pySZe.ACT(par)
ACTmod.TkSZ(dTkSZ_arcmins_fun)

ax.semilogy(ACTmod.thetad_s, ACTmod.dTkSZ_arcmins_grid_f90_thetad , c='C1', ls='-.', lw=3, label='NFW, 90GHz model')
ax.semilogy(ACTmod.thetad_s, ACTmod.dTkSZ_arcmins_grid_f150_thetad , c='C1', ls=':', lw=3, label='NFW, 150GHz model')

ax.axis([0.5,6.5,0.03,70])
plt.tight_layout()
plt.show()

exit()


from pySZe import ACT

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


####################################################################
# READ COSMO.DAT
####################################################################

vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(par.files.cosmofct, usecols=(0,1,2,3), unpack=True)
bias_tck = splrep(vc_m, vc_bias, s=0)
corr_tck = splrep(vc_r, vc_corr, s=0)

cosmo_corr = splev(rbin,corr_tck)


# Model gas profile

def model_rho(Mc, mu, thej, gamma, delta, eta, deta, Mv, A2h=1):
    par = bfc.par()
    par.code.beta_model = 1   # 0: old model from Schneider+18 1: new model
    par.baryon.Mc    = Mc     # beta(M,z): critical mass scale
    par.baryon.mu    = mu     # beta(M,z): critical mass scale
    par.baryon.nu    = 0      # beta(M,c): redshift dependence
    par.baryon.thej  = thej   # ejection factor thej=rej/rvir
    par.baryon.thco  = 0.1    # core factor thco=rco/rvir
    par.baryon.alpha = 1.0    # index in gas profile [default: 1.0]
    par.baryon.gamma = gamma  # index in gas profile [default: 2.0]
    par.baryon.delta = delta  # index in gas profile [default: 7.0 -> same asympt. behav. than NFWtrunc profile]  
    par.baryon.eta   = eta    # Mstar/Mvir~(10**11.435/Mvir)**-eta, total stellar fraction (cga + satellites)
    par.baryon.deta  = deta   # Mstar/Mvir~(10**11.435/Mvir)**-(eta+deta)  cga stellar fraction (cga = central galaxy)

    par.files.transfct = "CDM_PLANCK_tk.dat"
    par.files.cosmofct = "cosmofct.dat"
    vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(par.files.cosmofct, usecols=(0,1,2,3), unpack=True)
    bias_tck = splrep(vc_m, vc_bias, s=0)
    corr_tck = splrep(vc_r, vc_corr, s=0)
    cosmo_corr = splev(rbin,corr_tck)

    # cv = model_SZ.cvE
    # Mv = model_SZ.MvE
    cv = cvir_fct(Mv)
    cosmo_bias = splev(Mv, bias_tck)
    fE, dE, mE = bfc.profiles(rbin,Mv,cv,cosmo_corr,cosmo_bias,par) #bfc.profiles(rbin,Mv,cv,Mc,mu,Tej,cosmo_corr,cosmo_bias,par)
    frgasE = (fbar - fE['CGA'] - fE['SGA'])/fbar
    # rhoE = dE['HGA']*frgasE*model_SZ.fproton*model_SZ.Msun_Mpc3_to_1_cm3
    rhoE = (dE['HGA']+A2h*dE['BG'])*frgasE*Msun_Mpc3_to_1_cm3*fproton
    rhoE = (rhoE/units.cm**3*constants.m_p).to('g/cm^3') 
    return rhoE

def model_ne(Mc, mu, thej, gamma, delta, eta, deta, Mv, A2h=1):
    rhoE = model_rho(Mc, mu, thej, gamma, delta, eta, deta, Mv, A2h=A2h)
    X_H  = 0.76
    ne   = (rhoE*(X_H+1)/2/constants.m_p).to('1/cm^3')
    return ne


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
		ax.loglog(thetad_bins, TkSZ_thetad, c=cr, ls=ls, lw=lw, label=label)

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
		ax.loglog(thetad_bins, TkSZ_thetad, c=cr, ls=ls, lw=lw, label=label)

	ax.legend()
	ax.set_yscale('log')
	#ax.yaxis.set_major_formatter(ScalarFormatter())
	return ax


z = 0.55
z_min, z_max = 0.4, 0.7

# rr = 10*np.linspace(-1.5,1)*units.Mpc

Mv = 7.3e14
Mc, mu, thej, eta, deta = 10**15.3, 0.4, 6, 0.32, 0.3
gamma, delta = 3, 7
theta_bar = [Mc, mu, thej, eta, deta]

ne_bin, r_bin = model_ne(Mc, mu, thej, gamma, delta, eta, deta, Mv, A2h=1), rbin/h*units.Mpc

TkSZ_f90_thetad, TkSZ_f150_thetad, thetad_bins = ACT.get_TkSZ_thetad(Mv, z, 
                                                    z_min=z_min, z_max=z_max, 
                                                    n_bins=256, theta_nbin=1000, l_nbin=1000,
                                                    func_model_ne=None, cosmo=cosmo,
                                                    ne_bin=ne_bin, r_bin=r_bin
                                                )

ax = plot_TkSZ_f90()
ax = plot_TkSZ_f150(ax=ax)

ax.loglog(thetad_bins, TkSZ_f90_thetad, c='k', ls='-.', lw=5, label='90GHz model')
ax.loglog(thetad_bins, TkSZ_f150_thetad, c='g', ls=':', lw=5, label='150GHz model')

ax.legend()
plt.show()


Mv = 3e13  # CMASS halo mass in Amodeo+2020
Mc, mu, thej, eta, deta = 10**15.3, 0.4, 6, 0.32, 0.3
gamma, delta = 3, 7
theta_bar = [Mc, mu, thej, eta, deta]

ne_bin, r_bin = model_ne(Mc, mu, thej, gamma, delta, eta, deta, Mv, A2h=1), rbin/h*units.Mpc

TkSZ_f90_thetad, TkSZ_f150_thetad, thetad_bins = ACT.get_TkSZ_thetad(Mv, z, 
                                                    z_min=z_min, z_max=z_max, 
                                                    n_bins=256, theta_nbin=1000, l_nbin=1000,
                                                    func_model_ne=None, cosmo=cosmo,
                                                    ne_bin=ne_bin, r_bin=r_bin
                                                )

ax = plot_TkSZ_f90()
ax = plot_TkSZ_f150(ax=ax)

ax.loglog(thetad_bins, TkSZ_f90_thetad, c='k', ls='-.', lw=5, label='90GHz model')
ax.loglog(thetad_bins, TkSZ_f150_thetad, c='g', ls=':', lw=5, label='150GHz model')

ax.legend()
plt.show()



Mv = 1e13  # CMASS halo mass in Amodeo+2020
Mc, mu, thej, eta, deta = 10**14, 0.4, 4, 0.32, 0.3
gamma, delta = 2.5, 7
theta_bar = [Mc, mu, thej, eta, deta]

ne_bin, r_bin = model_ne(Mc, mu, thej, gamma, delta, eta, deta, Mv, A2h=1), rbin/h*units.Mpc

TkSZ_f90_thetad, TkSZ_f150_thetad, thetad_bins = ACT.get_TkSZ_thetad(Mv, z, 
                                                    z_min=z_min, z_max=z_max, 
                                                    n_bins=256, theta_nbin=1000, l_nbin=1000,
                                                    func_model_ne=None, cosmo=cosmo,
                                                    ne_bin=ne_bin, r_bin=r_bin
                                                )

ax = plot_TkSZ_f90()
ax = plot_TkSZ_f150(ax=ax)

ax.loglog(thetad_bins, TkSZ_f90_thetad, c='k', ls='-.', lw=5, label='90GHz model')
ax.loglog(thetad_bins, TkSZ_f150_thetad, c='g', ls=':', lw=5, label='150GHz model')

ax.legend()
plt.show()
