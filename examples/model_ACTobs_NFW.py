import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splrep, splev
from scipy.integrate import quad, simps
from scipy.signal import convolve2d, fftconvolve
from scipy.optimize import fsolve

import pySZe
import pySZe as bfc
from tqdm import tqdm
import pickle 

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
# par.baryon.thco = 0.1
par.code.eps = 1

Mv = 3e13*par.cosmo.h0 

TkSZmod = pySZe.deltaTkSZ(par)
TkSZmod.run(Mv, NFW=True)
dTkSZ_arcmins_fun = TkSZmod.dTkSZ_arcmins_fun

ACTmod = pySZe.ACT(par)
ACTmod.TkSZ(dTkSZ_arcmins_fun)
ACTmod.plot_Schaan2020()
plt.tight_layout()
plt.show()


# def comoving_dist(z,par):
# 	c  = 3e8/1000 #km/s
# 	H0 = 100*par.cosmo.h0 #km s^-1 Mpc^-1
# 	fn = lambda x: 1/np.sqrt(par.cosmo.Om*(1+x)**3+(1-par.cosmo.Om))
# 	dc = (c/H0)*quad(fn, 0, z)[0] # Mpc
# 	return dc 

# def angular_dist(z,par):
# 	dc = comoving_dist(z,par)
# 	return dc/(1+z)

# def cvir_fct(mvir,z):
#     """
#     Concentrations form Dutton+Maccio (2014)
#     c200 (200 times RHOC)
#     Assumes PLANCK coismology
#     """
#     A = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
#     B = -0.101 + 0.026*z
#     return 10.0**A*(mvir/1.0e12)**(B)


# def M200_fct(M500,c):
#     f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
#     y0 = fsolve(f,1.0)
#     return 2.0/5.0/y0**3.0*M500

# def B_f90(theta):
# 	'''
# 	Beam profiles: f90
# 	'''
# 	G_fwhm  = 2.1 # arcmin
# 	G_sigma = G_fwhm/2.355 #(2*np.sqrt(2*np.log(2)))
# 	print('f90', 2*np.pi*G_sigma**2)
# 	return np.exp(-theta**2/G_sigma**2/2)

# def B_f150(theta):
# 	'''
# 	Beam profiles: f150
# 	'''
# 	G_fwhm  = 1.3 # arcmin
# 	G_sigma = G_fwhm/2.355 #(2*np.sqrt(2*np.log(2)))
# 	print('f150', 2*np.pi*G_sigma**2)
# 	return np.exp(-theta**2/G_sigma**2/2)


# def W_thetad(theta, thetad):
# 	'''
# 	CAP filter
# 	'''
# 	# print(theta.size)
# 	if theta.size>1:
# 		out = np.zeros(theta.shape)
# 		out[theta<=np.sqrt(2)*thetad] = -1
# 		out[theta<thetad] = 1
# 		return out
# 	if theta<thetad: return 1
# 	if thetad<=theta and theta<=np.sqrt(2)*thetad: return -1
# 	return 0

# #radius bins
# N_rbin = 100
# rmax = 49.0
# rmin = 0.0005
# rbin = np.logspace(np.log10(rmin),np.log10(rmax),N_rbin,base=10)

# vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(par.files.cosmofct, usecols=(0,1,2,3), unpack=True)
# bias_tck = splrep(vc_m, vc_bias, s=0)
# corr_tck = splrep(vc_r, vc_corr, s=0)
# cosmo_corr = splev(rbin,corr_tck)

# #2-halo term
# bin_r, bin_m, bin_bias, bin_corr = bfc.cosmo(par)


# Mv = 3e13*par.cosmo.h0 # solmass

# cv = cvir_fct(Mv,par.cosmo.z)
# #Mv = M200_fct(Mv,cv)
# cosmo_bias = splev(Mv, bias_tck)
# fE, dE, mE = bfc.profiles(rbin,Mv,cv,cosmo_corr,cosmo_bias,par)

# dens_NFW = dE['NFW'] #+ dE['BG'] # h^2 Msun/Mpc^3
# dens_gas_NFW = dens_NFW*par.cosmo.Ob/par.cosmo.Om  # h^2 Msun/Mpc^3
# dens_gas_tck = splrep(np.log10(rbin/par.cosmo.h0), np.log10(dens_gas_NFW))
# # dens_gas_fun = lambda r: 10**splev(np.log10(r), dens_gas_tck)

# # dcom = comoving_dist(par.cosmo.z,par)
# dang = angular_dist(par.cosmo.z,par) #
# Xh = 0.76
# mp = 8.348e-58   # Msun
# mpc_to_cm = 3.086e24
# ne_NFW = (Xh+1)/2*dens_gas_NFW/mp / mpc_to_cm**3 * par.cosmo.h0**2 # 1/cm^3
# ne_gas_tck = splrep(np.log10(rbin/par.cosmo.h0), np.log10(ne_NFW))
# ne_gas_fun = lambda r: 10**splev(np.log10(r), ne_gas_tck)

# m_to_cm = 100
# sig_T   = 6.65e-29 * m_to_cm**2 # cm^2

# arcmins_to_rad = 1./60*np.pi/180

# theta_arcmins = 10**np.linspace(-2,1,100) #np.linspace(0.001,10,100)
# tau_arcmins   = np.zeros_like(theta_arcmins)
# for i in tqdm(range(theta_arcmins.size)): 
# 	th = theta_arcmins[i]*arcmins_to_rad
# 	# print(i,th)
# 	fn = lambda x: ne_gas_fun(x)/np.sqrt(x**2-dang**2*th**2)*x
# 	tau_arcmins[i] = 2*sig_T*quad(fn, np.abs(dang*th), 30)[0] * mpc_to_cm 


# Tcmb_muK   = 2.725*(1+par.cosmo.z) * 1e6 # muK
# dTkSZ_arcmins = tau_arcmins*1.06e-3*Tcmb_muK
# dTkSZ_arcmins_tck = splrep(np.log10(theta_arcmins), np.log10(dTkSZ_arcmins))
# dTkSZ_arcmins_fun = lambda x: 10**splev(np.log10(x), dTkSZ_arcmins_tck)


# d_theta = 0.01
# theta_max  = 5
# theta_uniq = np.arange(-theta_max,theta_max,d_theta)+d_theta/2
# theta_x, theta_y = np.meshgrid(theta_uniq,theta_uniq)
# theta_r = np.sqrt(theta_x**2+theta_y**2)

# dTkSZ_arcmins_grid = dTkSZ_arcmins_fun(theta_r)

# Bf90, Bf150 = B_f90(theta_r), B_f150(theta_r)
# dTkSZ_arcmins_grid_f90  = fftconvolve(dTkSZ_arcmins_grid,Bf90/Bf90.sum(),mode='same')
# dTkSZ_arcmins_grid_f150 = fftconvolve(dTkSZ_arcmins_grid,Bf150/Bf150.sum(),mode='same')
# # dTkSZ_arcmins_grid_f90  = fftconvolve(dTkSZ_arcmins_grid,Bf90/Bf90.max(),mode='same')
# # dTkSZ_arcmins_grid_f150 = fftconvolve(dTkSZ_arcmins_grid,Bf150/Bf150.max(),mode='same')
# print(Bf90.sum(),Bf150.sum())

# thetad_s = np.linspace(1,6,11) #10**np.linspace(np.log10(1),np.log10(6),11)
# dTkSZ_arcmins_grid_f90_thetad  = np.zeros_like(thetad_s)
# dTkSZ_arcmins_grid_f150_thetad = np.zeros_like(thetad_s)
# for i in tqdm(range(thetad_s.size)):
# 	thd = thetad_s[i]
# 	Wd = W_thetad(theta_r,thd)
# 	dTkSZ_arcmins_grid_f90_thetad[i] = simps(simps(dTkSZ_arcmins_grid_f90*Wd, theta_uniq), theta_uniq)
# 	dTkSZ_arcmins_grid_f150_thetad[i] = simps(simps(dTkSZ_arcmins_grid_f150*Wd, theta_uniq), theta_uniq)




# ACT_f90_mean = np.array([[1.053763440860215, 0.09412049672680661],
# 						[1.6774193548387095, 0.7690683028568883],
# 						[2.311827956989247, 1.7963293461795375],
# 						[2.9247311827956985, 2.2432475028984156],
# 						[3.559139784946236, 3.226799119945805],
# 						[4.18279569892473, 6.158482110660257],
# 						[4.806451612903225, 8.507942799627436],
# 						[5.43010752688172, 11.06266214330072],
# 						[6.053763440860214, 13.268047497147224]
# 						]).T
# ACT_f90_up = np.array([[1.053763440860215, 0.1623776739188721],
# 						[1.6774193548387095, 0.9223851039358476],
# 						[2.311827956989247, 2.111355692539677],
# 						[2.9247311827956985, 2.85851417968447],
# 						[3.548387096774193, 4.281332398719389],
# 						[4.18279569892473, 7.847599703514603],
# 						[4.806451612903225, 10.841458689358333],
# 						[5.43010752688172, 14.677992676220676],
# 						[6.053763440860214, 18.329807108324328]
# 						]).T
# ACT_f90_dn = np.array([[1.053763440860215, 0.029763514416313162],
# 						[1.6774193548387095, 0.6035340185482205],
# 						[2.311827956989247, 1.5283067326587672],
# 						[2.935483870967741, 1.6907141034735784],
# 						[3.559139784946236, 2.2432475028984156],
# 						[4.18279569892473, 4.548777947003771],
# 						[4.806451612903225, 6.0353401854821955],
# 						[5.43010752688172, 7.53690398089853],
# 						[6.053763440860214, 8.681534415584908]
# 						]).T
# ACT_f150_mean = np.array([[1.0000000000000002, 0.22890175519210493],
# 						[1.6344086021505375, 0.960408821250537],
# 						[2.258064516129032, 2.027764631912644],
# 						[2.8817204301075265, 3.1622776601683764],
# 						[3.505376344086021, 4.368686466212333],
# 						[4.118279569892472, 6.284136559286956],
# 						[4.75268817204301, 9.223851039358467],
# 						[5.376344086021504, 12.487952210263613],
# 						[5.999999999999999, 14.977474763451994]
# 						]).T
# ACT_f150_up = np.array([[1.0000000000000002, 0.291683780824842],
# 						[1.6344086021505375, 1.1288378916846873],
# 						[2.247311827956989, 2.335721469090119],
# 						[2.8817204301075265, 3.792690190732246],
# 						[3.494623655913978, 5.455594781168512],
# 						[4.118279569892472, 8.007718024241928],
# 						[4.75268817204301, 11.993539462092325],
# 						[5.387096774193547, 16.237767391887193],
# 						[6.010752688172041, 19.872184654880524]
# 						]).T
# ACT_f150_dn = np.array([[1.0107526881720432, 0.1656907430678826],
# 						[1.6236559139784945, 0.8171103315457195],
# 						[2.247311827956989, 1.7252105499420392],
# 						[2.8817204301075265, 2.532262781698792],
# 						[3.505376344086021, 3.3598182862837778],
# 						[4.129032258064515, 4.641588833612772],
# 						[4.75268817204301, 6.676692939187556],
# 						[5.376344086021504, 8.858667904100809],
# 						[5.999999999999999, 9.800045006276935]
# 						]).T

# NFW_f90 = np.array([[1.021505376344086, 1.9872184654880534],
# 					[1.6344086021505375, 6.5431891297129585],
# 					[2.258064516129032, 9.604088212505369],
# 					[3.0645161290322576, 11.06266214330072],
# 					[3.86021505376344, 11.753722651306344],
# 					[4.709677419354838, 11.993539462092325],
# 					[5.462365591397848, 12.487952210263613],
# 					[5.96774193548387, 12.487952210263613]
# 					]).T
# NFW_f150 = np.array([[1.0107526881720432, 4.931538819950567],
# 					[1.6451612903225805, 10.204034770855607],
# 					[2.258064516129032, 11.753722651306344],
# 					[3.032258064516128, 12.487952210263613],
# 					[3.720430107526881, 13.00274626174631],
# 					[4.612903225806451, 13.268047497147224],
# 					[5.311827956989246, 13.00274626174631],
# 					[5.999999999999999, 13.268047497147224]
# 					]).T


# fig, ax = plt.subplots(figsize=(8,6))
# ax.errorbar(ACT_f90_mean[0], ACT_f90_mean[1], yerr=[ACT_f90_up[1]-ACT_f90_mean[1],ACT_f90_mean[1]-ACT_f90_dn[1]], c='m', ms=15, ls=' ', alpha=0.99, label='90GHz', marker='.')
# ax.errorbar(ACT_f150_mean[0], ACT_f150_mean[1], yerr=[ACT_f150_up[1]-ACT_f150_mean[1],ACT_f150_mean[1]-ACT_f150_dn[1]], c='b', ms=15, ls=' ', alpha=0.99, label='150GHz', marker='.')
# ax.plot(NFW_f90[0], NFW_f90[1], c='m', ms=15, ls='--', alpha=0.99, label='NFW, 90GHz')
# ax.plot(NFW_f150[0], NFW_f150[1], c='b', ms=15, ls='--', alpha=0.99, label='NFW, 150GHz')
# ax.loglog(thetad_s, dTkSZ_arcmins_grid_f90_thetad, c='grey', ls='-.', lw=3, label='NFW, 90GHz model')
# ax.loglog(thetad_s, dTkSZ_arcmins_grid_f150_thetad, c='grey', ls=':', lw=3, label='NFW, 150GHz model')
# #ax.axis([0.5,6.5,0.03,70])
# ax.set_xscale('linear')
# ax.set_yscale('log')
# ax.set_ylabel('$T_\mathrm{kSZ}$ [$\mu K~\mathrm{arcmin}^2$]')
# ax.set_xlabel('$R$ [arcmin]')
# ax.legend()
# plt.tight_layout()
# plt.show()





