import numpy as np
import matplotlib.pyplot as plt

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


par = pySZe.par()

# COSMO PARAMETERS ################################################

par.cosmo.Ob = 0.044
par.cosmo.Om = 0.25
par.cosmo.h0 = 0.7

par.code.verbose = False

#proton number density in 1/cm^3
Msun_Mpc3_to_1_cm3 = (1.189e57/2.939e+73)
fproton = par.cosmo.Ob/par.cosmo.Om/2.0
fbar = par.cosmo.Ob/par.cosmo.Om

print("Cosmic baryon fraction = ", fbar)

# PARAMETERS #######################################################

par.files.transfct = "CDM_PLANCK_tk.dat"
par.files.cosmofct = "cosmofct.dat"
par.cosmo.z = 0.55
par.code.eps = 1

Mv = 3e13*par.cosmo.h0 

TkSZmod0 = pySZe.deltaTkSZ(par)
TkSZmod0.run(Mv, NFW=True)
dTkSZ_arcmins_fun0 = TkSZmod0.dTkSZ_arcmins_fun

TkSZmod1 = pySZe.deltaTkSZ(par)
TkSZmod1.run(Mv, NFW=False)
dTkSZ_arcmins_fun1 = TkSZmod1.dTkSZ_arcmins_fun

dTkSZ_arcmins_fun2 = np.vectorize(lambda x: 1)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(TkSZmod0.theta_arcmins, dTkSZ_arcmins_fun0(TkSZmod0.theta_arcmins),
            c='C0', ls='-',  label='NFW')
ax.plot(TkSZmod1.theta_arcmins, dTkSZ_arcmins_fun1(TkSZmod1.theta_arcmins),
            c='C1', ls='--', label='BCM')
ax.plot(TkSZmod1.theta_arcmins, dTkSZ_arcmins_fun2(TkSZmod1.theta_arcmins),
            c='C2', ls='-.', label='Ones')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.show()

ACTmod0 = pySZe.ACT(par)
ACTmod0.TkSZ(dTkSZ_arcmins_fun0)
ACTmod1 = pySZe.ACT(par)
ACTmod1.TkSZ(dTkSZ_arcmins_fun1)
ACTmod2 = pySZe.ACT(par)
ACTmod2.TkSZ(dTkSZ_arcmins_fun2)

fig, axs = plt.subplots(1,2,figsize=(14,6))
axs[0].set_title('90 GHz')
axs[1].set_title('150 GHz')
axs[0].semilogy(ACTmod0.thetad_s, ACTmod0.dTkSZ_arcmins_grid_f90_thetad , 
                c='C0', ls='-.', lw=3, label='NFW')
axs[1].semilogy(ACTmod0.thetad_s, ACTmod0.dTkSZ_arcmins_grid_f150_thetad , 
                c='C0', ls=':', lw=3, label='NFW')
axs[0].semilogy(ACTmod1.thetad_s, ACTmod1.dTkSZ_arcmins_grid_f90_thetad , 
                c='C1', ls='-.', lw=3, label='BCM')
axs[1].semilogy(ACTmod1.thetad_s, ACTmod1.dTkSZ_arcmins_grid_f150_thetad , 
                c='C1', ls=':', lw=3, label='BCM')
axs[0].semilogy(ACTmod2.thetad_s, ACTmod2.dTkSZ_arcmins_grid_f90_thetad , 
                c='C2', ls='-.', lw=3, label='Ones')
axs[1].semilogy(ACTmod2.thetad_s, ACTmod2.dTkSZ_arcmins_grid_f150_thetad , 
                c='C2', ls=':', lw=3, label='Ones')
for ax in axs: ax.axis([0.5,6.5,0.03,70])
ax.legend()
ACTmod = pySZe.ACT(par)
ACTmod.plot_Schaan2020(ax=axs)
plt.tight_layout()
plt.show()


