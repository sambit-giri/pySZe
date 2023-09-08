import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

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
rcParams['axes.labelsize']  = 12
rcParams['font.size']       = 12 
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

### Varying Mc parameter
log10Mc_list = np.linspace(12,16,20)

fig = plt.figure(figsize=(13, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], 
        left=0.07, right=0.93, top=0.98, bottom=0.11, hspace=0.2, wspace=0.15)
axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

# Add the colorbar
cbar_ax = plt.subplot(gs[2])
cmap = cm.summer
norm = colors.Normalize(vmin=log10Mc_list.min(), vmax=log10Mc_list.max())
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
scalar_map.set_array([])
cbar = fig.colorbar(scalar_map, cax=cbar_ax)
cbar.set_label(r'$\log_{10}\left(\frac{M_c}{h^{-1}M_\odot}\right)$', rotation=270, labelpad=25)
colors = cmap(np.linspace(0, 1, len(log10Mc_list)))

TkSZmod = pySZe.deltaTkSZ(par)
TkSZmod.run(Mv, NFW=True)
dTkSZ_arcmins_fun = TkSZmod.dTkSZ_arcmins_fun
ACTmod = pySZe.ACT(par)
ACTmod.TkSZ(dTkSZ_arcmins_fun)
ACTmod.plot_Schaan2020(ax=axs)

TkSZmod_f90 = {'1':np.array([]), '2':np.array([]), '3':np.array([])}
for ii,log10Mc in enumerate(log10Mc_list):
    par.baryon.Mc = 10**log10Mc
    TkSZmod = pySZe.deltaTkSZ(par)
    TkSZmod.run(Mv, NFW=False)
    dTkSZ_arcmins_fun = TkSZmod.dTkSZ_arcmins_fun
    ACTmod = pySZe.ACT(par)
    ACTmod.TkSZ(dTkSZ_arcmins_fun)
    axs[0].semilogy(ACTmod.thetad_s, ACTmod.dTkSZ_arcmins_grid_f90_thetad, ls='-.', lw=3 , color=colors[ii])
    axs[1].semilogy(ACTmod.thetad_s, ACTmod.dTkSZ_arcmins_grid_f150_thetad, ls=':', lw=3, color=colors[ii])
    TkSZmod_f90['1'] = np.append(TkSZmod_f90['1'],ACTmod.dTkSZ_arcmins_grid_f90_thetad[np.abs(ACTmod.thetad_s-1).argmin()])
    TkSZmod_f90['2'] = np.append(TkSZmod_f90['1'],ACTmod.dTkSZ_arcmins_grid_f90_thetad[np.abs(ACTmod.thetad_s-2).argmin()])
    TkSZmod_f90['3'] = np.append(TkSZmod_f90['1'],ACTmod.dTkSZ_arcmins_grid_f90_thetad[np.abs(ACTmod.thetad_s-3).argmin()])
axs[0].set_ylabel(r'$T_\mathrm{kSZ}$ [$\mu K~\mathrm{arcmin}^2$]', fontsize=15)
axs[0].set_xlabel(r'$R$ [arcmin]', fontsize=15)
axs[1].set_ylabel('', fontsize=15)
axs[1].set_xlabel(r'$R$ [arcmin]', fontsize=15)
plt.show()

