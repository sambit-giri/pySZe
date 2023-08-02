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

ax.semilogy(ACTmod.thetad_s, ACTmod.dTkSZ_arcmins_grid_f90_thetad , c='C1', ls='-.', lw=3, label='Best-fit baryonic feedback model, 90GHz model')
ax.semilogy(ACTmod.thetad_s, ACTmod.dTkSZ_arcmins_grid_f150_thetad , c='C1', ls=':', lw=3, label='Best-fit baryonic feedback model, 150GHz model')
ax.axis([0.5,6.5,0.03,70])

ax.legend()
plt.tight_layout()
plt.show()

