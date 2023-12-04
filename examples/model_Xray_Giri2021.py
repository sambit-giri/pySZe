import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 

import pySZe
from scipy.interpolate import splrep,splev

from matplotlib import rcParams, ticker, cm
rcParams['font.family'] = 'sans-serif'
#rcParams['font.family'] = 'small-caps'
#rcParams['text.usetex'] = True
rcParams['axes.labelsize']  = 12
rcParams['font.size']       = 12 
rcParams['axes.linewidth']  = 1.6

Xray_gas = {
    'f_mean': np.array([[1.766e+13, 7.410e-2],
                        [5.640e+13, 7.998e-2],
                        [1.773e+14, 1.085e-1],
                        [5.618e+14, 1.320e-1],
                        ]),
    'f_upp':  np.array([[1.766e+13, 9.550e-2],
                        [5.596e+13, 9.802e-2],
                        [1.759e+14, 1.345e-1],
                        [5.618e+14, 1.564e-1],
                        ]),
    'f_dow':  np.array([[1.752e+13, 5.312e-2],
                        [5.596e+13, 6.109e-2],
                        [1.759e+14, 8.333e-2],
                        [5.618e+14, 1.064e-1],
                        ]),
    'M_dow':  np.array([[9.922e+12, 7.452e-2],
                        [3.168e+13, 7.998e-2],
                        [9.961e+13, 1.089e-1],
                        [3.181e+14, 1.316e-1],
                        ]),
    'M_upp':  np.array([[3.162e+13, 7.452e-2],
                        [9.923e+13, 7.956e-2],
                        [3.138e+14, 1.089e-1],
                        [9.923e+14, 1.316e-1],
                        ]),
    }

Xray_star = {
    'f_mean': np.array([[5.299e+13, 2.668e-2],
                        [3.746e+14, 1.661e-2],
                        ]),
    'f_upp':  np.array([[5.299e+13, 3.423e-2],
                        [3.776e+14, 2.248e-2],
                        ]),
    'f_dow':  np.array([[5.258e+13, 1.829e-2],
                        [3.746e+14, 1.073e-2],
                        ]),
    'M_upp':  np.array([[1.393e+14, 2.668e-2],
                        [9.922e+14, 1.619e-2],
                        ]),
    'M_dow':  np.array([[1.985e+13, 2.626e-2],
                        [1.403e+14, 1.619e-2],
                        ]),
    }

S19_profile_data = {
    'DMO': np.array([
                    [7.000e-3, 3.491e+12],
                    [1.065e-2, 4.984e+12],
                    [1.604e-2, 6.928e+12],
                    [2.966e-2, 1.049e+13],
                    [5.559e-2, 1.406e+13],
                    [1.082e-1, 1.575e+13],
                    [1.882e-1, 1.473e+13],
                    [4.142e-1, 1.042e+13],
                    [9.036e-1, 6.155e+12],
                    [1.407e+0, 4.464e+12],
                    [2.210e+0, 3.855e+12],
                    [3.737e+0, 4.979e+12],
                    [6.219e+0, 8.466e+12],
                    [1.045e+1, 1.635e+13],
                    ]),
    'beta=0,thej=4': np.array([
                    [6.414e-3, 5.217e+12],
                    [1.163e-2, 6.519e+12],
                    [2.086e-2, 7.646e+12],
                    [3.338e-2, 8.594e+12],
                    [7.354e-2, 1.133e+13],
                    [1.247e-1, 1.214e+13],
                    [3.332e-1, 9.630e+12],
                    [6.023e-1, 7.165e+12],
                    [1.127e+0, 5.220e+12],
                    [2.231e+0, 4.341e+12],
                    [4.328e+0, 5.812e+12],
                    [8.626e+0, 1.246e+13],
                    ]),
    'beta=3,thej=4': np.array([
                    [6.635e-3, 5.385e+12],
                    [1.395e-2, 7.326e+12],
                    [3.072e-2, 9.966e+12],
                    [5.961e-2, 1.341e+13],
                    [1.214e-1, 1.532e+13],
                    [2.889e-1, 1.248e+13],
                    [5.349e-1, 8.900e+12],
                    [9.919e-1, 5.680e+12],
                    [2.185e+0, 3.803e+12],
                    [5.370e+0, 7.033e+12],
                    [9.657e+0, 1.430e+13],
                    ]),
    }

Mgas_mean, fgas_mean = Xray_gas['f_mean'][:,0], Xray_gas['f_mean'][:,1]
fgas_std = np.vstack(((Xray_gas['f_upp'][:,1]-Xray_gas['f_mean'][:,1]),
                      (Xray_gas['f_mean'][:,1]-Xray_gas['f_dow'][:,1]))).min(axis=0)
Mgas_std = np.vstack(((Xray_gas['M_upp'][:,0]-Xray_gas['f_mean'][:,0]),
                      (Xray_gas['f_mean'][:,0]-Xray_gas['M_dow'][:,0]))).min(axis=0)

Mstar_mean, fstar_mean = Xray_star['f_mean'][:,0], Xray_star['f_mean'][:,1]
fstar_std = np.vstack(((Xray_star['f_upp'][:,1]-Xray_star['f_mean'][:,1]),
                      (Xray_star['f_mean'][:,1]-Xray_star['f_dow'][:,1]))).min(axis=0)
Mstar_std = np.vstack(((Xray_star['M_upp'][:,0]-Xray_star['f_mean'][:,0]),
                      (Xray_star['f_mean'][:,0]-Xray_star['M_dow'][:,0]))).min(axis=0)

# M500_hse = lambda bhse,M500_true: (1 - bhse)*M500_true

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

def mass_profile(Mv, par):
    N_rbin = 100
    rmax = 49.0
    rmin = 0.0005
    rbin = np.logspace(np.log10(rmin),np.log10(rmax),N_rbin,base=10)

    vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(par.files.cosmofct, usecols=(0,1,2,3), unpack=True)
    bias_tck = splrep(vc_m, vc_bias, s=0)
    corr_tck = splrep(vc_r, vc_corr, s=0)
    cosmo_corr = splev(rbin,corr_tck)

    #2-halo term
    # bin_r, bin_m, bin_bias, bin_corr = bfc.cosmo(self.par)
    bin_r, bin_m, bin_bias, bin_corr = pySZe.cosmo(par)

    cv = pySZe.cvir_fct(Mv,par.cosmo.z)
    cosmo_bias = splev(Mv, bias_tck)

    ff, dd, mm = pySZe.profiles(rbin,Mv,cv,cosmo_corr,cosmo_bias,par)
    rho2h, rhoDMO, rhoACM, rhoCGA, rhoHGA, rhoDMB = dd['BG'], dd['DMO'], dd['ACM'], dd['CGA'], dd['HGA'], dd['DMB']
    M2h, MDMO, MACM, MCGA, MHGA, MDMB = mm['BG'], mm['DMO'], mm['ACM'], mm['CGA'], mm['HGA'], mm['DMB']
    fcdm, fcga, fsga, fhga = ff['CDM'], ff['CGA'], ff['SGA'], ff['HGA']

    profile_gas = {
            'r': rbin,
            'd': fhga*rhoHGA + fcga*rhoCGA, #fcdm*rhoACM
            'm': fhga*MHGA + fcga*MCGA, #fcdm*MACM
        }
    profile_star = {
            'r': rbin,
            'd': fsga*rhoACM,
            'm': fsga*MACM,
        }
    profile_dmo = {
            'r': rbin,
            'd': rhoDMO + rho2h,
            'm': MDMO + M2h,
        }
    return {'gas': profile_gas, 'star': profile_star, 'DMO': profile_dmo,
            'DMB': {'r':rbin, 'd': rhoDMB, 'm': mm['DMB']} }

def mass_fraction(M200, par):
    c200 = pySZe.cvir_fct(M200,par)
    M500 = pySZe.M500_fct(M200,c200)
    profiles = mass_profile(M200,par)
    MDMB_tck = splrep(profiles['DMB']['r'], profiles['DMB']['m'], s=0, k=3)
    func = lambda r: splev(r, MDMB_tck) - M500
    r500 = fsolve(func, profiles['DMB']['r'].min())
    Mgas_tck = splrep(profiles['gas']['r'], profiles['gas']['m'], s=0, k=3)
    Mstar_tck = splrep(profiles['star']['r'], profiles['star']['m'], s=0, k=3)
    fgas  = splev(r500, Mgas_tck)/splev(r500, MDMB_tck)
    fstar = splev(r500, Mstar_tck)/splev(r500, MDMB_tck)
    return M500, fgas, fstar 

def mass_fraction_obs(par, bhse=0.30):
    M200_list = 10**np.linspace(12.5,15.5,15)
    M500_list, fgas_list, fstar_list = np.array([]), np.array([]), np.array([])
    for M200 in M200_list:
        M500, fgas, fstar = mass_fraction(M200, par)
        M500_list = np.append(M500_list,M500)
        fgas_list = np.append(fgas_list,fgas)
        fstar_list = np.append(fstar_list,fstar)
    return (1-bhse)*M500_list, fgas_list, fstar_list


# fig, ax = plt.subplots(1,1,figsize=(8,6))
# Mv = 1e14 
# par.code.beta_model = 0.0
# par.baryon.beta = 0.0
# profiles1 = mass_profile(Mv,par)
# par.baryon.beta = 3.0
# profiles2 = mass_profile(Mv,par)
# rbin = profiles1['gas']['r']
# rhoDMO = profiles1['DMO']['d']
# # ax.plot(rbin, rbin**2*(rhoDMO),
# #         color='black', alpha=0.4)
# ax.plot(rbin, rbin**2*(profiles1['gas']['d']+profiles1['star']['d']),
#         color='blue', alpha=0.4)
# ax.plot(S19_profile_data['DMO'][:,0], S19_profile_data['DMO'][:,1],
#            color='black', ls='--', lw=1, marker='o', label=r'DMO')
# ax.plot(S19_profile_data['beta=0,thej=4'][:,0], S19_profile_data['beta=0,thej=4'][:,1],
#            color='blue', ls='--', lw=1, marker='o', label=r'DMB ($\beta=0,\theta_{\rm ej}=4$)')
# ax.plot(S19_profile_data['beta=3,thej=4'][:,0], S19_profile_data['beta=3,thej=4'][:,1],
#            color='cyan', ls='--', lw=1, marker='o', label=r'DMB ($\beta=3,\theta_{\rm ej}=4$)')
# ax.axis([6e-3,10,3.5e12,2e13])
# ax.set_xlabel(r'$r$ [$h^{-1}$Mpc]', fontsize=15)
# ax.set_ylabel(r'$r^2\rho(r)$ [M$_\odot$/Mpc]', fontsize=15)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()
# plt.tight_layout()
# plt.show()

# if __name__ == '__main__':
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.errorbar(Mgas_mean, fgas_mean, xerr=Mgas_std, yerr=fgas_std, 
            label='gas', ls='', marker='o')
ax.errorbar(Mstar_mean, fstar_mean, xerr=Mstar_std, yerr=fstar_std, 
            label='star', ls='', marker='o')
par.baryon.Mc = 10**13
M500_arr, fgas_arr, fstar_arr = mass_fraction_obs(par, bhse=0.30)
ax.plot(M500_arr, fgas_arr, c='C0')
ax.plot(M500_arr, fstar_arr, c='C1')
par.baryon.Mc = 10**15
M500_arr, fgas_arr, fstar_arr = mass_fraction_obs(par, bhse=0.30)
ax.plot(M500_arr, fgas_arr, c='C0')
ax.plot(M500_arr, fstar_arr, c='C1')
ax.set_xlabel(r'$M_\mathrm{500}$ [$h^{-1}$M$_{\odot}$]', fontsize=15)
ax.set_ylabel(r'$f_\mathrm{gas,star}$', fontsize=15)
ax.set_xscale('log')
plt.tight_layout()
plt.show()


