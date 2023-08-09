import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

import camb
from astropy import units
import pySZe

par = pySZe.par()

par.cosmo.Ob = 0.049
par.cosmo.Om = 0.315
par.cosmo.h0 = 0.673
par.cosmo.As = 2.1e-9
par.cosmo.ns = 0.963

# Set cosmological parameters in camb
params = camb.CAMBparams()
params.set_cosmology(
	                H0=par.cosmo.h0*100, 
		            ombh2=par.cosmo.Ob*par.cosmo.h0**2, 
		            omch2=par.cosmo.Om*par.cosmo.h0**2,
		            )
params.set_dark_energy(w=-1.0)
params.InitPower.set_params(As=par.cosmo.As, ns=par.cosmo.ns)
params.set_matter_power(redshifts=[0.], kmax=2.0)

# Compute results
results = camb.get_results(params)
zs = 10**np.linspace(-3,3,100)
background_evol = results.get_background_redshift_evolution(zs)
xs = background_evol['x_e']

ze, xe = zs.copy(), xs.copy()
xe[xe>1] = 1

# Plot ionization history
plt.figure()
plt.loglog(zs, xs)
plt.loglog(ze, xe, '--')
plt.xlabel('Redshift (z)')
plt.ylabel('Ionization Fraction (x_e)')
plt.title('Ionization History from CAMB')
plt.grid()
plt.show()

stochGW = pySZe.gertsenshtein_effect.StochasticGW(xe={'x_e': xe, 'z_e': ze})

plt.figure()
plt.loglog(ze, stochGW.Idash_zini(ze), label='$I\'(z_\mathrm{ini})$')
plt.loglog(ze, stochGW.I_zini(ze), label='$I(z_\mathrm{ini})$')
plt.xlabel('Redshift (z)')
plt.ylabel('$I (z_\mathrm{ini})$')
plt.legend()
plt.grid()
plt.show()

G_to_nG = (units.G).to('nG')
B0_list = np.logspace(-18.22,-7.4,100)  # in G
lam0_B_list = np.logspace(-6.22,4.4,90) # in Mpc

zz = [20,1100]
P = {z_ini: stochGW.conversion_probability(B0=B0_list[:,None]*G_to_nG, 
				    w0=56.78*2*np.pi, lam0_B=lam0_B_list, z_ini=z_ini)
		    for z_ini in zz}

cmap = 'plasma_r' #'plasma' 
lstyle, contour_color = 'dashed', 'white'

format_tick_raise_to_exponent_d  = lambda x, pos: r'$10^{{{}}}$'.format(int(x))
format_tick_raise_to_exponent_1f = lambda x, pos: r'$10^{{{:.1f}}}$'.format(x)

# fig = plt.figure(figsize=(11, 5))
# gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05],
# 			left=0.09, right=0.91, top=0.96, bottom=0.13, hspace=0.2, wspace=0.06)  
# axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
# for ii,ax in enumerate(axs):
# 	zi = zz[ii]
# 	Pi = P[zi]
# 	pcm = ax.pcolormesh(np.log10(lam0_B_list), np.log10(B0_list), np.log10(Pi), cmap=cmap)
# 	# for contour_level in [100,500,1500,5000,15000,50000]: 
# 	# 	cs = ax.contour(Z, K, l_kz(10**Z, 10**K), levels=[contour_level], colors=contour_color, linewidths=2, linestyles=lstyle)
# 	# 	ax.clabel(cs, inline=True, fmt=r'$l=%d$' % contour_level, fontsize=12, inline_spacing=10)
# 	ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_tick_raise_to_exponent_d))
# 	ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick_raise_to_exponent_d))
# 	ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
# 	ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
# 	ax.tick_params(axis='both', which='major', labelsize=13)
# 	ax.tick_params(axis='both', which='minor', labelsize=8)
# 	ax.set_xlabel('coherence length $\lambda^0_\mathrm{B}$ [Mpc]', fontsize=18)
# 	if ii==0: ax.set_ylabel('magnetic field $B_0$ [G]', fontsize=18)
# 	else: ax.yaxis.set_ticks([]) 
# cax = plt.subplot(gs[2])
# cbar = plt.colorbar(pcm, cax=cax)
# cbar_label = r'log$_\mathrm{10}\left[\left(\frac{T_0)}{\omega_0}\right)^2\mathcal{P}\right]$'
# cbar.set_label(label=cbar_label, fontsize=18)
# cbar.ax.tick_params(labelsize=13)
# # plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(1,1,figsize=(8, 6))
pcm = ax.pcolormesh(np.log10(lam0_B_list), np.log10(B0_list), np.log10(P[zz[0]]), cmap=cmap)
for contour_level in np.arange(-37,-10,3): 
	cs1 = ax.contour(np.log10(lam0_B_list), np.log10(B0_list), np.log10(P[20]), levels=[contour_level], colors=contour_color, linewidths=2, linestyles='-')
	cs2 = ax.contour(np.log10(lam0_B_list), np.log10(B0_list), np.log10(P[1100]), levels=[contour_level], colors=contour_color, linewidths=2, linestyles='--')
	ax.clabel(cs1, inline=True, fmt=r'$10^{%d}$'%contour_level, fontsize=10, inline_spacing=10)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_tick_raise_to_exponent_d))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick_raise_to_exponent_d))
ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_xlabel('coherence length $\lambda^0_\mathrm{B}$ [Mpc]', fontsize=18)
ax.set_ylabel('magnetic field $B_0$ [G]', fontsize=18)
cbar = plt.colorbar(pcm)
cbar_label = r'log$_\mathrm{10}\left[\left(\frac{T_0}{\omega_0}\right)^2\mathcal{P}\right]$'
cbar.set_label(label=cbar_label, fontsize=18)
cbar.ax.tick_params(labelsize=13)
plt.subplots_adjust(left=0.12, bottom=0.11, right=0.97, top=0.97, hspace=0.5, wspace=0.5)
# plt.tight_layout()
plt.show()

Hz_to_GHz = (units.Hz).to('GHz')
f_list = np.logspace(np.log10(5e5),np.log10(6e10)) #Hz

fig, ax = plt.subplots(1,1,figsize=(8, 6))
ax.loglog(f_list, stochGW.gw_overdensity_to_characteristic_strain(par.cosmo.Og,f=f_list*Hz_to_GHz), 
	    c='k', ls=':')
ax.axis([5e5,6e10,9.5e-33,4.5e-12])
plt.show()