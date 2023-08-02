import numpy as np
import matplotlib.pyplot as plt 

import pickle 

from matplotlib import rcParams, ticker, cm
rcParams['font.family'] = 'sans-serif'
#rcParams['font.family'] = 'small-caps'
#rcParams['text.usetex'] = True
rcParams['axes.labelsize']  = 13
rcParams['font.size']       = 13 
rcParams['axes.linewidth']  = 1.6


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



ax = plot_TkSZ_f90()
ax = plot_TkSZ_f150(ax=ax)
ax.legend()
plt.show()


