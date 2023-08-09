import numpy as np 
from scipy.integrate import quad
from scipy.interpolate import splev, splrep
from astropy import units, constants

from .params import par

class StochasticGW:
    def __init__(self, param=None, xe=None, **kwargs):
        if param is None:
            param = par()
        self.param = param
        if xe is None:
            print('Getting ionisation history from camb...')
            import camb
            # Set cosmological parameters
            params = camb.CAMBparams()
            params.set_cosmology(
                                H0=param.cosmo.h0*100, 
                                ombh2=param.cosmo.Ob*param.cosmo.h0**2, 
                                omch2=param.cosmo.Om*param.cosmo.h0**2,
                                )
            params.set_dark_energy(w=-1.0)
            params.InitPower.set_params(As=param.cosmo.As, ns=param.cosmo.ns)
            params.set_matter_power(redshifts=[0.], kmax=2.0)
            # Compute results
            results = camb.get_results(params)
            zs = 10**np.linspace(-3,3,100)
            background_evol = results.get_background_redshift_evolution(zs)
            xs = background_evol['x_e']
            xe = {'z_e': zs, 'x_e': xs}
            print('...done')
        self.set_ne_evolution(xe) 

    def set_ne_evolution(self, xe, **kwargs):
        fit_kwargs = kwargs.get('fit_kwargs')
        self.xe = xe 
        self.I_zini_fit = None
        self.I_zini(0, kind='fit', fit_kwargs=fit_kwargs)
        return None
    
    def Idash_zini(self, z):
        zs, xs = self.xe['z_e'], self.xe['x_e']
        x   = lambda z: 10**splev(np.log10(z), splrep(np.log10(zs),np.log10(xs)))
        itg = lambda z: x(z)**-2*(1+z)**(-3/2)
        return itg(z)

    def I_zini(self, z, kind='fit', fit_kwargs=None):
        if fit_kwargs is None: 
            fit_kwargs={'nkbins':150, 'zmin':1e-3, 'zmax':1111}
        itg = lambda z: self.Idash_zini(z)
        Iz  = np.vectorize(lambda z: quad(itg, 0, z, args=())[0])
        if self.I_zini_fit is None:
            print('Creating a fit for I(z_ini)...')
            zz  = np.logspace(np.log10(fit_kwargs['zmin']),np.log10(fit_kwargs['zmax']),fit_kwargs['nkbins'])
            self.I_zini_fit = lambda z: 10**splev(np.log10(z), splrep(np.log10(zz),np.log10(Iz(zz))))
            print('...done')
        if kind.lower()=='fit': return self.I_zini_fit(z)
        else: return Iz(z)

    def Tcmb_to_T0(self, z):
        Tcmb0 = 2.725*units.K
        Tcmb  = lambda z: Tcmb0*(1+z)
        T0 = 2*np.pi*((constants.k_B/constants.h)*Tcmb(z)).to('GHz')
        return T0.value

    def conversion_probability(self, **kwargs):
        kind = kwargs.get('kind', 'Domcke2021')
        kind_valid = ['domcke2021']
        if kind.lower()==kind_valid[0]:
            # Solving Eq. 7 in https://arxiv.org/pdf/2006.01161.pdf
            try: B0 = kwargs.get('B0').to('nG').value
            except: B0 = kwargs.get('B0')
            try: w0 = kwargs.get('w0').to('GHz').value
            except: w0 = kwargs.get('w0')
            if w0 is None:
                try: f = kwargs.get('f').to('GHz').value
                except: f = kwargs.get('f')
                w0 = f*2*np.pi #GHz
            try: dl0 = kwargs.get('dl0').to('Mpc').value
            except: dl0 = kwargs.get('dl0')
            if dl0 is None:
                lam_EQ = 95.0*2*np.pi #Mpc
                try: lam0_B = kwargs.get('lam0_B').to('Mpc').value
                except: lam0_B = kwargs.get('lam0_B')
                try: dl0 = min(lam_EQ, lam0_B)
                except: dl0 = np.vstack((lam_EQ*np.ones_like(lam0_B), lam0_B)).min(axis=0)
            z_ini = kwargs.get('z_ini')
            T0 = self.Tcmb_to_T0(0) #56.78*2*np.pi #GHz
            P  = 6.3e-19*B0**2*(w0/T0)**2*(1/dl0)*self.I_zini(z_ini)/1e6
        else:
            print('kind={} calculation is not implemented.')
            print('Implemented options are', kind_valid)
            P = None
        return P
    
    def gw_overdensity_to_characteristic_strain(self, Ogw, **kwargs):
        # Solving Eq. 11 in https://arxiv.org/pdf/2006.01161.pdf
        try: f = kwargs.get('f').to('GHz').value
        except: f = kwargs.get('f')
        if f is None:
            try: w0 = kwargs.get('w0').to('GHz').value
            except: w0 = kwargs.get('w0')
            f = w0/(2*np.pi) #GHz
        H0 = self.param.cosmo.h0*100*units.km/units.s/units.Mpc
        H0_GHz = H0.to('GHz').value
        hc = np.sqrt(3*H0_GHz**2/4/np.pi**2*Ogw/f**2)
        return hc 
    
    def characteristic_strain_to_gw_overdensity(self, hc, **kwargs):
        # Solving Eq. 11 in https://arxiv.org/pdf/2006.01161.pdf
        try: f = kwargs.get('f').to('GHz').value
        except: f = kwargs.get('f')
        if f is None:
            try: w0 = kwargs.get('w0').to('GHz').value
            except: w0 = kwargs.get('w0')
            f = w0/(2*np.pi) #GHz
        H0 = self.param.cosmo.h0*100*units.km/units.s/units.Mpc
        H0_GHz = H0.to('GHz').value
        Ogw = hc**2*f**2/(3*H0_GHz**2/4/np.pi**2)
        return Ogw
    
    def gw_overdensity_to_dlnfgamma(self, Ogw, P, **kwargs):
        # Solving Eq. 10 in https://arxiv.org/pdf/2006.01161.pdf
        try: w0 = kwargs.get('w0').to('GHz').value
        except: w0 = kwargs.get('w0')
        if w0 is None:
            try: f = kwargs.get('f').to('GHz').value
            except: f = kwargs.get('f')
            w0 = f*2*np.pi #GHz  
        T0 = kwargs.get('T0', self.Tcmb_to_T0(0)) #56.78*2*np.pi #GHz
        dlnfgamma = np.pi**4/15*(T0/w0)**3*P*Ogw/self.param.cosmo.Og
        return dlnfgamma
    
    def dlnfgamma_to_gw_overdensity(self, dlnfgamma, P, **kwargs):
        # Solving Eq. 10 in https://arxiv.org/pdf/2006.01161.pdf
        try: w0 = kwargs.get('w0').to('GHz').value
        except: w0 = kwargs.get('w0')
        if w0 is None:
            try: f = kwargs.get('f').to('GHz').value
            except: f = kwargs.get('f')
            w0 = f*2*np.pi #GHz  
        T0 = kwargs.get('T0', self.Tcmb_to_T0(0)) #56.78*2*np.pi #GHz
        Ogw = dlnfgamma/(np.pi**4/15*(T0/w0)**3*P)*self.param.cosmo.Og
        return Ogw

    
