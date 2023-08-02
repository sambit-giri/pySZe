import numpy as np
import pySZe

def test_hubble():
    par = pySZe.par()
    assert np.abs(100*par.cosmo.h0-pySZe.hubble(1,par))<1e-2

def test_rhoc():
    par = pySZe.par()
    assert np.abs(pySZe.rhoc_of_z(par)-277600000000.0)<1e-2
