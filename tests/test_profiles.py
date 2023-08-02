import numpy as np
import pySZe

def test_cvir_fct():
    mvir = 1e12
    par  = pySZe.par()
    cvir = pySZe.cvir_fct(mvir,par)
    assert np.abs(cvir-8.035)<1e-1

def test_r500_fct():
    par  = pySZe.par()
    r200 = 1.0
    cvir = 8.035
    r500 = pySZe.r500_fct(r200,cvir)
    assert np.abs(r500-0.677)<1e-1

def test_M500_fct():
    par  = pySZe.par()
    mvir = 1e12
    cvir = pySZe.cvir_fct(mvir,par)
    M500 = pySZe.M500_fct(mvir,cvir)
    assert np.abs(np.log10(M500)-11.89)<1e-1