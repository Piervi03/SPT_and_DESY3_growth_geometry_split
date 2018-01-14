from __future__ import division
import numpy as np
import scipy.integrate

DIST_H = 2997.92458

#G = 6.67408e-11
#pc_m = 3.0857e16
#Msun = 1.98855e30
#H = 100e3/pc_m/1e6
# 3*H**2/8/np.pi/G/Msun*pc_m**3*1e18
RHOCRIT = 2.77528233987e11

def Ez(z, cosmology):
    """Return the growth factor."""
    return (cosmology['Omega_m']*(1+z)**3 + cosmology['Omega_l']*(1+z)**(3*(1+cosmology['w0'])))**.5

def Omega_m_z(z, cosmology):
    """Return Omega_m(z)."""
    return cosmology['Omega_m'] * (1+z)**3 / Ez(z, cosmology)**2

def dA(z, cosmology):
    """Return angular diameter distance in Mpc/h."""
    integrand = lambda z_int: 1/Ez(z_int, cosmology)
    return scipy.integrate.quad(integrand, 0., z)[0] *DIST_H/(1+z)

def dA_two_z(z1, z2, cosmology):
    """Return angular diameter distance between two redshifts (z1<z2) in Mpc/h."""
    integrand = lambda z_int: 1/Ez(z_int, cosmology)
    return scipy.integrate.quad(integrand, z1, z2)[0] * DIST_H/(1+z2)

def deltaV(z_arr, cosmology):
    """Return solid angle volume as a function of redshift [(Mpc/h)^3]."""
    dA_arr = [dA(z, cosmology) for z in z_arr]
    return DIST_H*((1+z_arr)*dA_arr)**2 / Ez(z_arr, cosmology)
