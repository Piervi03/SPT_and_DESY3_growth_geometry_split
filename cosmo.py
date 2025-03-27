import numpy as np
import scipy.integrate
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

DIST_H = 2997.92458
RHOCRIT = 2.77537e11
c2_4piG = 1.662919e+18


def Ez(z, cosmology):
    """Return the dimensionless Hubble parameter."""
    return np.sqrt(cosmology['Omega_m']*(1+z)**3
                   + cosmology['Omega_l']*(1+z)**(3*(1+cosmology['w0']+cosmology['wa']))*np.exp(-3*cosmology['wa']*z/(1+z)))


def Omega_m_z(z, cosmology):
    """Return Omega_m(z)."""
    return cosmology['Omega_m'] * (1+z)**3 / Ez(z, cosmology)**2


def dA(z, cosmology):
    """Return angular diameter distance to redshift `z`  in Mpc/h."""
    def integrand(z_int):
        return 1/Ez(z_int, cosmology)
    return scipy.integrate.quad(integrand, 0., z)[0] * DIST_H/(1+z)


def dA_two_z(z1, z2, cosmology):
    """Return angular diameter distance between two redshifts (`z1`<`z2`) in Mpc/h."""
    def integrand(z_int):
        return 1/Ez(z_int, cosmology)
    return scipy.integrate.quad(integrand, z1, z2)[0] * DIST_H/(1+z2)


def deltaV(z, cosmology):
    """Return solid angle volume as a function of redshift `z` [(Mpc/h)^3]."""
    dA_ = [dA(z_, cosmology) for z_ in z]
    return DIST_H * ((1+z)*dA_)**2 / Ez(z, cosmology)


def get_dAs(z_cl_min, z_cl_max, z_s_max, cosmology, num_z_DA=32, num_z_Dl=32, num_z_Ds=32):
    """Precompute angular diameter distances for an array of redshifts and
    set up spline interpolation in ln(z). Units [Mpc/h]."""
    # Angular diameter distance to redshift z
    z = np.logspace(np.log10(z_cl_min), np.log10(z_s_max), num_z_DA)
    dA_ = np.array([dA(z_, cosmology) for z_ in z])
    lndA_interp = InterpolatedUnivariateSpline(np.log(z), np.log(dA_))
    # Angular diameter distance from redshift z_cl to z_s
    z_cl = np.logspace(np.log10(z_cl_min), np.log10(z_cl_max), num_z_Dl)
    z_s = np.logspace(np.log10(z_cl_min), np.log10(z_s_max), num_z_Ds)
    tmp = np.array([dA_two_z(z_cl_, z_s_, cosmology) for z_cl_ in z_cl for z_s_ in z_s]).reshape(num_z_Dl, num_z_Ds)
    dA_twoz_interp = RectBivariateSpline(np.log(z_cl), np.log(z_s), tmp)
    return lndA_interp, dA_twoz_interp
