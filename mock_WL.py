from __future__ import division, print_function
import sys
import time
import h5py
import numpy as np
from numpy.lib import scimath as sm
from scipy import stats
import importlib
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline

import cosmo, Mconversion_concentration

# Syntax
# python mock_WL.py WLconfig mockconfig catalog.fits

def main():
    WLconfigMod = importlib.import_module(sys.argv[1][:-3])
    mockconfigMod = importlib.import_module(sys.argv[2][:-3])
    np.random.seed(WLconfigMod.random_seed)
    cosmology = mockconfigMod.cosmology
    MCrel = Mconversion_concentration.ConcentrationConversion(mockconfigMod.mcType, cosmology, setup_interp=True)
    mock_WL = MockUpWL(cosmology, MCrel)
    cat = Table.read(sys.argv[3])

    with h5py.File('mock_WL_%s.hdf5'%time.strftime("%y%m%d-%H%M%S"), 'w') as f:
        for i,name in enumerate(cat['SPT_ID']):
            if cat['REDSHIFT'][i]>0 and cat['REDSHIFT'][i]<=WLconfigMod.WL_z_max:

                r_deg, g_2d, g_2d_err, source_dist = mock_WL(cat[i])

                g = f.create_group(name)
                d = g.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                d = g.create_dataset('shear_profile', data=((r_deg, g_2d, g_2d_err)))
                d = g.create_dataset('Nz', data=source_dist)


##### Compute the inverse sec of the complex number z.
# by Joerg Dietrich
def arcsec(z):
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1/z**2)
    val = 1j * np.log(val2 + val1)
    return .5 * np.pi + val

##### Delta Sigma
# by Joerg Dietrich
def get_Delta_Sigma(x, r_s, rho_c_z, delta_c):
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (1 - x**2)
    num = ((3 * x**2) - 2) * arcsec(x)
    div = x**2 * (sm.sqrt(x**2 - 1))**3
    val2 = (num / div).real
    val3 = 2 * np.log(x / 2) / x**2
    return fac * (val1+val2+val3)

##### Sigma_NFW
# by Joerg Dietrich
def get_Sigma(x, r_s, rho_c_z, delta_c):
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (x**2 - 1)
    val2 = (arcsec(x) / (sm.sqrt(x**2 - 1))**3).real
    return fac * (val1-val2)



################################################################################

class MockUpWL:

    def __init__(self, cosmology, MCrel):
        self.cosmology = cosmology
        self.MCrel = MCrel
        self.config_mod = importlib.import_module('WL_input')

        self.rho0 = .63
        self.sigma0 = .07
        self.sigma1 = .25


    def get_gt(self, z, beta_avg, beta2_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        ##### M200 and scale radius, wrt critical density, everything in h units
        c200c = self.MCrel.calC200(self.M200c, z)
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        rs = self.r200c/c200c

        ##### Now let's do WL!
        # dimensionless radial distance
        x = self.config_mod.r_bin_Mpc*self.cosmology['h'] / rs
        r_deg = self.config_mod.r_bin_Mpc*self.cosmology['h'] / self.Dl * 180/np.pi

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/self.Dl/beta_avg
        gamma_t = get_Delta_Sigma(x, rs, self.rho_c_z, delta_c) / Sigma_c
        kappa_t = get_Sigma(x, rs, self.rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_t/(1-kappa_t) * (1 + kappa_t*(beta2_avg/beta_avg**2 - 1))

        return r_deg, g_2d


    def get_Rmis_SPT(self):
        """Draw from SPT positional uncertainty and from double-Rayleigh
        miscentering probability."""
        # SPT positional uncertainty
        sigma_SPT_arcmin = np.sqrt(1.3**2 + self.theta_c**2)/self.xi
        sigma_SPT = sigma_SPT_arcmin/60 * np.pi/180 * self.Dl
        SPT_draw = sigma_SPT*np.random.randn()

        # Double Rayleigh
        temp = np.random.rand()
        if temp<self.rho0:
            R_draw = self.sigma0 * np.random.randn()
        else:
            R_draw = self.sigma1 * np.random.randn()

        theta = np.pi * np.random.rand()
        draw = np.sqrt(SPT_draw**2 + R_draw**2 + 2*np.cos(theta)*SPT_draw*R_draw)

        return draw



    def get_miscentered_gt(self, z, beta_avg, beta2_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        ##### M200 and scale radius, wrt critical density, everything in h units

        c200c = self.MCrel.calC200(self.M200c, z)
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        rs = self.r200c/c200c

        ##### Now let's do WL!
        # dimensionless radial distance
        r_arr = np.logspace(-3, 1, 100)
        x = r_arr / rs

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/self.Dl/beta_avg

        # Miscentered Sigma(r)
        r_mis = self.get_Rmis_SPT()
        Sigma = get_Sigma(x, rs, self.rho_c_z, delta_c) / Sigma_c
        Sigma_mis = self.miscenter_profile(r_arr, Sigma, r_mis)

        integrand = InterpolatedUnivariateSpline(r_arr, r_arr*Sigma_mis)
        Delta_Sigma_mis = np.array([2/r_**2 * integrand.integral(0, r_) for r_ in r_arr])

        g_t = Delta_Sigma_mis/(1-Sigma_mis) * (1 + Sigma_mis*(beta2_avg/beta_avg**2 - 1))

        r_deg = self.config_mod.r_bin_Mpc*self.cosmology['h'] / self.Dl * 180/np.pi
        g_t_interp = InterpolatedUnivariateSpline(r_arr, g_t)
        g_t = g_t_interp(self.config_mod.r_bin_Mpc*self.cosmology['h'])

        return r_deg, g_t


    def miscenter_profile(self, r, profile, r_mis):
        profile_interp = InterpolatedUnivariateSpline(r, profile)
        theta = np.linspace(0, np.pi, 64)
        # [r, theta]
        r_eff = np.sqrt(r[:,None]**2 + r_mis**2 + 2*np.cos(theta[None,:])*r[:,None]*r_mis)
        profile_theta = profile_interp(r_eff)/np.pi

        profile_r_mis = np.trapz(profile_theta, theta, axis=-1)
        return profile_r_mis



    def get_source_gals(self, z):
        """Return stochastic realization of source galaxy redshifts with `z>z_cl
        + z_offset` for each radial bin."""
        r_deg = self.config_mod.r_edge_Mpc*self.cosmology['h'] / self.Dl * 180/np.pi
        area_bin_arcmin = np.pi * 60**2 * (r_deg[1:]**2 - r_deg[:-1]**2)
        N = np.random.poisson(area_bin_arcmin * self.config_mod.source_p_arcmin2)
        z_dist = [np.random.lognormal(np.log(self.config_mod.source_lognorm_dist_mean),
                                     self.config_mod.source_lognorm_dist_sigma,
                                     this_N) for this_N in N]
        for i in range(len(N)):
            idx_behind = np.where(z_dist[i] > z+self.config_mod.z_offset)
            z_dist[i] = z_dist[i][idx_behind]

        z_dist_total = [item for sublist in z_dist for item in sublist]

        return z_dist, z_dist_total


    def get_beta(self, z_cl, z_dist):
        """Return `<beta>` and `<beta**2>` given a redshift distribution."""
        beta = np.array([cosmo.dA_two_z(z_cl, z, self.cosmology)/cosmo.dA(z, self.cosmology) for z in z_dist])
        beta_avg = np.mean(beta)
        beta2_avg = np.mean(beta**2)
        return beta_avg, beta2_avg

    def __call__(self, cat):
        """Wrapper function: Call all workers and return everything."""
        z_cl = cat['REDSHIFT']
        self.xi = cat['XI']
        self.theta_c = cat['THETA_CORE']
        M500c = cat['Mwl_500']

        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z_cl, self.cosmology)**2
        self.Dl = cosmo.dA(z_cl, self.cosmology)

        self.r500c = (3*M500c/4/np.pi/500/self.rho_c_z)**(1/3)
        self.M200c = self.MCrel.lnM_to_lnM200(z_cl, M500c)[0,0]
        self.r200c = (3*self.M200c/4/np.pi/200/self.rho_c_z)**(1/3)


        source_dist_r, source_dist = self.get_source_gals(z_cl)
        beta_avg, beta2_avg = self.get_beta(z_cl, source_dist)
        r_deg, g_2d = self.get_miscentered_gt(z_cl, beta_avg, beta2_avg)
        # r_deg, g_2d = self.get_gt(z_cl, beta_avg, beta2_avg)
        # Error on shear is shape_noise / sqrt(N(r))
        N_r = np.array([len(source_dist_r[i]) for i in range(len(source_dist_r))])

        good_idx = (N_r>4).nonzero()[0]

        N_r = N_r[good_idx]
        g_2d = g_2d[good_idx]

        g_2d_err = self.config_mod.shape_noise / np.sqrt(N_r)
        g_2d+= np.random.normal(0, g_2d_err)

        return r_deg[good_idx], g_2d, g_2d_err, source_dist



if __name__ == '__main__':
    main()
