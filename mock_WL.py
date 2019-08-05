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
                r_arcmin_full, g_2d_fid, r_arcmin, g_2d, g_2d_err, source_dist = mock_WL(cat[i])

                g = f.create_group(name)
                d = g.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                d = g.create_dataset('shear_profile', data=((r_arcmin, g_2d, g_2d_err)))
                d = g.create_dataset('shear_profile_fid', data=((r_arcmin_full, g_2d_fid)))
                d = g.create_dataset('Nz', data=source_dist)
                d = g.create_dataset('R_mis_arcmin', data=((WLconfigMod.rho0, WLconfigMod.sigma0, WLconfigMod.sigma1)))


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
        self.Delta_crit = self.config_mod.Delta_crit

        data_ = np.loadtxt(self.config_mod.beta_bias_file, unpack=True)[:3]
        self.betabias_mean = InterpolatedUnivariateSpline(data_[1], data_[0])

        self.rho0 = self.config_mod.rho0
        self.sigma0 = self.config_mod.sigma0
        self.sigma1 = self.config_mod.sigma1


    def get_gt(self, z, beta_avg, beta2_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        ##### M200 and scale radius, wrt critical density, everything in h units
        c = 3#self.MCrel.calC200(self.M_Delta, z)
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        rs = self.r_Delta/c

        ##### Now let's do WL!
        # dimensionless radial distance
        x = self.config_mod.r_bin_Mpc*self.cosmology['h'] / rs
        r_arcmin = self.config_mod.r_bin_Mpc*self.cosmology['h'] / self.Dl * 60*180/np.pi

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/self.Dl/beta_avg
        # Beta bias
        Sigma_c*= self.betabias_mean(z)

        gamma_t = get_Delta_Sigma(x, rs, self.rho_c_z, delta_c) / Sigma_c
        kappa_t = get_Sigma(x, rs, self.rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_t/(1-kappa_t)# * (1 + kappa_t*(beta2_avg/beta_avg**2 - 1))

        return r_arcmin, g_2d


    def get_Rmis_SPT(self):
        """Draw from SPT positional uncertainty and from double-Rayleigh
        miscentering probability."""
        # SPT positional uncertainty and offset
        sigma_SPT_arcmin = np.sqrt(1.3**2 + self.theta_c**2)/self.xi
        sigma_SPT_Mpc = sigma_SPT_arcmin * np.pi/180/60 * self.Dl

        sigma0 = np.sqrt(self.sigma0**2 + (sigma_SPT_Mpc/self.r500c)**2)
        sigma1 = np.sqrt(self.sigma1**2 + (sigma_SPT_Mpc/self.r500c)**2)

        # Double Rayleigh
        temp = np.random.rand()
        if temp<self.rho0:
            R_draw = stats.rayleigh.rvs(scale=sigma0)
        else:
            R_draw = stats.rayleigh.rvs(scale=sigma1)

        return R_draw


    def get_Rmis_r200c(self):
        """Draw from double-Rayleigh miscentering probability."""
        # Double Rayleigh
        temp = np.random.rand()
        if temp<self.rho0:
            x_draw = stats.rayleigh.rvs(scale=self.sigma0)
        else:
            x_draw = stats.rayleigh.rvs(scale=self.sigma1)

        return x_draw * self.r_Delta

    def get_Rmis_arcmin(self):
        """Draw from double-Rayleigh miscentering probability. Return r_mis
        [arcmin]"""
        # Double Rayleigh
        temp = np.random.rand()
        if temp<self.rho0:
            return stats.rayleigh.rvs(scale=self.sigma0)
        else:
            return stats.rayleigh.rvs(scale=self.sigma1)


    def get_miscentered_gt(self, z, beta_avg, beta2_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        ##### M200 and scale radius, wrt critical density, everything in h units
        c = 5#10**np.random.rand()#self.MCrel.calC200(self.M_Delta, z)
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        rs = self.r_Delta/c

        ##### Now let's do WL!
        # dimensionless radial distance
        r_arr = np.logspace(-4, 1, 256)
        x = r_arr / rs

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/self.Dl/beta_avg

        # Beta bias
        Sigma_c*= self.betabias_mean(z)

        # Miscentered Sigma(r)
        r_mis = self.get_Rmis_arcmin() * self.Dl * np.pi/(60*180)
        Sigma = get_Sigma(x, rs, self.rho_c_z, delta_c)

        Sigma_mis = self.miscenter_profile(r_arr, Sigma, r_mis)

        integrand = InterpolatedUnivariateSpline(r_arr, r_arr*Sigma_mis)
        Delta_Sigma_mis = np.array([2/r_**2 * integrand.integral(r_arr[0], r_) for r_ in r_arr]) - Sigma_mis

        g_t = Delta_Sigma_mis/Sigma_c / (1 - Sigma_mis/Sigma_c)# * (1 + Sigma_mis*(beta2_avg/beta_avg**2 - 1))

        r_arcmin = self.config_mod.r_bin_Mpc*self.cosmology['h'] / self.Dl * 60*180/np.pi
        g_t_interp = InterpolatedUnivariateSpline(r_arr, g_t)
        g_t = g_t_interp(self.config_mod.r_bin_Mpc*self.cosmology['h'])

        return r_arcmin, g_t


    def miscenter_profile(self, r, profile, r_mis):
        profile_interp = InterpolatedUnivariateSpline(np.log(r), np.log(profile))
        theta = np.linspace(0, np.pi, 256)
        # [r, theta]
        r_eff = np.sqrt(r[:,None]**2 + r_mis**2 + 2*np.cos(theta[None,:])*r[:,None]*r_mis)
        profile_theta = np.exp(profile_interp(np.log(r_eff)))/np.pi

        profile_r_mis = np.trapz(profile_theta, theta, axis=-1)
        return profile_r_mis



    def get_source_gals(self, z):
        """Return stochastic realization of source galaxy redshifts with `z>z_cl
        + z_offset` for each radial bin."""
        r_arcmin = self.config_mod.r_edge_Mpc*self.cosmology['h'] / self.Dl * 60*180/np.pi
        area_bin_arcmin = np.pi * (r_arcmin[1:]**2 - r_arcmin[:-1]**2)
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
        self.SPT_ID = cat['SPT_ID']
        self.xi = cat['XI']
        self.theta_c = cat['THETA_CORE']
        self.M_Delta = cat['Mwl_500']

        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z_cl, self.cosmology)**2
        self.Dl = cosmo.dA(z_cl, self.cosmology)

        self.r_Delta = (3*self.M_Delta/4/np.pi/self.Delta_crit/self.rho_c_z)**(1/3)

        source_dist_r, source_dist = self.get_source_gals(z_cl)
        beta_avg, beta2_avg = self.get_beta(z_cl, source_dist)
        r_arcmin, g_2d_fid = self.get_miscentered_gt(z_cl, beta_avg, beta2_avg)
        # Error on shear is shape_noise / sqrt(N(r))
        N_r = np.array([len(source_dist_r[i]) for i in range(len(source_dist_r))])

        good_idx = (N_r>4).nonzero()[0]

        N_r = N_r[good_idx]
        g_2d = g_2d_fid[good_idx]

        g_2d_err = self.config_mod.shape_noise / np.sqrt(N_r)
        g_2d+= np.random.normal(0, g_2d_err)

        return r_arcmin, g_2d_fid, r_arcmin[good_idx], g_2d, g_2d_err, source_dist



if __name__ == '__main__':
    main()
