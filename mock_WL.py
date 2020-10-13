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
    cosmology = mockconfigMod.cosmology
    MCrel = Mconversion_concentration.ConcentrationConversion(mockconfigMod.mcType, cosmology, setup_interp=True)
    mock_WL = MockUpWL(cosmology, MCrel)
    cat = Table.read(sys.argv[3])

    z_bins = np.linspace(.1, 3.1, 32)
    z_cen = .5*(z_bins[1:]+z_bins[:-1])

    with h5py.File('mock_WL_%s.hdf5'%time.strftime("%y%m%d-%H%M%S"), 'w') as f:
        for i,name in enumerate(cat['SPT_ID']):
            if cat['REDSHIFT'][i]>0 and cat['REDSHIFT'][i]<WLconfigMod.WL_z_max:
                r_arcmin, g_2d, g_2d_err, source_dist = mock_WL(cat[i])

                g = f.create_group(name)

                d = g.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                d = g.create_dataset('r_arcmin', data=r_arcmin)
                d = g.create_dataset('shear', data=g_2d)
                d = g.create_dataset('shear_err', data=g_2d_err)
                d = g.create_dataset('source_redshifts', data=z_cen)
                d = g.create_dataset('source_Nz', data=np.histogram(source_dist, bins=z_bins)[0])
                g.create_dataset('r200_fid', data=cat['r200_fid'][i])


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
        self.rng = np.random.default_rng(self.config_mod.random_seed)


    def get_Rmis_opt(self, lam, z, miscenter_opt):
        sigma0 = miscenter_opt['sigma0'] * ((1+lam)/60)**miscenter_opt['sigma0_lam'] * ((1+z)/1.6)**miscenter_opt['sigma0_z']
        sigma1 = miscenter_opt['sigma1'] * ((1+lam)/60)**miscenter_opt['sigma1_lam']
        mis = miscenter_opt['rho']*sigma0 + (1-miscenter_opt['rho'])*sigma1

        return mis


    def get_miscentered_gt(self, z, beta_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        ##### M200 and scale radius, wrt critical density, everything in h units
        c = 3#10**self.rng.random()#self.MCrel.calC200(self.M_Delta, z)
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        rs = self.r_Delta/c

        ##### Now let's do WL!
        x = self.r_arr / rs

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/self.Dl/beta_avg

        # Miscentered Sigma(r)
        R_mis = self.get_Rmis_opt(self.lam, z, self.config_mod.WL_params['miscenter_opt']) * self.Dl * np.pi/(60*180)

        Sigma_NFW = get_Sigma(x, rs, self.rho_c_z, delta_c)
        Delta_Sigma_NFW = get_Delta_Sigma(x, rs, self.rho_c_z, delta_c)
        Sigma_NFW_mean = Sigma_NFW - Delta_Sigma_NFW

        # NFW surface mass densities at Rmis [mass]
        x_Rmis = R_mis/rs
        Sigma_NFW_at_Rmis = get_Sigma(x_Rmis, rs, self.rho_c_z, delta_c)
        Delta_Sigma_NFW_at_Rmis = get_Delta_Sigma(x_Rmis, rs, self.rho_c_z, delta_c)
        Sigma_NFW_mean_at_Rmis = Sigma_NFW_at_Rmis - Delta_Sigma_NFW_at_Rmis

        # Miscentered quantities
        # Sigma = Sigma(R_mis) for r<R_mis
        Sigma_mis = Sigma_NFW.copy()
        if not R_mis<self.r_arr[0]:
            Sigma_mis[:,self.r_arr<R_mis] = Sigma_NFW_at_Rmis

        Sigma_mis_mean = np.empty(Sigma_NFW.shape)
        Sigma_mis_mean[self.r_arr<R_mis] = Sigma_NFW_at_Rmis
        Sigma_mis_mean[self.r_arr>R_mis] = Sigma_NFW_mean[:] - (R_mis/self.r_arr)**2 * (Sigma_NFW_at_Rmis-Sigma_NFW_mean_at_Rmis)

        # Reduced shear profile [mass][radius]
        g_t = (Sigma_mis-Sigma_mis_mean)/Sigma_c / (1 - Sigma_mis/Sigma_c)

        return g_t


    def get_source_gals(self, z):
        """Return stochastic realization of source galaxy redshifts with `z>z_cl
        + z_offset` for each radial bin."""
        area_bin_arcmin = np.pi * (self.r_arcmin[1:]**2 - self.r_arcmin[:-1]**2)
        N = self.rng.poisson(area_bin_arcmin * self.config_mod.source_p_arcmin2)
        z_dist = [self.rng.lognormal(np.log(self.config_mod.source_lognorm_dist_mean),
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
        return beta_avg


    def apply_cl_mem_contamination(self, z, g_2d):
        r_s_fcl = self.r200_fid / self.config_mod.WL_params['c_fcl']
        delta_c_fcl = 200/3 * self.config_mod.WL_params['c_fcl']**3 / (np.log(1+self.config_mod.WL_params['c_fcl']) - self.config_mod.WL_params['c_fcl']/(1+self.config_mod.WL_params['c_fcl']))
        x_fcl = self.r_arr / r_s_fcl
        Sigma_fcl = get_Sigma(x_fcl, r_s_fcl, self.rho_c_z, delta_c_fcl)/get_Sigma(self.config_mod.WL_params['x0_fcl'], r_s_fcl, self.rho_c_z, delta_c_fcl)
        idx = (z>self.config_mod.WL_params['A_fcl_z'])[0]
        f_cl = self.config_mod.WL_params['A_fcl'][idx] * (self.lam/self.config_mod.WL_params['lambda_piv_fcl'])**self.config_mod.WL_params['B_fcl'] * Sigma_fcl
        reduced_shear_cont = (1-f_cl) * g_2d

        return reduced_shear_cont


    def __call__(self, cat):
        """Wrapper function: Call all workers and return everything."""
        z_cl = cat['REDSHIFT']
        self.SPT_ID = cat['SPT_ID']
        self.xi = cat['XI']
        self.theta_c = cat['THETA_CORE']
        self.M_Delta = cat['Mwl_200']
        self.lam = cat['LAMBDA_MCMF_COMB']
        self.r200_fid = cat['r200_fid']

        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z_cl, self.cosmology)**2
        self.Dl = cosmo.dA(z_cl, self.cosmology)

        self.r_Delta = (3*self.M_Delta/4/np.pi/self.Delta_crit/self.rho_c_z)**(1/3)

        # Radii
        r_min = .25
        r_max = 3  * (1+z_cl)**(-1.3)
        N = int(np.log(r_max/r_min)/.3)
        self.r_arr = np.exp(np.linspace(np.log(r_min), np.log(r_max), N))
        self.r_arcmin = self.r_arr / self.Dl * 60*180/np.pi

        source_dist_r, source_dist = self.get_source_gals(z_cl)
        beta_avg = self.get_beta(z_cl, source_dist)
        g_2d_fid = self.get_miscentered_gt(z_cl, beta_avg)
        g_2d_cont = self.apply_cl_mem_contamination(z_cl, g_2d_fid)
        # Error on shear is shape_noise / sqrt(N(r))
        N_r = np.array([len(source_dist_r[i]) for i in range(len(source_dist_r))])

        good_idx = (N_r>4).nonzero()[0]

        N_r = N_r[good_idx]
        g_2d = g_2d_cont[good_idx]

        # Shape and shot noise
        g_2d_err = self.config_mod.shape_noise / np.sqrt(N_r)

        # Apply scatter
        g_2d+= g_2d_err*self.rng.normal(len(g_2d))

        return self.r_arcmin[good_idx], g_2d, g_2d_err, source_dist



if __name__ == '__main__':
    main()
