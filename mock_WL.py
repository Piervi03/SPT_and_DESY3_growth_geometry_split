from __future__ import division, print_function
import sys
import time
import fitsio
import h5py
import numpy as np
from numpy.lib import scimath as sm
from scipy import stats
import importlib
from astropy.table import Table
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

import cosmo, Mconversion_concentration, miscentering

# Syntax
# python mock_WL.py WLconfig mockconfig catalog.fits

def main():
    WLconfigMod = importlib.import_module(sys.argv[1][:-3])
    mockconfigMod = importlib.import_module(sys.argv[2][:-3])
    cosmology = mockconfigMod.cosmology
    MCrel = Mconversion_concentration.ConcentrationConversion(mockconfigMod.mcType, cosmology, setup_interp=True)
    mock_WL = MockUpWL(cosmology, MCrel)
    cat = Table.read(sys.argv[3])

    with h5py.File('mock_WL_%s.hdf5'%time.strftime("%y%m%d-%H%M%S"), 'w') as f:
        for i,name in enumerate(cat['SPT_ID']):
            if (cat['REDSHIFT'][i]>0)&(cat['REDSHIFT'][i]<WLconfigMod.WL_z_max)&(cat['FIELD'][i] not in ['ra11hdec-25', 'ra13hdec-25']):
                r_Mpch, r_arcmin, g_2d, g_2d_err, source_dist, g_2d_cen, g_2d_mis, g_2d_noerr, beta = mock_WL(cat[i])

                g = f.create_group(name)

                d = g.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                d = g.create_dataset('r_arcmin', data=r_arcmin)
                d = g.create_dataset('r_Mpch', data=r_Mpch)
                d = g.create_dataset('shear', data=g_2d)
                d = g.create_dataset('shear_err', data=g_2d_err)
                d = g.create_dataset('shear_cen', data=g_2d_cen)
                d = g.create_dataset('shear_mis', data=g_2d_mis)
                d = g.create_dataset('shear_noerr', data=g_2d_noerr)
                d = g.create_dataset('source_Nz', data=source_dist)
                d = g.create_dataset('beta', data=beta)


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
        self.miscenterer = miscentering.MisCentering(self.config_mod.miscenter_opt)
        # DES Y3 source P(z)
        fits = fitsio.FITS(self.config_mod.source_Pz_file)
        self.source_z = {'z': fits['nz_source']['Z_MID'][:]}
        for i in range(2,5):
            self.source_z['BIN%d'%i] = fits['nz_source']['BIN%d'%i][:]
        self.source_z['allbins'] = np.array([self.source_z['BIN%d'%i] for i in range(2,5)])
        # DES Y3 source weights
        self.source_weights = np.loadtxt(self.config_mod.source_weights_file, unpack=True)
        self.source_weights_mean = np.average(self.source_weights[0]*np.ones(self.source_weights[1:].shape), weights=self.source_weights[1:], axis=1)
        self.source_weights_cum = np.cumsum(self.source_weights[1:], axis=1)
        self.source_weights_cum/= self.source_weights_cum[:,-1][:,None]
        # DES Y3 tomo bin weights
        weights = np.load(self.config_mod.tomo_bin_weight_file)
        self.w_interp = interp1d(weights[0], weights[1:])


    def get_miscentered_gt(self, z, beta_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        ##### M200 and scale radius, wrt critical density, everything in h units
        c = self.MCrel.calC200(self.M_Delta, z)
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        rs = self.r_Delta/c

        ##### Now let's do WL!
        x = self.r_arr / rs

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/self.Dl/beta_avg

        # Miscentered Sigma(r)
        R_mis = self.miscenterer.get_mean_Rmis(self.cat, self.cosmology)

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
            Sigma_mis[self.r_arr<R_mis] = Sigma_NFW_at_Rmis

        Sigma_mis_mean = np.empty(Sigma_NFW.shape)
        Sigma_mis_mean[self.r_arr<R_mis] = Sigma_NFW_at_Rmis
        Sigma_mis_mean[self.r_arr>R_mis] = Sigma_NFW_mean[self.r_arr>R_mis] + (R_mis/self.r_arr[self.r_arr>R_mis])**2 * (Sigma_NFW_at_Rmis-Sigma_NFW_mean_at_Rmis)

        # Reduced shear profile [mass][radius]
        g_t_mis = (Sigma_mis-Sigma_mis_mean)/Sigma_c / (1 - Sigma_mis/Sigma_c)
        g_t_cen = Delta_Sigma_NFW/Sigma_c / (1-Sigma_NFW/Sigma_c)

        return g_t_mis, g_t_cen


    def draw_source_weight(self, BIN, N):
        """Return `N` draws from the distribution of weights of tomo bin `BIN`, labeled from 1 to 4."""
        devs = self.rng.random(N)
        w = np.interp(devs, self.source_weights_cum[BIN-2], self.source_weights[0])
        return w


    def get_source_gals(self, z_cl):
        """Return stochastic realization of source galaxy redshifts and weights for each radial bin.
        Assume equal number of sources in all tomographic bins."""
        area_bin_arcmin = np.pi * (self.r_arcmin_edges[1:]**2 - self.r_arcmin_edges[:-1]**2)
        z_dist_r = np.empty((len(area_bin_arcmin), len(self.source_z['z'])))
        w_dist_b = 3*[None]
        N_r = np.zeros(len(area_bin_arcmin))
        for i in range(len(area_bin_arcmin)):
            # Each tomo bin gets N/3 sources with weights w_dist_b
            for b in range(3):
                this_N = self.rng.poisson(area_bin_arcmin[i] * self.config_mod.source_p_arcmin2 /3)
                N_r[i]+= this_N
                w_dist_b[b] = self.draw_source_weight(b+2, this_N)
            sum_w = self.w_interp(z_cl)[1:] * [np.sum(w_dist_b[b]) for b in range(3)]
            z_dist_r[i] = np.sum(self.source_z['allbins']*sum_w[:,None], axis=0)/np.sum(sum_w)
        z_dist = np.average(self.source_z['allbins'], weights=self.w_interp(z_cl)[1:]*self.source_weights_mean, axis=0)
        return z_dist_r, z_dist, N_r 


    def get_beta(self, z_cl, z_dist):
        """Return `<beta>` and `<beta**2>` given a redshift distribution."""
        beta = np.array([cosmo.dA_two_z(z_cl, z, self.cosmology)/cosmo.dA(z, self.cosmology) for z in self.source_z['z']])
        beta[self.source_z['z']<=z_cl] = 0
        beta_2d = beta * np.ones(z_dist.shape)
        beta_avg = np.average(beta_2d, weights=z_dist, axis=1)
        return beta_avg


    def apply_cl_mem_contamination(self, z, g_2d):
        r_s_fcl = (self.cat['richness']/70)**(1/3) / 10**self.config_mod.boost['logc']
        Sigma_fcl = get_Sigma(self.r_arr/r_s_fcl, r_s_fcl, self.rho_c_z, 1)/get_Sigma(1/r_s_fcl, r_s_fcl, self.rho_c_z, 1)
        A_z = np.exp(self.config_mod.boost['A_inf'] + np.sum(self.config_mod.boost['A'] * np.exp(-.5*(z-self.config_mod.boost['z_arr'])**2/self.config_mod.boost['corr_len']**2)))
        A = (self.cat['richness']/70)**self.config_mod.boost['Blambda'] * A_z * Sigma_fcl
        reduced_shear_cont = 1/(1+A) * g_2d

        return reduced_shear_cont


    def __call__(self, cat):
        """Wrapper function: Call all workers and return everything."""
        self.cat = cat
        z_cl = cat['REDSHIFT']
        self.M_Delta = cat['Mwl_200']

        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z_cl, self.cosmology)**2
        self.Dl = cosmo.dA(z_cl, self.cosmology)

        self.r_Delta = (3*self.M_Delta/4/np.pi/self.Delta_crit/self.rho_c_z)**(1/3)

        # Radii
        r_min = .5
        r_max = 3.2 / (1+z_cl)
        all_edges = np.logspace(-1, 1, 21)*self.cosmology['h']
        good_idx = ((r_min<=all_edges)&(all_edges<=r_max)).nonzero()[0]
        these_edges = all_edges[good_idx]
        these_edges = np.append(np.insert(these_edges, 0, r_min), r_max)
        self.r_arr = 2/3 * (these_edges[1:]**3-these_edges[:-1]**3)/(these_edges[1:]**2-these_edges[:-1]**2)
        self.r_arcmin = self.r_arr / self.Dl * 60*180/np.pi
        self.r_arcmin_edges = these_edges / self.Dl * 60*180/np.pi

        source_dist_r, source_dist, N_r = self.get_source_gals(z_cl)
        beta_avg = self.get_beta(z_cl, source_dist_r)
        # beta_avg0 = self.get_beta(z_cl, source_dist[None,:])
        # print(beta_avg, beta_avg0)
        g_2d_mis, g_2d_cen = self.get_miscentered_gt(z_cl, beta_avg)
        g_2d_cont = self.apply_cl_mem_contamination(z_cl, g_2d_mis)

        # Error on shear is shape_noise / sqrt(N(r))
        good_idx = (np.isfinite(g_2d_cont)&(N_r>4)).nonzero()[0]
        g_2d = g_2d_cont[good_idx]
        g_2d_err = self.config_mod.shape_noise / np.sqrt(N_r[good_idx])
        g_2d+= g_2d_err*self.rng.standard_normal(len(g_2d))

        return self.r_arr[good_idx], self.r_arcmin[good_idx], g_2d, g_2d_err, source_dist, g_2d_cen[good_idx], g_2d_mis[good_idx], g_2d_cont[good_idx], beta_avg[good_idx]



if __name__ == '__main__':
    main()
