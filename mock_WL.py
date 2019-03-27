from __future__ import division, print_function
import sys
import h5py
import numpy as np
from numpy.lib import scimath as sm
from scipy import stats
import importlib
from astropy.table import Table

import cosmo, Mconversion_concentration


def main():
    configMod = importlib.import_module(sys.argv[1][:-3])
    np.random.seed(configMod.random_seed)
    cosmology = configMod.cosmology
    MCrel = Mconversion_concentration.ConcentrationConversion(configMod.mcType, cosmology)
    mock_WL = MockUpWL(cosmology, MCrel)
    cat = Table.read(sys.argv[2])

    with h5py.File('mock_WL.hdf5', 'w') as f:
        for i,name in enumerate(cat['SPT_ID']):
            r_deg, g_2d, g_2d_err, source_dist = mock_WL(cat['Mwl_200'][i], cat['REDSHIFT'][i])

            g = f.create_group(name)
            d = g.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
            d = g.create_dataset('shear', data=((r_deg, g_2d, g_2d_err)))
            d = g.create_dataset('N_z', data=source_dist)


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




class MockUpWL:

    def __init__(self, cosmology, MCrel):
        self.cosmology = cosmology
        self.MCrel = MCrel
        self.config_mod = importlib.import_module('WL_input')
        Delta_r = self.config_mod.r_deg[1] - self.config_mod.r_deg[0]
        self.area_bin_arcmin = np.pi * 60**2 * ((self.config_mod.r_deg+Delta_r/2)**2 - (self.config_mod.r_deg-Delta_r/2)**2)


    def get_gt(self, M200c, z, beta_avg, beta2_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        ##### M200 and scale radius, wrt critical density, everything in h units
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z, self.cosmology)**2
        Dl = cosmo.dA(z, self.cosmology)

        r200c = (3*M200c/4/np.pi/200/rho_c_z)**(1/3)
        c200c = self.MCrel.calC200(M200c, z)
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        rs = r200c/c200c

        ##### Now let's do WL!
        # dimensionless radial distance
        x = self.config_mod.r_deg * Dl * np.pi/180 / rs
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/Dl/beta_avg
        gamma_t = get_Delta_Sigma(x, rs, rho_c_z, delta_c) / Sigma_c
        kappa_t = get_Sigma(x, rs, rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_t/(1-kappa_t) * (1 + kappa_t*(beta2_avg/beta_avg**2 - 1))

        return g_2d


    def get_source_gals(self, z):
        """Return stochastic realization of source galaxy redshifts with `z>z_cl
        + z_offset` for each radial bin."""
        N = np.random.poisson(self.area_bin_arcmin * self.config_mod.source_p_arcmin2)
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

    def __call__(self, M200c, z_cl):
        """Wrapper function: Call all workers and return everything."""
        source_dist_r, source_dist = self.get_source_gals(z_cl)
        beta_avg, beta2_avg = self.get_beta(z_cl, source_dist)
        g_2d = self.get_gt(M200c, z_cl, beta_avg, beta2_avg)
        # Error on shear is shape_noise / sqrt(N(r))
        N_r = np.array([len(source_dist_r[i]) for i in range(len(source_dist_r))])
        g_2d_err = self.config_mod.shape_noise / np.sqrt(N_r)
        g_2d+= np.random.normal(0, g_2d_err)

        return self.config_mod.r_deg, g_2d, g_2d_err, source_dist



if __name__ == '__main__':
    main()
