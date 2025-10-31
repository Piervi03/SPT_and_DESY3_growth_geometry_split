import sys
import time
import fitsio
import h5py
import numpy as np
import importlib
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.special import erf

import cosmo
import lensing
import Mconversion_concentration
import miscentering


# Syntax
# python mock_WL.py WLconfig mockconfig catalog.fits

def main():
    datetime = time.strftime("%y%m%d-%H%M%S")
    spec = importlib.util.spec_from_file_location('dummy', sys.argv[1])
    WLconfigMod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(WLconfigMod)
    spec = importlib.util.spec_from_file_location('dummy', sys.argv[2])
    mockconfigMod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mockconfigMod)
    cosmology = mockconfigMod.cosmology
    cat = Table.read(sys.argv[3])

    # DES weak lensing
    mock_WL = MockUpDESWL(cosmology, sys.argv[1])
    with h5py.File('mock_WL_DES_%s.hdf5' % datetime, 'w') as f:
        g = f.create_group('config')
        fits = fitsio.FITS(WLconfigMod.DES['source_Pz_file'])
        d = g.create_dataset('SOM_Z_MID', data=fits['nz_source']['Z_MID'][:])
        d = g.create_dataset('SOM_BINs', data=[fits['nz_source']['BIN%d' % i][:] for i in range(1, 5)])
        d = g.create_dataset('shape_noise', data=WLconfigMod.DES['shape_noise'])
        g = f.create_group('clusters')
        N = 0
        for i, name in enumerate(cat['SPT_ID']):
            if (cat['REDSHIFT'][i] > 0) & (cat['REDSHIFT'][i] < WLconfigMod.DES['WL_z_max']) & (cat['richness'][i] > 0.):
                res_dict = mock_WL(cat[i])

                gg = g.create_group(name)
                d = gg.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                for k in res_dict.keys():
                    d = gg.create_dataset(k, data=res_dict[k])
                N += 1
        print('DES', N, 'halos')

    # Euclid weak lensing
    mock_WL = MockUpEuclidWL(cosmology, sys.argv[1])
    with h5py.File('mock_WL_Euclid_%s.hdf5' % datetime, 'w') as f:
        g = f.create_group('config')
        _ = g.create_dataset('shape_noise', data=WLconfigMod.Euclid['shape_noise'])
        _ = g.create_dataset('z_s', data=mock_WL.z_s)
        _ = g.create_dataset('Pz', data=mock_WL.tomo_dist)
        g = f.create_group('clusters')
        print("Start processing clusters")
        N = 0
        for i, name in enumerate(cat['SPT_ID']):
            if (cat['REDSHIFT'][i] > 0) & (cat['REDSHIFT'][i] < WLconfigMod.Euclid['WL_z_max']):
                res_dict = mock_WL(cat[i])
                if res_dict is None:
                    continue

                gg = g.create_group(name)
                d = gg.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                for k in res_dict.keys():
                    d = gg.create_dataset(k, data=res_dict[k])
                N += 1
        print('Euclid', N, 'halos')

    # HST weak lensing
    mock_WL = MockUpHSTWL(cosmology, sys.argv[1])
    corr = np.ones((2, 11))
    corr[0, :] = np.linspace(0, .25, 11)
    with h5py.File('mock_WL_HST_%s.hdf5' % datetime, 'w') as f:
        for i, name in enumerate(cat['SPT_ID']):
            if cat['Mwl_HST_200'][i] > 0:
                res_dict = mock_WL(cat[i])
                g = f.create_group(name)
                g.attrs['center'] = 'SZ'
                d = g.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                d = g.create_dataset('shear_profile', data=[res_dict['r_deg'], res_dict['shear'], res_dict['shear_err']])
                d = g.create_dataset('shear_noerr', data=res_dict['shear_noerr'])
                d = g.create_dataset('redshifts', data=res_dict['pz'][0])
                d = g.create_dataset('magbinid', data=np.zeros(len(res_dict['r_deg']), dtype=int))
                d = g.create_dataset('beta', data=res_dict['beta'])
                d = g.create_dataset('beta2', data=res_dict['beta2'])
                gg = g.create_group('magbindata')
                ggg = gg.create_group('0')
                ddd = ggg.create_dataset('magnificationcorr', data=corr)
                ddd = ggg.create_dataset('pzs', data=res_dict['pz'][1])


################################################################################

class MockUpDESWL:

    def __init__(self, cosmology, WLconfigname):
        self.cosmology = cosmology
        spec = importlib.util.spec_from_file_location('dummy', WLconfigname)
        self.config_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.config_mod)
        self.Delta_crit = self.config_mod.Delta_crit
        self.MCrel = Mconversion_concentration.ConcentrationConversion(self.config_mod.DES['mcType'], cosmology, setup_interp=True)
        self.rng = np.random.default_rng(self.config_mod.random_seed)
        # Read boost chain
        with open(self.config_mod.DES['DESboostfile'], 'r') as f:
            tmp = f.readline().split()[1:]
        dat = np.mean(np.loadtxt(self.config_mod.DES['DESboostfile']), axis=0)
        self.boost_dict = {'z_arr': np.linspace(.2, 1., 11)}
        for n, name in enumerate(tmp):
            self.boost_dict[name] = dat[n]
        # Initialize miscentering
        with open(self.config_mod.DES['DESmiscenterfile'], 'r') as f:
            tmp = f.readline().split()[1:]
        dat = np.mean(np.loadtxt(self.config_mod.DES['DESmiscenterfile']), axis=0)
        miscenter_dict = {}
        for n, name in enumerate(tmp):
            miscenter_dict[name] = dat[n]
        miscenter_dict['SPT'] = {'kind': self.config_mod.DES['DEScentertype'], 'kappa_SPT': miscenter_dict['kappa_SPT']}
        miscenter_dict['MCMF'] = {'kind': self.config_mod.DES['DEScentertype']}
        for glob, this in zip(['alpha_SZ_0', 'alpha_SZ_z', 'alpha_SZ_lam', 'SZ_comp0_0', 'SZ_comp0_z', 'SZ_comp0_lam', 'SZ_comp1_0', 'SZ_comp1_z', 'SZ_comp1_lam'],
                              ['alpha_0', 'alpha_z', 'alpha_lam', 'comp0_0', 'comp0_z', 'comp0_lam', 'comp1_0', 'comp1_z', 'comp1_lam']):
            miscenter_dict['SPT'][this] = miscenter_dict[glob]
        for glob, this in zip(['alpha_opt_0', 'alpha_opt_z', 'alpha_opt_lam', 'opt_comp0_0', 'opt_comp0_z', 'opt_comp0_lam', 'opt_comp1_0', 'opt_comp1_z', 'opt_comp1_lam'],
                              ['alpha_0', 'alpha_z', 'alpha_lam', 'comp0_0', 'comp0_z', 'comp0_lam', 'comp1_0', 'comp1_z', 'comp1_lam']):
            miscenter_dict['MCMF'][this] = miscenter_dict[glob]
        self.miscenterer = miscentering.MisCentering(miscenter_dict[self.config_mod.DES['DEScentertype']])
        # DES Y3 source P(z)
        fits = fitsio.FITS(self.config_mod.DES['source_Pz_file'])
        self.source_z = {'z': fits['nz_source']['Z_MID'][:]}
        for i in range(2, 5):
            self.source_z['BIN%d' % i] = fits['nz_source']['BIN%d' % i][:]
        self.source_z['allbins'] = np.array([self.source_z['BIN%d' % i] for i in range(2, 5)])
        # DES Y3 source weights
        self.source_weights = np.loadtxt(self.config_mod.DES['source_weights_file'], unpack=True)
        self.source_weights_mean = np.average(self.source_weights[0]*np.ones(self.source_weights[1:].shape), weights=self.source_weights[1:], axis=1)
        self.source_weights_cum = np.cumsum(self.source_weights[1:], axis=1)
        self.source_weights_cum /= self.source_weights_cum[:, -1][:, None]
        # DES Y3 tomo bin Sigma_crit
        tmp = np.load(self.config_mod.DES['Sigmacrit_file'])
        self.invSigmacrit_interp = interp1d(tmp[0], tmp[1:])

    def get_gt(self, z, beta_bin, w_r_bin):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        # M200 and scale radius, wrt critical density, everything in h units
        c = self.MCrel.calC200(self.M_Delta, z)
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        rs = self.r_Delta/c
        x = self.r_arr / rs
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        invSigma_c = self.Dl*beta_bin/cosmo.c2_4piG
        # Centered shear profiles for reference
        Sigma_NFW = lensing.get_Sigma(x, rs, self.rho_c_z, delta_c)
        DeltaSigma_NFW = lensing.get_DeltaSigma(x, rs, self.rho_c_z, delta_c)
        g_t_cen = DeltaSigma_NFW[:, None]*invSigma_c[None, :] / (1-Sigma_NFW[:, None]*invSigma_c[None, :])
        g_t_cen = np.sum(g_t_cen*w_r_bin[:, 1:]*self.tomo_rescale[None, 1:], axis=1)/np.sum(w_r_bin[:, 1:], axis=1)
        # Miscentered profiles
        R_mis = self.miscenterer.get_mean_Rmis(self.cat, self.cosmology)
        Sigma_mis = lensing.get_Sigma_mis(self.r_arr, rs, self.rho_c_z, delta_c, R_mis)
        DeltaSigma_mis = lensing.get_DeltaSigma_mis(self.r_arr, rs, self.rho_c_z, delta_c, R_mis)
        g_t_mis = DeltaSigma_mis[:, None]*invSigma_c[None, :] / (1-Sigma_mis[:, None]*invSigma_c[None, :])
        g_t_mis = np.sum(g_t_mis*w_r_bin[:, 1:]*self.tomo_rescale[None, 1:], axis=1)/np.sum(w_r_bin[:, 1:], axis=1)

        return g_t_mis, g_t_cen, R_mis

    def draw_source_weight(self, BIN, N):
        """Return `N` draws from the distribution of weights of tomo bin `BIN`, labeled from 1 to 4."""
        devs = self.rng.random(N)
        w = np.interp(devs, self.source_weights_cum[BIN-2], self.source_weights[0])
        return w

    def get_source_gals(self, z_cl):
        """Return stochastic realization of source galaxy redshifts and weights for each radial bin.
        Assume equal number of sources in all tomographic bins."""
        area_bin_arcmin = np.pi * (self.r_arcmin_edges[1:]**2 - self.r_arcmin_edges[:-1]**2)
        w_dist_b = 3*[None]
        N_r = np.zeros(len(area_bin_arcmin))
        sum_w = np.zeros((len(area_bin_arcmin), 4))
        for i in range(len(area_bin_arcmin)):
            # Each tomo bin gets N/3 sources with weights w_dist_b
            for b in range(3):
                this_N = self.rng.poisson(area_bin_arcmin[i] * self.config_mod.DES['source_p_arcmin2'] / 3)
                N_r[i] += this_N
                w_dist_b[b] = self.draw_source_weight(b+2, this_N)
            sum_w[i, 1:] = [np.sum(w_dist_b[b]) for b in range(3)]
        return N_r, sum_w

    def get_beta(self, z_cl):
        """Return `<beta>` and `<beta**2>` for the Y3 redshift distributions."""
        beta = np.array([cosmo.dA_two_z(z_cl, z, self.cosmology)/cosmo.dA(z, self.cosmology) for z in self.source_z['z']])
        beta[self.source_z['z'] <= z_cl] = 0
        beta_bin = np.sum(self.source_z['allbins']*beta[None, :], axis=1)/np.sum(self.source_z['allbins'], axis=1)
        return beta_bin

    def apply_cl_mem_contamination(self, z, Rmis, g_t):
        A = lensing.boost_get_A('Gausssmooth', z, self.cat['richness'], self.r_arr, Rmis, **self.boost_dict)
        reduced_shear_cont = 1/(1+A) * g_t
        return reduced_shear_cont

    def __call__(self, cat):
        """Wrapper function: Call all workers and return everything."""
        self.cat = cat
        z_cl = cat['REDSHIFT']
        self.M_Delta = cat['Mwl_DES_200']
        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z_cl, self.cosmology)**2
        self.Dl = cosmo.dA(z_cl, self.cosmology)
        self.r_Delta = (3*self.M_Delta/4/np.pi/self.Delta_crit/self.rho_c_z)**(1/3)
        # Source bin scaling and weighting
        invSigmacrit = self.invSigmacrit_interp(z_cl)
        with np.errstate(all='ignore'):
            self.tomo_rescale = invSigmacrit[3]/invSigmacrit[:]
        self.tomo_weights = 1/self.tomo_rescale**2
        self.tomo_rescale[np.isinf(self.tomo_rescale)] = 0.
        # Radii
        r_min = .5
        r_max = 3.2 / (1+z_cl)
        all_edges = np.linspace(.5, 2.6, 8)
        good_idx = (r_min <= all_edges) & (all_edges <= r_max)
        these_edges = all_edges[good_idx]
        self.r_arr = 2/3 * (these_edges[1:]**3-these_edges[:-1]**3)/(these_edges[1:]**2-these_edges[:-1]**2)
        self.r_arcmin = self.r_arr / self.Dl * 60*180/np.pi
        self.r_arcmin_edges = these_edges / self.Dl * 60*180/np.pi
        # Source redshift distributions and shear profiles
        N_r, w_r_bin = self.get_source_gals(z_cl)
        beta_bin = self.get_beta(z_cl)
        w_r_bin *= self.tomo_weights
        g_t_mis, g_t_cen, R_mis = self.get_gt(z_cl, beta_bin, w_r_bin)
        g_t_cont = self.apply_cl_mem_contamination(z_cl, R_mis, g_t_mis)
        # Error on shear is shape_noise / sqrt(N(r))
        good_idx = np.isfinite(g_t_cont) & (N_r > 4)
        g_t = g_t_cont[good_idx]
        g_t_err = self.config_mod.DES['shape_noise'] / np.sqrt(N_r[good_idx])
        g_t += g_t_err*self.rng.standard_normal(len(g_t))
        # Return dictionary of outputs
        res_dict = {'r_Mpch': self.r_arr[good_idx],
                    'r_arcmin': self.r_arcmin[good_idx],
                    'N_source': N_r[good_idx],
                    'shear_cen': g_t_cen[good_idx],
                    'shear_mis': g_t_mis[good_idx],
                    'shear_noerr': g_t_cont[good_idx],
                    'shear': g_t,
                    'shear_err': g_t_err,
                    'beta': beta_bin,
                    'tomo_weights_R': w_r_bin[good_idx],
                    'tomo_rescale': self.tomo_rescale,
                    }
        return res_dict


################################################################################

class MockUpEuclidWL:

    def __init__(self, cosmology, WLconfigname):
        self.cosmology = cosmology
        spec = importlib.util.spec_from_file_location('dummy', WLconfigname)
        self.config_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.config_mod)
        self.Delta_crit = self.config_mod.Delta_crit
        self.MCrel = Mconversion_concentration.ConcentrationConversion(self.config_mod.Euclid['mcType'], cosmology, setup_interp=True)
        self.rng = np.random.default_rng(self.config_mod.random_seed)
        # Read boost chain
        with open(self.config_mod.Euclid['DESboostfile'], 'r') as f:
            tmp = f.readline().split()[1:]
        dat = np.mean(np.loadtxt(self.config_mod.Euclid['DESboostfile']), axis=0)
        self.boost_dict = {'z_arr': np.linspace(.01, 1., 11)}
        for n, name in enumerate(tmp):
            self.boost_dict[name] = dat[n]
        # Initialize miscentering
        with open(self.config_mod.Euclid['DESmiscenterfile'], 'r') as f:
            tmp = f.readline().split()[1:]
        dat = np.mean(np.loadtxt(self.config_mod.Euclid['DESmiscenterfile']), axis=0)
        miscenter_dict = {}
        for n, name in enumerate(tmp):
            miscenter_dict[name] = dat[n]
        miscenter_dict['SPT'] = {'kind': self.config_mod.Euclid['DEScentertype'], 'kappa_SPT': miscenter_dict['kappa_SPT']}
        miscenter_dict['MCMF'] = {'kind': self.config_mod.Euclid['DEScentertype']}
        for glob, this in zip(['alpha_SZ_0', 'alpha_SZ_z', 'alpha_SZ_lam', 'SZ_comp0_0', 'SZ_comp0_z', 'SZ_comp0_lam', 'SZ_comp1_0', 'SZ_comp1_z', 'SZ_comp1_lam'],
                              ['alpha_0', 'alpha_z', 'alpha_lam', 'comp0_0', 'comp0_z', 'comp0_lam', 'comp1_0', 'comp1_z', 'comp1_lam']):
            miscenter_dict['SPT'][this] = miscenter_dict[glob]
        for glob, this in zip(['alpha_opt_0', 'alpha_opt_z', 'alpha_opt_lam', 'opt_comp0_0', 'opt_comp0_z', 'opt_comp0_lam', 'opt_comp1_0', 'opt_comp1_z', 'opt_comp1_lam'],
                              ['alpha_0', 'alpha_z', 'alpha_lam', 'comp0_0', 'comp0_z', 'comp0_lam', 'comp1_0', 'comp1_z', 'comp1_lam']):
            miscenter_dict['MCMF'][this] = miscenter_dict[glob]
        self.miscenterer = miscentering.MisCentering(miscenter_dict[self.config_mod.Euclid['DEScentertype']])
        # Source redshift distribution
        self.z_s = np.linspace(.001, 2.6, 2600)
        z_m = .9
        z_0 = z_m/np.sqrt(2)
        p = (self.z_s/z_0) * np.exp(-(self.z_s/z_0)**1.5)
        p /= np.trapezoid(p, self.z_s)
        self.tomo_bin_edges = np.append(np.arange(0, 2.2, .2), 2.6)
        self.tomo_dist = p * np.ones((len(self.tomo_bin_edges)-1, len(self.z_s)))
        tomo_idx = np.arange(0, len(self.tomo_dist), 1)
        for i in tomo_idx:
            if i > 0:
                self.tomo_dist[i] *= (1 + erf((self.z_s-self.tomo_bin_edges[i])/.06/np.sqrt(2))) / 2
            if i < len(self.tomo_dist)-1:
                self.tomo_dist[i] *= (1 + erf(-(self.z_s-self.tomo_bin_edges[i+1])/.06/np.sqrt(2))) / 2
        int_tomo_dist = np.trapezoid(self.tomo_dist, self.z_s, axis=1)
        int_tomo_dist /= np.sum(int_tomo_dist)
        self.tomo_dist_cum = np.insert(np.cumsum(int_tomo_dist), 0, 0.)
        self.get_invSigmac()

    def get_invSigmac(self):
        """Return array of `<invSigmac>` for all tomo bins."""
        z_cluster = np.arange(.01, self.config_mod.Euclid['WL_z_max']+.01, .01)
        # Distances
        dAs = np.array([cosmo.dA(z, self.cosmology) for z in self.z_s])
        dAl = np.array([cosmo.dA(z, self.cosmology) for z in z_cluster])
        dA_two_zs = np.array([cosmo.dA_two_z(z_l, z_s, self.cosmology)
                              for z_l in z_cluster
                              for z_s in self.z_s]).reshape((len(z_cluster), -1))
        dA_two_zs[dA_two_zs < 0] = 0
        # Lensing efficiency [z_cl, z_s]
        invSigmac = dA_two_zs/dAs*dAl[:, None]/cosmo.c2_4piG
        # Weight with N(z) distributions [bin, z_cl, z_s]
        invSigmac_bin = np.sum(invSigmac[None, :, :]*self.tomo_dist[:, None, :], axis=-1)/np.sum(self.tomo_dist, axis=-1)[:, None]
        self.invSigmac_bin_interp = interp1d(z_cluster, invSigmac_bin)
        return 0

    def get_gt(self, z, tomo_rescale, invSigma_c, w_r_bin):
        """Return the predicted radial shear profile for a given mass, redshift,
        and invSigmac."""
        # M200 and scale radius, wrt critical density, everything in h units
        c = self.MCrel.calC200(self.M_Delta, z)
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        rs = self.r_Delta/c
        x = self.r_arr / rs
        # Centered shear profiles for reference
        Sigma_NFW = lensing.get_Sigma(x, rs, self.rho_c_z, delta_c)
        DeltaSigma_NFW = lensing.get_DeltaSigma(x, rs, self.rho_c_z, delta_c)
        g_t_cen = DeltaSigma_NFW[:, None]*invSigma_c[None, :] / (1-Sigma_NFW[:, None]*invSigma_c[None, :])
        g_t_cen = np.sum(g_t_cen*w_r_bin*tomo_rescale[None, :], axis=1)/np.sum(w_r_bin, axis=1)
        # Miscentered profiles
        R_mis = self.miscenterer.get_mean_Rmis(self.cat, self.cosmology)
        Sigma_mis = lensing.get_Sigma_mis(self.r_arr, rs, self.rho_c_z, delta_c, R_mis)
        DeltaSigma_mis = lensing.get_DeltaSigma_mis(self.r_arr, rs, self.rho_c_z, delta_c, R_mis)
        g_t_mis = DeltaSigma_mis[:, None]*invSigma_c[None, :] / (1-Sigma_mis[:, None]*invSigma_c[None, :])
        g_t_mis = np.sum(g_t_mis*w_r_bin*tomo_rescale[None, :], axis=1)/np.sum(w_r_bin, axis=1)
        return g_t_mis, g_t_cen, R_mis

    def get_source_gals(self, z_cl):
        """Return stochastic realization of source galaxy redshifts for each radial bin."""
        # Total sources per radial bin
        area_bin_arcmin = np.pi * (self.r_arcmin_edges[1:]**2 - self.r_arcmin_edges[:-1]**2)
        N_r = self.rng.poisson(area_bin_arcmin * self.config_mod.Euclid['source_p_arcmin2'])
        # Indices of tomo bin to discard
        nokeep = (self.tomo_bin_edges[:-1] < z_cl+self.config_mod.Euclid['z_cl_offset']).nonzero()[0]
        # Draw sources and assign tomo bin
        w_tomo_bin = np.zeros((len(N_r), len(self.tomo_bin_edges)-1))
        for i in range(len(N_r)):
            tomo_bin = np.digitize(self.rng.random(N_r[i]), self.tomo_dist_cum) - 1
            w_tomo_bin[i] = np.histogram(tomo_bin, bins=np.arange(-.5, len(self.tomo_bin_edges)-.5, 1))[0]
            w_tomo_bin[i][nokeep] = 0
        # Sources after tomo bin cut
        N_r_cut = np.sum(w_tomo_bin, axis=1)
        return N_r_cut, w_tomo_bin

    def apply_cl_mem_contamination(self, z, Rmis, g_t):
        A = lensing.boost_get_A('Gausssmooth', z, self.cat['richness'], self.r_arr, Rmis, **self.boost_dict)
        reduced_shear_cont = 1/(1+A) * g_t
        return reduced_shear_cont

    def __call__(self, cat):
        """Wrapper function: Call all workers and return everything."""
        self.cat = cat
        z_cl = cat['REDSHIFT']
        self.M_Delta = cat['Mwl_Euclid_200']
        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z_cl, self.cosmology)**2
        Dl = cosmo.dA(z_cl, self.cosmology)
        self.r_Delta = (3*self.M_Delta/4/np.pi/self.Delta_crit/self.rho_c_z)**(1/3)
        # Source bin scaling and weighting
        invSigmacrit = self.invSigmac_bin_interp(z_cl)
        with np.errstate(all='ignore'):
            tomo_rescale = invSigmacrit[-1]/invSigmacrit[:]
        self.tomo_weights = 1/tomo_rescale**2
        tomo_rescale[np.isinf(tomo_rescale)] = 0.
        # Radii
        r_min = .5
        r_max = 3.2 / (1+z_cl)
        all_edges = np.linspace(.5, 2.6, 8)
        good_idx = ((r_min <= all_edges) & (all_edges <= r_max)).nonzero()[0]
        these_edges = all_edges[good_idx]
        self.r_arr = 2/3 * (these_edges[1:]**3-these_edges[:-1]**3)/(these_edges[1:]**2-these_edges[:-1]**2)
        self.r_arcmin = self.r_arr / Dl * 60*180/np.pi
        self.r_arcmin_edges = these_edges / Dl * 60*180/np.pi
        # Source redshift distributions and shear profiles
        N_r, w_r_bin = self.get_source_gals(z_cl)
        w_r_bin *= self.tomo_weights
        g_t_mis, g_t_cen, R_mis = self.get_gt(z_cl, tomo_rescale, invSigmacrit, w_r_bin)
        g_t_cont = self.apply_cl_mem_contamination(z_cl, R_mis, g_t_mis)
        # Error on shear is shape_noise / sqrt(N(r))
        good_idx = (np.isfinite(g_t_cont) & (N_r > 4)).nonzero()[0]
        g_t = g_t_cont[good_idx]
        g_t_err = self.config_mod.Euclid['shape_noise'] / np.sqrt(N_r[good_idx])
        g_t += g_t_err*self.rng.standard_normal(len(g_t))
        # Return dictionary of outputs
        res_dict = {'r_Mpch': self.r_arr[good_idx],
                    'r_arcmin': self.r_arcmin[good_idx],
                    'N_source': N_r[good_idx],
                    'shear_cen': g_t_cen[good_idx],
                    'shear_mis': g_t_mis[good_idx],
                    'shear_noerr': g_t_cont[good_idx],
                    'shear': g_t,
                    'shear_err': g_t_err,
                    'invSigmac': invSigmacrit,
                    'tomo_weights_R': w_r_bin[good_idx],
                    'tomo_rescale': tomo_rescale,
                    }
        if len(good_idx) == 0:
            return None
        else:
            return res_dict


################################################################################

class MockUpHSTWL:

    def __init__(self, cosmology, WLconfigname):
        self.cosmology = cosmology
        spec = importlib.util.spec_from_file_location('dummy', WLconfigname)
        self.config_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.config_mod)
        self.Delta_crit = self.config_mod.Delta_crit
        self.MCrel = Mconversion_concentration.ConcentrationConversion(self.config_mod.HST['mcType'], cosmology, setup_interp=True)
        self.rng = np.random.default_rng(self.config_mod.random_seed)
        self.pz = np.loadtxt(self.config_mod.HST['source_Pz_file'], unpack=True)

    def get_N_source_gals(self):
        """Return stochastic realization of source galaxy redshifts."""
        area_bin_arcmin = np.pi * (self.r_arcmin_edges[1:]**2 - self.r_arcmin_edges[:-1]**2)
        N = self.rng.poisson(area_bin_arcmin * self.config_mod.HST['source_p_arcmin2'])
        return N

    def get_beta(self, z_cl):
        """Return `<beta>` and `<beta**2>` given a redshift distribution."""
        beta = np.array([cosmo.dA_two_z(z_cl, z, self.cosmology)/cosmo.dA(z, self.cosmology) for z in self.pz[0]])
        beta[self.pz[0] <= z_cl] = 0
        beta2 = beta**2
        beta_avg = np.average(beta, weights=self.pz[1])
        beta2_avg = np.average(beta2, weights=self.pz[1])
        return beta_avg, beta2_avg

    def get_gt(self, z, beta_avg, beta2_avg):
        """Return the predicted radial shear profile for a given mass, redshift,
        and betas."""
        # M200 and scale radius, wrt critical density, everything in h units
        c = self.MCrel.calC200(self.M_Delta, z)
        delta_c = self.Delta_crit/3 * c**3 / (np.log(1+c) - c/(1+c))
        rs = self.r_Delta/c
        # Now let's do WL!
        x = self.r_arr / rs
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/self.Dl/beta_avg
        # NFW halo [mass][radius]
        Sigma_NFW = lensing.get_Sigma(x, rs, self.rho_c_z, delta_c)
        DeltaSigma_NFW = lensing.get_DeltaSigma(x, rs, self.rho_c_z, delta_c)
        # Beta correction [Radius][Mass]
        betaratio = beta2_avg/beta_avg**2
        betaCorr = 1 + Sigma_NFW/Sigma_c*(betaratio-1)
        g_t = betaCorr * DeltaSigma_NFW/Sigma_c / (1-Sigma_NFW/Sigma_c)
        return g_t

    def __call__(self, cat):
        """Wrapper function: Call all workers and return everything."""
        self.cat = cat
        z_cl = cat['REDSHIFT']
        self.M_Delta = cat['Mwl_HST_200']
        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z_cl, self.cosmology)**2
        self.Dl = cosmo.dA(z_cl, self.cosmology)
        self.r_Delta = (3*self.M_Delta/4/np.pi/self.Delta_crit/self.rho_c_z)**(1/3)
        # Radii
        r_min = .5
        r_max = 1.1
        all_edges = np.logspace(-1, 1, 21)*self.cosmology['h']
        good_idx = (r_min <= all_edges) & (all_edges <= r_max)
        these_edges = all_edges[good_idx]
        these_edges = np.append(np.insert(these_edges, 0, r_min), r_max)
        self.r_arr = 2/3 * (these_edges[1:]**3-these_edges[:-1]**3)/(these_edges[1:]**2-these_edges[:-1]**2)
        self.r_arcmin = self.r_arr / self.Dl * 60*180/np.pi
        self.r_deg = self.r_arcmin / 60
        self.r_arcmin_edges = these_edges / self.Dl * 60*180/np.pi
        # Create shear profile
        N_r = self.get_N_source_gals()
        beta_avg, beta2_avg = self.get_beta(z_cl)
        g_t_fid = self.get_gt(z_cl, beta_avg, beta2_avg)
        # Error on shear is shape_noise / sqrt(N(r))
        good_idx = N_r > 4
        g_t = g_t_fid[good_idx]
        g_t_err = self.config_mod.HST['shape_noise'] / np.sqrt(N_r[good_idx])
        g_t += g_t_err*self.rng.standard_normal(len(g_t))
        # Return dict
        res_dict = {'r_Mpch': self.r_arr[good_idx],
                    'r_deg': self.r_deg[good_idx],
                    'shear_noerr': g_t_fid[good_idx],
                    'shear': g_t,
                    'shear_err': g_t_err,
                    'pz': self.pz,
                    'beta': beta_avg,
                    'beta2': beta2_avg,
                    }
        return res_dict


if __name__ == '__main__':
    main()
