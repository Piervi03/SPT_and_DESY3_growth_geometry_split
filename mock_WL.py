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

import cosmo, lensing, Mconversion_concentration, miscentering

# Syntax
# python mock_WL.py WLconfig mockconfig catalog.fits

def main():
    datetime = time.strftime("%y%m%d-%H%M%S")
    WLconfigMod = importlib.import_module(sys.argv[1][:-3])
    mockconfigMod = importlib.import_module(sys.argv[2][:-3])
    cosmology = mockconfigMod.cosmology
    cat = Table.read(sys.argv[3])

    # DES weak lensing
    mock_WL = MockUpDESWL(cosmology, sys.argv[1][:-3])
    with h5py.File('mock_WL_DES_%s.hdf5'%datetime, 'w') as f:
        g = f.create_group('config')
        fits = fitsio.FITS(WLconfigMod.DES['source_Pz_file'])
        d = g.create_dataset('SOM_Z_MID', data=fits['nz_source']['Z_MID'][:])
        d = g.create_dataset('SOM_BINs', data=[fits['nz_source']['BIN%d'%i][:] for i in range(1,5)])
        g = f.create_group('clusters')
        for i,name in enumerate(cat['SPT_ID']):
            if (cat['REDSHIFT'][i]>0)&(cat['REDSHIFT'][i]<WLconfigMod.DES['WL_z_max'])&(cat['FIELD'][i] not in ['ra11hdec-25', 'ra13hdec-25', 'ra23hdec-25', 'ra23hdec-35']):
                res_dict = mock_WL(cat[i])

                gg = g.create_group(name)
                d = gg.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                for k in res_dict.keys():
                    d = gg.create_dataset(k, data=res_dict[k])

    # HST weak lensing
    mock_WL = MockUpHSTWL(cosmology, sys.argv[1][:-3])
    corr = np.ones((2,11))
    corr[0,:] = np.linspace(0, .25, 11)
    with h5py.File('mock_WL_HST_%s.hdf5'%datetime, 'w') as f:
        for i,name in enumerate(cat['SPT_ID']):
            if cat['Mwl_HST_200'][i]>0:
                r_Mpch, r_deg, g_t, g_t_err, source_dist = mock_WL(cat[i])
                g = f.create_group(name)
                g.attrs['center'] = 'SZ'
                d = g.create_dataset('z_cluster', data=cat['REDSHIFT'][i])
                d = g.create_dataset('shear_profile', data=[r_deg, g_t, g_t_err])
                d = g.create_dataset('redshifts', data=source_dist[0])
                d = g.create_dataset('magbinid', data=np.zeros(len(r_deg), dtype=int))
                gg = g.create_group('magbindata')
                ggg = gg.create_group('0')
                ddd = ggg.create_dataset('magnificationcorr', data=corr)
                ddd = ggg.create_dataset('pzs', data=source_dist[1])

    # Create setup file
    with open ('WLsimcalib_%s.py'%datetime, 'w') as f:
        f.write("import numpy as np\n\n")
        #f.write("WLcalibration = {\n    # HST\n    'HSTsim': {\n")
        #for i,name in enumerate(cat['SPT_ID']):
        #    if cat['Mwl_HST_200'][i]>0:
        #        f.write("        '%s': {'z': %.3f, 'bias': [%.3f, %.3f, %.3f, %.3f], 'obs_scatter': %.3e, 'center_err': %.3f},\n"%(
        #        name, cat['REDSHIFT'][i], 1, .02, mockconfigMod.scaling['DWL_HST'], .05, 6e13, .075))
        #f.write("    },\n")
        #f.write("    'HSTmcErr': .04,\n")
        #f.write("    'HSTshearErr': .023,\n")
        #f.write("    'HSTzDistErr': .047,\n")
        #f.write("\n    # DES\n")
        #f.write("    'miscenter_opt': {\n")
        #for k in WLconfigMod.DES['miscenter_opt'].keys():
        #    f.write("        '%s': %s,\n"%(k, str(WLconfigMod.DES['miscenter_opt'][k])))
        #f.write("    },\n")
        #f.write("    'boost': {\n")
        #for k in WLconfigMod.DES['boost'].keys():
        #    f.write("        '%s': %s,\n"%(k, str(WLconfigMod.DES['boost'][k])))
        #f.write("    },\n")
        f.write("    # Megacam\n    'MegacamSim': (.938, .028, .214, .04),\n    'MegacamMcErr': .015,\n    'MegacamCenterErr': .03,\n    'MegacamShearErr': .032,\n    'MegacamzDistErr': .012,\n    'MegacamContamCorr': .009,\n    'Megacam_LSS': (6.3e13, 7e12),\n")
        f.write("}\n")



################################################################################

class MockUpDESWL:

    def __init__(self, cosmology, WLconfigname):
        self.cosmology = cosmology
        self.config_mod = importlib.import_module(WLconfigname)
        self.Delta_crit = self.config_mod.Delta_crit
        self.MCrel = Mconversion_concentration.ConcentrationConversion(self.config_mod.DES['mcType'], cosmology, setup_interp=True)
        self.rng = np.random.default_rng(self.config_mod.random_seed)
        # Read boost chain
        with open(self.config_mod.DES['DESboostfile'], 'r') as f:
            tmp = f.readline().split()[1:]
        dat = np.median(np.loadtxt(self.config_mod.DES['DESboostfile']), axis=0)
        self.boost_dict = {'z_arr': np.linspace(.2, .9, 10)}
        for n,name in enumerate(tmp):
            self.boost_dict[name] = dat[n]
        # Initialize miscentering
        with open(self.config_mod.DES['DESmiscenterfile'], 'r') as f:
            tmp = f.readline().split()[1:]
        dat = np.median(np.loadtxt(self.config_mod.DES['DESmiscenterfile']), axis=0)
        miscenter_dict = {}
        for n,name in enumerate(tmp):
            miscenter_dict[name] = dat[n]
        miscenter_dict['SPT'] = {'kind': self.config_mod.DES['DEScentertype'], 'kappa_SPT': miscenter_dict['kappa_SPT']}
        miscenter_dict['MCMF'] = {'kind': self.config_mod.DES['DEScentertype']}
        for glob,this in zip(['alpha_SZ_0', 'alpha_SZ_z', 'alpha_SZ_lam', 'SZ_comp0_0', 'SZ_comp0_z', 'SZ_comp0_lam', 'SZ_comp1_0', 'SZ_comp1_z', 'SZ_comp1_lam'],
                             ['alpha_0', 'alpha_z', 'alpha_lam', 'comp0_0', 'comp0_z', 'comp0_lam', 'comp1_0', 'comp1_z', 'comp1_lam']):
            miscenter_dict['SPT'][this] = miscenter_dict[glob]
        for glob,this in zip(['alpha_opt_0', 'alpha_opt_z', 'alpha_opt_lam', 'opt_comp0_0', 'opt_comp0_z', 'opt_comp0_lam', 'opt_comp1_0', 'opt_comp1_z', 'opt_comp1_lam'],
                             ['alpha_0', 'alpha_z', 'alpha_lam', 'comp0_0', 'comp0_z', 'comp0_lam', 'comp1_0', 'comp1_z', 'comp1_lam']):
            miscenter_dict['MCMF'][this] = miscenter_dict[glob]
        self.miscenterer = miscentering.MisCentering(miscenter_dict[self.config_mod.DES['DEScentertype']])
        # DES Y3 source P(z)
        fits = fitsio.FITS(self.config_mod.DES['source_Pz_file'])
        self.source_z = {'z': fits['nz_source']['Z_MID'][:]}
        for i in range(2,5):
            self.source_z['BIN%d'%i] = fits['nz_source']['BIN%d'%i][:]
        self.source_z['allbins'] = np.array([self.source_z['BIN%d'%i] for i in range(2,5)])
        # DES Y3 source weights
        self.source_weights = np.loadtxt(self.config_mod.DES['source_weights_file'], unpack=True)
        self.source_weights_mean = np.average(self.source_weights[0]*np.ones(self.source_weights[1:].shape), weights=self.source_weights[1:], axis=1)
        self.source_weights_cum = np.cumsum(self.source_weights[1:], axis=1)
        self.source_weights_cum/= self.source_weights_cum[:,-1][:,None]
        # DES Y3 tomo bin weights
        weights = np.load(self.config_mod.DES['tomo_bin_weight_file'])
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

        Sigma_NFW = lensing.get_Sigma(x, rs, self.rho_c_z, delta_c)
        Delta_Sigma_NFW = lensing.get_Delta_Sigma(x, rs, self.rho_c_z, delta_c)
        Sigma_NFW_mean = Sigma_NFW - Delta_Sigma_NFW

        # NFW surface mass densities at Rmis [mass]
        x_Rmis = R_mis/rs
        Sigma_NFW_at_Rmis = lensing.get_Sigma(x_Rmis, rs, self.rho_c_z, delta_c)
        Delta_Sigma_NFW_at_Rmis = lensing.get_Delta_Sigma(x_Rmis, rs, self.rho_c_z, delta_c)
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
        z_dist_r = np.empty((len(area_bin_arcmin), len(self.source_z['z'])))
        w_dist_b = 3*[None]
        N_r = np.zeros(len(area_bin_arcmin))
        for i in range(len(area_bin_arcmin)):
            # Each tomo bin gets N/3 sources with weights w_dist_b
            for b in range(3):
                this_N = self.rng.poisson(area_bin_arcmin[i] * self.config_mod.DES['source_p_arcmin2'] /3)
                N_r[i]+= this_N
                w_dist_b[b] = self.draw_source_weight(b+2, this_N)
            sum_w = self.tomo_weights[1:] * [np.sum(w_dist_b[b]) for b in range(3)]
            z_dist_r[i] = np.sum(self.source_z['allbins']*sum_w[:,None], axis=0)/np.sum(sum_w)
        z_dist = np.average(self.source_z['allbins'], weights=self.tomo_weights[1:]*self.source_weights_mean, axis=0)
        return z_dist_r, z_dist, N_r


    def get_beta(self, z_cl, z_dist):
        """Return `<beta>` and `<beta**2>` given a redshift distribution."""
        beta = np.array([cosmo.dA_two_z(z_cl, z, self.cosmology)/cosmo.dA(z, self.cosmology) for z in self.source_z['z']])
        beta[self.source_z['z']<=z_cl] = 0
        beta_2d = beta * np.ones(z_dist.shape)
        beta_avg = np.average(beta_2d, weights=z_dist, axis=1)
        return beta_avg


    def apply_cl_mem_contamination(self, z, Rmis, g_t):


        A = lensing.boost_get_A(self.boost_dict, 'Gausssmooth', self.boost_dict['z_arr'], z, self.cat['richness'], self.r_arr, Rmis)
        reduced_shear_cont = 1/(1+A) * g_t

        return reduced_shear_cont


    def __call__(self, cat):
        """Wrapper function: Call all workers and return everything."""
        self.cat = cat
        z_cl = cat['REDSHIFT']
        self.M_Delta = cat['Mwl_DES_200']

        self.tomo_weights = self.w_interp(z_cl)
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
        g_t_mis, g_t_cen, R_mis = self.get_miscentered_gt(z_cl, beta_avg)
        g_t_cont = self.apply_cl_mem_contamination(z_cl, R_mis, g_t_mis)

        # Error on shear is shape_noise / sqrt(N(r))
        good_idx = (np.isfinite(g_t_cont)&(N_r>4)).nonzero()[0]
        g_t = g_t_cont[good_idx]
        g_t_err = self.config_mod.DES['shape_noise'] / np.sqrt(N_r[good_idx])
        g_t+= g_t_err*self.rng.standard_normal(len(g_t))

        res_dict = {'r_Mpch': self.r_arr[good_idx],
                    'r_arcmin': self.r_arcmin[good_idx],
                    'shear_cen': g_t_cen[good_idx],
                    'shear_mis': g_t_mis[good_idx],
                    'shear_noerr': g_t_cont[good_idx],
                    'shear': g_t,
                    'shear_err': g_t_err,
                    'source_dist': source_dist,
                    'beta': beta_avg[good_idx],
                    'tomo_weights': self.tomo_weights,
                   }
        return res_dict


################################################################################

class MockUpHSTWL:

    def __init__(self, cosmology, WLconfigname):
        self.cosmology = cosmology
        self.config_mod = importlib.import_module(WLconfigname)
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
        beta[self.pz[0]<=z_cl] = 0
        beta2 = beta**2
        beta_avg = np.average(beta, weights=self.pz[1])
        beta2_avg = np.average(beta2, weights=self.pz[1])
        return beta_avg, beta2_avg


    def get_gt(self, z, beta_avg, beta2_avg):
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

        # NFW halo [mass][radius]
        Sigma_NFW = lensing.get_Sigma(x, rs, self.rho_c_z, delta_c)
        Delta_Sigma_NFW = lensing.get_Delta_Sigma(x, rs, self.rho_c_z, delta_c)

        # Beta correction [Radius][Mass]
        betaratio = beta2_avg/beta_avg**2
        betaCorr = 1 + Sigma_NFW/Sigma_c*(betaratio-1)
        g_t = betaCorr * Delta_Sigma_NFW/Sigma_c / (1-Sigma_NFW/Sigma_c)

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
        good_idx = ((r_min<=all_edges)&(all_edges<=r_max)).nonzero()[0]
        these_edges = all_edges[good_idx]
        these_edges = np.append(np.insert(these_edges, 0, r_min), r_max)
        self.r_arr = 2/3 * (these_edges[1:]**3-these_edges[:-1]**3)/(these_edges[1:]**2-these_edges[:-1]**2)
        self.r_arcmin = self.r_arr / self.Dl * 60*180/np.pi
        self.r_deg = self.r_arcmin / 60
        self.r_arcmin_edges = these_edges / self.Dl * 60*180/np.pi

        N_r = self.get_N_source_gals()
        beta_avg, beta2_avg = self.get_beta(z_cl)
        g_t = self.get_gt(z_cl, beta_avg, beta2_avg)

        # Error on shear is shape_noise / sqrt(N(r))
        good_idx = (N_r>4).nonzero()[0]
        g_t = g_t[good_idx]
        g_t_err = self.config_mod.HST['shape_noise'] / np.sqrt(N_r[good_idx])
        g_t+= g_t_err*self.rng.standard_normal(len(g_t))

        return self.r_arr[good_idx], self.r_deg[good_idx], g_t, g_t_err, self.pz



if __name__ == '__main__':
    main()
