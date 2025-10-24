import numpy as np
from math import sqrt as msqrt

from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simpson
from scipy.special import ndtr

from colossus.cosmology import cosmology as colossus_cosmology
from colossus.lss import bias

import multivariate_normal as cy_multivariate_normal
import convolution
import scaling_relations


sqrt2pi = msqrt(2.*np.pi)
# For interpolation of HMF in redshift
z_tol = .001


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MultiObsConvolution.get_P_multiobs_z(*arg)

################################################################################


class MultiObsConvolution:

    def __init__(self, observable_pairs,
                 zmin, zmax, Nz,
                 lambda_cut, richness_scatter_model,
                 do_bias=False,
                 NPROC=0):
        # Sigma-clipping in convolutions
        self.N_sigma = np.array([5, 3])
        # Number of processes (0 for simple loop)
        self.NPROC = NPROC
        # Cut in richness
        self.lambda_cut = lambda_cut
        # Poisson-style scatter in richness
        self.richness_scatter_model = richness_scatter_model
        # Multiply mass function with halo bias
        self.do_bias = do_bias
        self.observable_pairs = observable_pairs
        self.z_arr = np.linspace(zmin, zmax, Nz)

    def execute(self, HMF, scaling, cosmology=None):
        """Return dict with multi-obs mass functions for each pair of
        observables."""
        self.HMF = HMF
        self.scaling = scaling
        # For interpolation of HMF
        with np.errstate(divide='ignore'):
            self.lndNdlnM = np.log(self.HMF['dNdlnM'])
        self.Delta_lnM = HMF['lnM_arr'][1]-self.HMF['lnM_arr'][0]
        # Check length of HMF mass array for compression factor
        self.HMF['len_M'] = len(self.HMF['lnM_arr'])
        # Needed for halo bias
        if self.do_bias:
            colossus_params = {'flat': True,
                               'H0': 100*cosmology['h'],
                               'Om0': cosmology['Omega_m'],
                               'Ob0': cosmology['Omega_b'],
                               'sigma8': cosmology['sigma8'],
                               'ns': cosmology['n_s']}
            this_colossus_cosmo = colossus_cosmology.setCosmology('myCosmo', colossus_params)
        # Pre-compute the intrinsic scatter convolutions
        output_dict = {'lnM_arr': HMF['lnM_arr'],
                       'z_arr': self.z_arr}
        for pair_idx, pair_name in enumerate(self.observable_pairs):
            output_dict['{}_lndNdlnM'.format(pair_name)] = self.get_P_multiobs_allz(pairname=pair_name)
        return output_dict

    def dNdlnM_at_z(self, z):
        """Return dN/dlnM at redshift `z` by linearly interpolating ln(dN/dlnM)."""
        if np.any(np.abs(z - self.HMF['z_arr']) < z_tol):
            lndNdlnM = self.lndNdlnM[np.argmin(np.abs(z - self.HMF['z_arr'])), :]
        else:
            idx_lo = (self.HMF['z_arr'] < z).nonzero()[0][-1]
            Delta_x = self.HMF['z_arr'][idx_lo+1] - self.HMF['z_arr'][idx_lo]
            with np.errstate(invalid='ignore'):
                Delta_y = self.lndNdlnM[idx_lo+1, :] - self.lndNdlnM[idx_lo, :]
            Delta_y[np.isnan(Delta_y)] = 0.
            lndNdlnM = self.lndNdlnM[idx_lo, :] + Delta_y * (z - self.HMF['z_arr'][idx_lo]) / Delta_x
        return np.exp(lndNdlnM)

    def get_P_multiobs_allz(self, pairname):
        """Return P(obs, xi | M, z, p) for each redshift in z_arr. Optional
        multiprocess."""
        # Write to self to make function pickleable for multiprocessing
        self.pairname = pairname

        if self.NPROC == 0:
            # Iterate through redshift array
            P_obs_grid = np.array([self.get_P_multiobs_z(z) for z in self.z_arr])
        else:
            # Launch and execute a multiprocessing pool
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*len(self.z_arr), self.z_arr)
                P_obs_grid = np.array(pool.map(unwrap_self_f, argin))
        return P_obs_grid

    def get_P_multiobs_z(self, z):
        """Decide whether it's a 2D or 3D observable array or whether it's the
        fancy DES stuff."""
        # Which HMF convolution?
        if self.pairname == 'SZ':
            # dN/dlnzeta
            return self.get_P_zeta_z(z)
        elif self.pairname == 'richness_SZ':
            # dN/dlnzeta/dlnlambda
            return self.get_P_zeta_lambda_lognormal_z(z)
        elif 'SZ_lambdacut_' in self.pairname:
            # dN/dlnzeta given lambda>lambda_min
            survey = self.pairname[13:]
            return self.get_P_zeta_lambdacut_z(z, survey)

    def get_Nbins_array(self, std):
        """Return number of bins and array that satisfy that std/Delta_lnM is
        covered self.N_sigma times. 0 is Nbins_hilo[1] first element."""
        # Number of bins below and above (without 0). At least 1
        Nbins_hilo = (self.N_sigma * std / self.Delta_lnM).astype(int) + 1
        # We want uneven total number. Add 1 to lower if needed
        if (Nbins_hilo[0]+Nbins_hilo[1]+1) % 2 == 0:
            Nbins_hilo[0] += 1
        lnobs_arr = self.Delta_lnM * np.linspace(-Nbins_hilo[0], Nbins_hilo[1], Nbins_hilo[0]+Nbins_hilo[1]+1)
        return Nbins_hilo, lnobs_arr

    def get_Nbins_array_vec(self, std):
        """Return number of bins and array that satisfy that std/Delta_lnM is
        covered self.N_sigma times. This is same as `get_Nbins_array` but for
        array of `std`. Too slow so propably shouldn't be used. 0 is
        Nbins_hilo[1] first element."""
        Nbins_hilo = (self.N_sigma[None, :] * std[:, None] / self.Delta_lnM).astype(int) + 1
        Nbins_hilo[(Nbins_hilo[:, 0]+Nbins_hilo[:, 1]+1) % 2 == 0, 0] += 1
        lnobs_arr = [self.Delta_lnM * np.linspace(-Nbins_hilo[i, 0], Nbins_hilo[i, 1], Nbins_hilo[i, 0]+Nbins_hilo[i, 1]+1)
                     for i in range(len(Nbins_hilo))]
        return Nbins_hilo, lnobs_arr

    def get_P_zeta_z(self, z):
        """Return dN/dlnzeta."""
        dN_dlnM = self.dNdlnM_at_z(z)
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        Nbin = self.scaling['Dsz'] * dlnM_dlnzeta / self.Delta_lnM
        HMF_1d = gaussian_filter1d(dN_dlnM, Nbin, mode='constant')
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_1d = np.log(HMF_1d)
        return lnHMF_1d

    def get_P_zeta_lambdacut_z(self, z, SZsurvey):
        """Return dN/dlnzeta accounting for richness confirmation."""
        dN_dlnM = self.dNdlnM_at_z(z)
        # Halo bias
        if self.do_bias:
            Tinker_bias = bias.haloBias(np.exp(self.HMF['lnM_arr']), model='tinker10', z=z, mdef='200c')
            dN_dlnM *= Tinker_bias
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs('richness', self.scaling, z=z)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        Dsz = self.scaling['Dsz']
        Drich = scaling_relations.richnessscatter(z, self.scaling)
        if self.richness_scatter_model in ['lognormal', 'lognormalGaussPoisson', 'lognormalGausssuperPoisson']:
            covmat = np.array([[Drich**2, self.scaling['rhoSZrichness']*Drich*Dsz],
                               [self.scaling['rhoSZrichness']*Drich*Dsz, Dsz**2]])[None, :, :]
        elif self.richness_scatter_model == 'lognormalrelPoisson':
            richness_ = np.exp(scaling_relations.lnmass2lnobs('richness', self.HMF['lnM_arr'], z, self.scaling))
            Drich = np.sqrt(Drich**2 + 1/richness_)
            covmat = (np.array([[1., self.scaling['rhoSZrichness']*Dsz],
                               [self.scaling['rhoSZrichness']*Dsz, Dsz**2]])[None, :, :]
                      * np.ones((len(richness_), 2, 2)))
            covmat[:, 0, :] *= Drich[:, None]
            covmat[:, :, 0] *= Drich[:, None]
        covmat_lnM = covmat * Jacobian[None, :, :]
        # Number of bins and arrays for each observable
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(np.amax(covmat_lnM[:, 1, 1])))
        Nbins_zeta = Nbins_zeta[None, :] * np.ones((self.HMF['len_M'], 2), dtype=int)
        lnmass_lambda_mean = (self.HMF['lnM_arr'][:, None]
                              + (covmat_lnM[:, 0, 1]/covmat_lnM[:, 1, 1])[:, None]*lnzeta_arr[None, :])
        lnlambda_mean = scaling_relations.lnmass2lnobs('richness', lnmass_lambda_mean, z, self.scaling)
        lnmass_lambda_std = np.sqrt(covmat_lnM[:, 0, 0] - covmat_lnM[:, 0, 1]**2/covmat_lnM[:, 1, 1])
        lnlambda_std = lnmass_lambda_std/dlnM_dlnobs
        # Cumulative distribution
        if self.richness_scatter_model in ['lognormal', 'lognormalrelPoisson']:
            P_lambda_gtr_cut = ndtr((lnlambda_mean-np.log(self.lambda_cut[SZsurvey](z)))/lnlambda_std[:, None])
        elif self.richness_scatter_model in ['lognormalGaussPoisson', 'lognormalGausssuperPoisson']:
            lnlambda_arr = np.linspace(lnlambda_mean-3*lnlambda_std, lnlambda_mean+3*lnlambda_std, 64)
            lambda_arr = np.exp(lnlambda_arr)
            P_lambda_intrinsic = np.exp(-.5*((lnlambda_arr-lnlambda_mean)/lnlambda_std)**2) / (sqrt2pi*lambda_arr*lnlambda_std)
            if self.richness_scatter_model == 'lognormalGaussPoisson':
                lambda_obs_std = np.sqrt(lambda_arr)
            elif self.richness_scatter_model == 'lognormalGausssuperPoisson':
                lambda_obs_std = np.sqrt(lambda_arr+10.) * (1.08 + .45*(z-.6))
            P_lambdaobs_gtr_cut = ndtr((lambda_arr-self.lambda_cut[SZsurvey](z))/lambda_obs_std)
            P_lambda_gtr_cut = simpson(P_lambdaobs_gtr_cut*P_lambda_intrinsic, lambda_arr, axis=0)
        # Convolution
        kernels = (P_lambda_gtr_cut
                   * np.exp(-.5*lnzeta_arr[None, :]**2/covmat_lnM[:, 1, 1][:, None])
                   / np.sqrt(2*np.pi*covmat_lnM[:, 1, 1])[:, None])
        HMF_1d = convolution.convolve_HMF_1obs_varkernel(dN_dlnM, self.Delta_lnM, kernels, Nbins_zeta)
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_1d = np.log(HMF_1d)
        return lnHMF_1d

    def get_P_zeta_lambda_lognormal_z(self, z):
        """Return P(obs, zeta | M, z[z_id], p) for constant correlated
        scatter."""
        dN_dlnM = self.dNdlnM_at_z(z)
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs('richness', self.scaling, z=z)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        Drich = scaling_relations.richnessscatter(z, self.scaling)
        Dsz = self.scaling['Dsz']
        covmat = np.array([[Drich**2, self.scaling['rhoSZrichness']*Drich*Dsz],
                           [self.scaling['rhoSZrichness']*Drich*Dsz, Dsz**2]])
        covmat_lnM = covmat * Jacobian
        # Number of bins and arrays for each observable
        Nbins_obs, lnobs_arr = self.get_Nbins_array(msqrt(covmat_lnM[0, 0]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[1, 1]))
        # Get the scatter kernel [lnobs, lnzeta]
        kernel = cy_multivariate_normal.bivariate_normal(lnobs_arr, lnzeta_arr, covmat_lnM)
        # Convolution
        HMF_2d = convolution.convolve_HMF_2obs_fixedkernel(dN_dlnM, self.Delta_lnM, kernel, Nbins_obs, Nbins_zeta)
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_2d = np.log(HMF_2d)
        return lnHMF_2d
