import numpy as np
from math import sqrt as msqrt

from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr

from colossus.cosmology import cosmology as colossus_cosmology
from colossus.lss import bias

import multivariate_normal as cy_multivariate_normal
import convolution
import scaling_relations

richscatter = {'base': 'Drichness',
               'ext': 'Drichness_ext'}


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MultiObsConvolution.get_P_multiobs_z(*arg)

################################################################################


class MultiObsConvolution:

    def __init__(self, observable_pairs,
                 pairs_zmin, pairs_zmax, pairs_Nz,
                 lambda_cut, richness_scatter_model,
                 do_bias=False,
                 NPROC=0):
        # Sigma-clipping in convolutions
        self.N_sigma = np.array([4, 3])
        # Sparsity of returned arrays
        self.compression = 2
        # Number of processes (0 for simple loop)
        self.NPROC = NPROC
        # Cut in richness
        self.lambda_cut = lambda_cut
        # Poisson-style scatter in richness
        self.richness_scatter_model = richness_scatter_model
        # Multiply mass function with halo bias
        self.do_bias = do_bias

        all_pairnames = ['SZ', 'SZ_lambdacut_base_shallow', 'SZ_lambdacut_base_deep',
                         'SZ_lambdacut_ext_shallow', 'SZ_lambdacut_ext_deep',
                         'richness_SZ_base', 'richness_SZ_ext',
                         'DESwl_richness_SZ_base']

        self.observable_pairs, self.pairs_zmin, self.pairs_zmax, self.pairs_Nz = [], [], [], []
        for pair, zmin, zmax, Nz in zip(observable_pairs, pairs_zmin, pairs_zmax, pairs_Nz):
            if pair in all_pairnames:
                self.observable_pairs.append(pair)
                self.pairs_zmin.append(zmin)
                self.pairs_zmax.append(zmax)
                self.pairs_Nz.append(Nz)

    ############################################################################

    def execute(self, HMF, scaling, cosmology=None):
        """Return dict with multi-obs mass functions for each pair of
        observables."""
        self.HMF = HMF
        self.scaling = scaling
        # Set up interpolation for HMF
        with np.errstate(divide='ignore'):
            lnHMF_in = np.log(self.HMF['dNdlnM'])
        self.HMF_interp = interp1d(self.HMF['z_arr'], lnHMF_in, axis=0, assume_sorted=True)
        self.Delta_lnM = HMF['lnM_arr'][1]-self.HMF['lnM_arr'][0]
        # Check length of HMF mass array for compression factor
        self.HMF['len_M'] = len(self.HMF['lnM_arr'])
        assert (self.HMF['len_M']-1) % self.compression == 0, "HMF has non-standard shape"
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
        output_dict = {'lnM_arr': HMF['lnM_arr'][::self.compression]}
        for pair_idx, pair_name in enumerate(self.observable_pairs):
            z_arr = np.linspace(self.pairs_zmin[pair_idx], self.pairs_zmax[pair_idx], self.pairs_Nz[pair_idx])
            output_dict[pair_name] = self.get_P_multiobs_allz(pairname=pair_name,
                                                              z_arr=z_arr)
            output_dict['%s_z' % pair_name] = z_arr
        return output_dict

    def get_P_multiobs_allz(self, pairname, z_arr):
        """Return P(obs, xi | M, z, p) for each redshift in z_arr. Optional
        multiprocess."""
        # Write to self to make function pickleable for multiprocessing
        self.pairname = pairname
        # Iterate through z_arr
        if self.NPROC == 0:
            P_obs_grid = np.array([self.get_P_multiobs_z(z) for z in z_arr])
        else:
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*len(z_arr), z_arr)
                P_obs_grid = np.array(pool.map(unwrap_self_f, argin))
        return P_obs_grid

    def get_P_multiobs_z(self, z):
        """Decide whether it's a 2D or 3D observable array or whether it's the
        fancy DES stuff."""
        # dN/dlnzeta
        if self.pairname == 'SZ':
            return self.get_P_zeta_z(z)
        # Do we have richness?
        if 'base' in self.pairname:
            richness = 'base'
        elif 'ext' in self.pairname:
            richness = 'ext'
        # dN/dlnzeta/dlnlambda
        if self.pairname in ['richness_SZ_base', 'richness_SZ_ext']:
            return self.get_P_zeta_lambda_lognormal_z(z, richness)
        # dN/dlnzeta given lambda>lambda_min
        elif self.pairname in ['SZ_lambdacut_base_shallow', 'SZ_lambdacut_ext_shallow']:
            return self.get_P_zeta_lambdacut_z(z, SZsurvey='shallow', richness=richness)
        elif self.pairname in ['SZ_lambdacut_base_deep', 'SZ_lambdacut_ext_deep']:
            return self.get_P_zeta_lambdacut_z(z, SZsurvey='deep', richness=richness)
        # dN/dlnMwl/dlnlambda/dlnzeta
        elif self.pairname == 'DESwl_richness_SZ_base':
            return self.get_P_DES_lambda_zeta_z(z, 'base')
        return 1

    def get_Nbins_array(self, std):
        """Return number of bins and array that satisfy that std/Delta_lnM is
        covered self.N_sigma times. 0 is Nbins_hilo[1] first element."""
        # Number of bins below and above (without 0). At least 1
        Nbins_hilo = (self.N_sigma * std / self.Delta_lnM).astype(int) +1
        # We want uneven total number. Add 1 to lower if needed
        if (Nbins_hilo[0]+Nbins_hilo[1]+1) % 2 == 0:
            Nbins_hilo[0]+= 1
        lnobs_arr = self.Delta_lnM * np.linspace(-Nbins_hilo[0], Nbins_hilo[1], Nbins_hilo[0]+Nbins_hilo[1]+1)
        return Nbins_hilo, lnobs_arr

    def get_P_zeta_z(self, z):
        """Return dN/dlnzeta."""
        dN_dlnM = np.exp(self.HMF_interp(z))
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        Nbin = self.scaling['Dsz'] * dlnM_dlnzeta / self.Delta_lnM
        HMF_1d = gaussian_filter1d(dN_dlnM, Nbin, mode='constant')
        # Compress
        HMF_1d = HMF_1d[::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_1d = np.log(HMF_1d)
        return lnHMF_1d

    def get_P_zeta_lambdacut_z(self, z, SZsurvey, richness):
        """Return dN/dlnzeta accounting for richness confirmation."""
        dN_dlnM = np.exp(self.HMF_interp(z))
        # Halo bias
        if self.do_bias:
            Tinker_bias = bias.haloBias(np.exp(self.HMF['lnM_arr']), model='tinker10', z=z, mdef='200c')
            dN_dlnM*= Tinker_bias
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs('richness_%s' % richness, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        Dsz = self.scaling['Dsz']
        Drich = self.scaling[richscatter[richness]]
        if self.richness_scatter_model in ['lognormal', 'lognormalGaussPoisson']:
            covmat = np.array([[Drich**2, self.scaling['rhoSZrichness']*Drich*Dsz],
                               [self.scaling['rhoSZrichness']*Drich*Dsz, Dsz**2]])[None, :, :]
        elif self.richness_scatter_model == 'lognormalrelPoisson':
            richness_ = np.exp(scaling_relations.lnmass2lnobs('richness_%s' % richness, self.HMF['lnM_arr'], z, self.scaling))
            Drich = np.sqrt(Drich**2 + 1/richness_)
            covmat = (np.array([[1., self.scaling['rhoSZrichness']*Dsz],
                               [self.scaling['rhoSZrichness']*Dsz, Dsz**2]])[None, :, :]
                      * np.ones((len(richness_),2,2)))
            covmat[:, 0, :] *= Drich[:, None]
            covmat[:, :, 0] *= Drich[:, None]
        covmat_lnM = covmat * Jacobian[None, :, :]
        # Number of bins and arrays for each observable
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(np.amax(covmat_lnM[:, 1,1])))
        Nbins_zeta = Nbins_zeta[None, :] * np.ones((self.HMF['len_M'], 2), dtype=int)
        lnmass_lambda_mean = self.HMF['lnM_arr'][:, None] + (covmat_lnM[:,0, 1]/covmat_lnM[:,1,1])[:, None]*lnzeta_arr[None, :]
        lnlambda_mean = scaling_relations.lnmass2lnobs('richness_%s' % richness, lnmass_lambda_mean, z, self.scaling)
        lnmass_lambda_std = np.sqrt(covmat_lnM[:,0,0] - covmat_lnM[:,0, 1]**2/covmat_lnM[:,1,1])
        lnlambda_std = lnmass_lambda_std/dlnM_dlnobs
        # Cumulative Gaussian
        if self.richness_scatter_model in ['lognormal', 'lognormalrelPoisson']:
            P_lambda_gtr_cut = ndtr((lnlambda_mean-np.log(self.lambda_cut[SZsurvey](z)))/lnlambda_std[:, None])
        elif self.richness_scatter_model == 'lognormalGaussPoisson':
            lnlambda_arr = np.linspace(lnlambda_mean-3*lnlambda_std, lnlambda_mean+3*lnlambda_std, 32)
            lambda_arr = np.exp(lnlambda_arr)
            P_lambda_intrinsic = (np.exp(-.5*(lnlambda_arr-lnlambda_mean)**2/lnlambda_std**2)
                                  / (msqrt(2*np.pi)*lambda_arr*lnlambda_std))
            P_lambda_gtr_cut = ndtr((lambda_arr-self.lambda_cut[SZsurvey](z))/np.sqrt(lambda_arr))
            P_lambda = np.trapezoid(P_lambda_gtr_cut*P_lambda_intrinsic, lambda_arr, axis=0)
            P_lambda_gtr_cut = P_lambda
        # Convolution
        kernels = (P_lambda_gtr_cut * np.exp(-.5*lnzeta_arr[None, :]**2/covmat_lnM[:, 1,1][:, None])
                   / np.sqrt(2*np.pi*covmat_lnM[:, 1,1])[:, None])
        HMF_1d = convolution.convolve_HMF_1obs_varkernel(dN_dlnM, self.Delta_lnM, kernels, Nbins_zeta)
        # Compress
        HMF_1d = HMF_1d[::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_1d = np.log(HMF_1d)
        return lnHMF_1d

    def get_P_zeta_lambda_lognormal_z(self, z, richness):
        """Return P(obs, zeta | M, z[z_id], p) for constant correlated
        scatter."""
        dN_dlnM = np.exp(self.HMF_interp(z))
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs('richness_%s' % richness, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        Drich = self.scaling[richscatter[richness]]
        Dsz = self.scaling['Dsz']
        covmat = np.array([[Drich**2, self.scaling['rhoSZrichness']*Drich*Dsz],
                           [self.scaling['rhoSZrichness']*Drich*Dsz, Dsz**2]])
        covmat_lnM = covmat * Jacobian
        # Number of bins and arrays for each observable
        Nbins_obs, lnobs_arr = self.get_Nbins_array(msqrt(covmat_lnM[0,0]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[1, 1]))
        # Get the scatter kernel [lnobs, lnzeta]
        kernel = cy_multivariate_normal.bivariate_normal(lnobs_arr, lnzeta_arr, covmat_lnM)
        # Convolution
        HMF_2d = convolution.convolve_HMF_2obs_fixedkernel(dN_dlnM, self.Delta_lnM, kernel, Nbins_obs, Nbins_zeta)
        # Compress
        HMF_2d = HMF_2d[::self.compression, ::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_2d = np.log(HMF_2d)
        return lnHMF_2d

    def get_P_2obs_z(self, obsname, covmat, z):
        """Return P(obs, zeta | M, z[z_id], p) for constant correlated
        scatter."""
        dN_dlnM = np.exp(self.HMF_interp(z))
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs(obsname, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Number of bins and arrays for each observable
        Nbins_obs, lnobs_arr = self.get_Nbins_array(msqrt(covmat_lnM[0, 0]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[1, 1]))
        # Get the scatter kernel [lnobs, lnzeta]
        kernel = cy_multivariate_normal.bivariate_normal(lnobs_arr, lnzeta_arr, covmat_lnM)
        # Convolution
        HMF_2d = convolution.convolve_HMF_2obs_fixedkernel(dN_dlnM, self.Delta_lnM, kernel, Nbins_obs, Nbins_zeta)
        # Compress
        HMF_2d = HMF_2d[::self.compression, ::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_2d = np.log(HMF_2d)
        return lnHMF_2d

    def get_P_2obs_DES_z(self, obsname, z):
        """Return P(DES_WL, zeta | M, z, p) with correlated scatter."""
        dN_dlnM = np.exp(self.HMF_interp(z))
        # (Mass-dependent) covariance matrices
        # Main component
        cov_base = np.array([[1, self.scaling['rhoSZWL']*self.scaling['Dsz']],
                             [self.scaling['rhoSZWL']*self.scaling['Dsz'], self.scaling['Dsz']**2]])
        DES_scatter = scaling_relations.WLscatter('WLDES', self.HMF['lnM_arr'], z, self.scaling)
        covmat_main = (cov_base
                       * np.array([DES_scatter**2, DES_scatter, DES_scatter, np.ones(len(DES_scatter))]).T.reshape(len(DES_scatter),2,2))
        covmat = covmat_main
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs(obsname, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Scatter kernels [lnobs, lnzeta]
        kernels = [None]*self.HMF['len_M']
        Nbins_obs = np.empty((self.HMF['len_M'], 2), dtype=int)
        Nbins_zeta = np.empty((self.HMF['len_M'], 2), dtype=int)
        for i in range(self.HMF['len_M']):
            # Number of bins and arrays for each observable
            Nbins_obs[i], lnobs_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 0, 0]))
            Nbins_zeta[i], lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 1, 1]))
            # Multivariate Gaussian kernel
            kernels[i] = cy_multivariate_normal.bivariate_normal(lnobs_arr, lnzeta_arr, covmat_lnM[i])
        # Actual convolution
        HMF_2d = convolution.convolve_HMF_2obs_varkernel(dN_dlnM, self.Delta_lnM, kernels, Nbins_obs, Nbins_zeta)
        # Compress
        HMF_2d = HMF_2d[::self.compression, ::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_2d = np.log(HMF_2d)
        return lnHMF_2d

    def get_P_DES_lambda_zeta_z(self, z, richness):
        """Return P(DES_WL, lambda, zeta | M, z, p) with correlated scatter."""
        # Mass function at this redshift
        lndN_dlnM = self.HMF_interp(z)
        # (Mass-dependent) covariance matrices in observable space
        DES_scatter = scaling_relations.WLscatter('WLDES', self.HMF['lnM_arr'], z, self.scaling)
        Drich = np.full(self.HMF['len_M'], self.scaling[richscatter[richness]])
        if self.richness_scatter_model == 'lognormalrelPoisson':
            richness_ = np.exp(scaling_relations.lnmass2lnobs('richness_%s' % richness, self.HMF['lnM_arr'], z, self.scaling))
            Drich = np.sqrt(Drich**2 + 1/richness_)
        covmat = np.array([[DES_scatter**2, self.scaling['rhoWLrichness']*DES_scatter*Drich, self.scaling['rhoSZWL']*DES_scatter*self.scaling['Dsz']],
                           [self.scaling['rhoWLrichness']*DES_scatter*Drich, Drich**2, self.scaling['rhoSZrichness']*self.scaling['Dsz']*Drich],
                           [self.scaling['rhoSZWL']*DES_scatter*self.scaling['Dsz'], self.scaling['rhoSZrichness']*self.scaling['Dsz']*Drich, np.full(self.HMF['len_M'], self.scaling['Dsz']**2)]]).T
        # Convert observable covmat into covmat in ln mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnrichness = scaling_relations.dlnM_dlnobs('richness_%s' % richness, self.scaling)
        dlnM_dlnMwl = scaling_relations.dlnM_dlnobs('WLDES', self.scaling)
        Jacobian = np.array([[dlnM_dlnMwl**2,               dlnM_dlnMwl*dlnM_dlnrichness,  dlnM_dlnMwl*dlnM_dlnzeta],
                             [dlnM_dlnMwl*dlnM_dlnrichness, dlnM_dlnrichness**2,           dlnM_dlnrichness*dlnM_dlnzeta],
                             [dlnM_dlnMwl*dlnM_dlnzeta,     dlnM_dlnrichness*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Scatter kernels [lnWL, lnrichness, lnzeta]
        lnkernels = [None]*self.HMF['len_M']
        Nbins_obs0 = np.empty((self.HMF['len_M'], 2), dtype=int)
        Nbins_obs1 = np.empty((self.HMF['len_M'], 2), dtype=int)
        Nbins_zeta = np.empty((self.HMF['len_M'], 2), dtype=int)
        for i in range(self.HMF['len_M']):
            # Number of bins and arrays for each observable
            Nbins_obs0[i], lnobs0_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 0, 0]))
            Nbins_obs1[i], lnobs1_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 1, 1]))
            Nbins_zeta[i], lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 2, 2]))
            # Multivariate Gaussian kernel
            # kernels[i] = cy_multivariate_normal.trivariate_normal(lnobs0_arr, lnobs1_arr, lnzeta_arr, cov=covmat_lnM[i])
            X = np.moveaxis(np.array(np.meshgrid(lnobs0_arr, lnobs1_arr, lnzeta_arr, indexing='ij')), 0, -1)
            lnkernels[i] = cy_multivariate_normal.lnpdf(X, covmat_lnM[i])
        # Actual convolution
        HMF_3d = convolution.convolve_HMF_3obs_varkernel(lndN_dlnM, self.Delta_lnM, lnkernels, Nbins_obs0, Nbins_obs1, Nbins_zeta)
        # Compress
        HMF_3d = HMF_3d[::self.compression, ::self.compression, ::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_3d = np.log(HMF_3d)
        return lnHMF_3d

    def get_P_3obs_DES_z(self, obsnames, z):
        """Return P(DES_WL, obs_1, zeta | M, z, p) with correlated scatter."""
        # Mass function at this redshift
        dN_dlnM = np.exp(self.HMF_interp(z))
        # (Mass-dependent) covariance matrices
        if obsnames[1] == 'richness':
            cov_base = np.array([[1, self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['rhoSZWL']*self.scaling['Dsz']],
                                 [self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['Drichness']**2, self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness']],
                                 [self.scaling['rhoSZWL']*self.scaling['Dsz'], self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness'], self.scaling['Dsz']**2]])
        DES_scatter = scaling_relations.WLscatter('WLDES', self.HMF['lnM_arr'], z, self.scaling)
        covmat = cov_base * np.array([DES_scatter**2, DES_scatter, DES_scatter,
                                      DES_scatter, np.ones(len(DES_scatter)), np.ones(len(DES_scatter)),
                                      DES_scatter, np.ones(len(DES_scatter)), np.ones(len(DES_scatter))]).T.reshape(len(DES_scatter),3,3)
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = [scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames]
        Jacobian = np.array([[dlnM_dlnobs[0]**2,             dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[0]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[1]**2,             dlnM_dlnobs[1]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnzeta,   dlnM_dlnobs[1]*dlnM_dlnzeta,   dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Scatter kernels [lnWL, lnobs, lnzeta]
        kernels = [None]*self.HMF['len_M']
        Nbins_obs0 = np.empty((self.HMF['len_M'], 2), dtype=int)
        Nbins_obs1 = np.empty((self.HMF['len_M'], 2), dtype=int)
        Nbins_zeta = np.empty((self.HMF['len_M'], 2), dtype=int)
        for i in range(self.HMF['len_M']):
            # Number of bins and arrays for each observable
            Nbins_obs0[i], lnobs0_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 0, 0]))
            Nbins_obs1[i], lnobs1_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 1, 1]))
            Nbins_zeta[i], lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[i, 2, 2]))
            # Multivariate Gaussian kernel
            kernels[i] = cy_multivariate_normal.trivariate_normal(lnobs0_arr, lnobs1_arr, lnzeta_arr, cov=covmat_lnM[i])
        # Actual convolution
        HMF_3d = convolution.convolve_HMF_3obs_varkernel(dN_dlnM, self.Delta_lnM, kernels, Nbins_obs0, Nbins_obs1, Nbins_zeta)
        # Compress
        HMF_3d = HMF_3d[::self.compression, ::self.compression, ::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_3d = np.log(HMF_3d)
        return lnHMF_3d

    def get_P_3obs_z(self, obsnames, covmat, z):
        """Return P(obs0, obs1, zeta | M, z(z_id), p) for constant correlated
        scatter."""
        dN_dlnM = np.exp(self.HMF_interp(z))
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = [scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames]
        Jacobian = np.array([[dlnM_dlnobs[0]**2,             dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[0]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[1]**2,             dlnM_dlnobs[1]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnzeta,   dlnM_dlnobs[1]*dlnM_dlnzeta,   dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Number of bins and observable arrays
        Nbins_obs0, lnobs0_arr = self.get_Nbins_array(msqrt(covmat_lnM[0, 0]))
        Nbins_obs1, lnobs1_arr = self.get_Nbins_array(msqrt(covmat_lnM[1, 1]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[2,2]))
        # Get the scatter kernel [lnobs, lnzeta]
        kernel = cy_multivariate_normal.trivariate_normal(lnobs0_arr, lnobs1_arr, lnzeta_arr, covmat)
        # Convolution
        HMF_3d = convolution.convolve_HMF_3obs_fixedkernel(dN_dlnM, self.Delta_lnM, kernel, Nbins_obs0, Nbins_obs1, Nbins_zeta)
        # Compress
        HMF_3d = HMF_3d[::self.compression, ::self.compression, ::self.compression]
        # We know we're doing log(0)...
        with np.errstate(divide='ignore'):
            lnHMF_3d = np.log(HMF_3d)
        return lnHMF_3d
