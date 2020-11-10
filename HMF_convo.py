from __future__ import division, print_function
import numpy as np
from math import sqrt as msqrt
import time

from multiprocessing import Pool
from scipy.interpolate import RectBivariateSpline

import multivariate_normal as cy_multivariate_normal
import convolution, scaling_relations

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MultiObsConvolution.get_P_multiobs_z(*arg)

################################################################################
class MultiObsConvolution:

    def __init__(self, observable_pairs,
                 pairs_zmin, pairs_zmax, pairs_Nz,
                 NPROC):
        # Sigma-clipping in convolutions
        self.N_sigma = np.array([4, 3])
        # Sparsity of returned arrays
        self.compression = 10
        # Number of processes (0 for simple loop)
        self.NPROC = NPROC

        self.pairnames_2d = ['Yx_SZ', 'Mgas_SZ', 'Megacam_SZ', 'richness_SZ']
        self.pairnames_2d_DES = ['DES_SZ',]
        self.pairnames_3d = ['Megacam_Yx_SZ', 'Megacam_Mgas_SZ']
        self.pairnames_3d_DES = ['DES_Yx_SZ', 'DES_Mgas_SZ', 'DES_richness_SZ']

        self.obsnames_dict = {'Yx_SZ': 'Yx',
                              'Mgas_SZ': 'Mgas',
                              'Megacam_SZ': 'WLMegacam',
                              'DES_SZ': 'WLDES',
                              'richness_SZ': 'richness',
                              'Megacam_Yx_SZ': ['WLMegacam', 'Yx'],
                              'Megacam_Mgas_SZ': ['WLMegacam', 'Mgas'],
                              'DES_Yx_SZ': ['WLDES', 'Yx'],
                              'DES_Mgas_SZ': ['WLDES', 'Mgas'],
                              'DES_richness_SZ': ['WLDES', 'richness'],
                              }

        self.observable_pairs, self.pairs_zmin, self.pairs_zmax, self.pairs_Nz = [], [], [], []
        for pair, zmin, zmax, Nz in zip(observable_pairs, pairs_zmin, pairs_zmax, pairs_Nz):
            if (pair in self.pairnames_2d) | (pair in self.pairnames_3d) | (pair in self.pairnames_2d_DES) | (pair in self.pairnames_3d_DES):
                self.observable_pairs.append(pair)
                self.pairs_zmin.append(zmin)
                self.pairs_zmax.append(zmax)
                self.pairs_Nz.append(Nz)


    ############################################################################
    def execute(self, HMF, scaling, covmat):
        """Return dict with multi-obs mass functions for each pair of
        observables."""
        self.HMF = HMF
        self.scaling = scaling
        self.covmat = covmat
        # Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in), kx=1, ky=1)
        self.Delta_lnM = np.log(HMF['M_arr'][1]/self.HMF['M_arr'][0])
        # Check length of HMF mass array for compression factor
        assert (len(HMF['M_arr'])-1)%self.compression==0, "HMF has non-standard shape"
        # Pre-compute the intrinsic scatter convolutions
        output_dict = {'M_arr': HMF['M_arr'][::self.compression]}
        for pair_idx,pair_name in enumerate(self.observable_pairs):
            z_arr = np.linspace(self.pairs_zmin[pair_idx], self.pairs_zmax[pair_idx], self.pairs_Nz[pair_idx])
            if (pair_name in self.pairnames_2d_DES) | (pair_name in self.pairnames_3d_DES):
                this_covmat_ = 0
            else:
                this_covmat_ = self.covmat['cov_%s'%pair_name]
            obsname_s_ = self.obsnames_dict[pair_name]
            output_dict[pair_name] = self.get_P_multiobs_allz(obsname=obsname_s_,
                                                              pairname=pair_name,
                                                              pair_covmat=this_covmat_,
                                                              z_arr=z_arr)
            output_dict['%s_z'%pair_name] = z_arr
        return output_dict


    def get_P_multiobs_allz(self, obsname, pairname, pair_covmat, z_arr):
        """Return P(obs, xi | M, z, p) for each redshift in z_arr. Optional
        multiprocess."""
        # Write to self to make function pickleable for multiprocessing
        self.obsname = obsname
        self.pair_covmat = pair_covmat
        self.pairname = pairname

        if self.NPROC==0:
            # Iterate through redshift array
            P_obs_grid = np.array([self.get_P_multiobs_z(z) for z in z_arr])
        else:
            # Launch and execute a multiprocessing pool
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len(z_arr), z_arr)
            P_obs_grid = np.array(pool.map(unwrap_self_f, argin, chunksize=len(z_arr)//self.NPROC))
            pool.close()
        return P_obs_grid


    def get_P_multiobs_z(self, z):
        """Decide whether it's a 2D or 3D observable array or whether it's the
        fancy DES stuff."""
        # Unpack self (again, because of multiprocessing)
        covmat = self.pair_covmat
        obsname = self.obsname
        pairname = self.pairname
        # Compute 2D or 3D multi-obs HMF convolution
        if pairname in self.pairnames_2d:
            return self.get_P_2obs_z(obsname, covmat, z)
        elif pairname in self.pairnames_2d_DES:
            return self.get_P_2obs_DES_z(obsname, z)
        elif pairname in self.pairnames_3d:
            return self.get_P_3obs_z(obsname, covmat, z)
        elif pairname in self.pairnames_3d_DES:
            return self.get_P_3obs_DES_z(obsname, z)


    def get_Nbins_array(self, std):
        """Return number of bins and array that satisfy that std/Delta_lnM is
        covered self.N_sigma times. 0 is Nbins_hilo[0]st element."""
        # Number of bins below and above (without 0). At least 1
        Nbins_hilo = (self.N_sigma * std / self.Delta_lnM).astype(int) +1
        # We want uneven total number. Add 1 to lower if needed
        if (Nbins_hilo[0]+Nbins_hilo[1]+1)%2 == 0:
            Nbins_hilo[0]+= 1
        lnobs_arr = self.Delta_lnM * np.linspace(-Nbins_hilo[0], Nbins_hilo[1], Nbins_hilo[0]+Nbins_hilo[1]+1)
        return Nbins_hilo, lnobs_arr


    def get_P_2obs_z(self, obsname, covmat, z):
        """Return P(obs, zeta | M, z[z_id], p) for constant correlated
        scatter."""
        dN_dlnM, = np.exp(self.HMF_interp(np.log(z), np.log(self.HMF['M_arr'])))
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs(obsname, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Number of bins and arrays for each observable
        Nbins_obs, lnobs_arr = self.get_Nbins_array(msqrt(covmat_lnM[0,0]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[1,1]))
        # Get the scatter kernel [lnobs, lnzeta]
        kernel = cy_multivariate_normal.bivariate_normal(lnobs_arr, lnzeta_arr, covmat_lnM)
        # Convolution
        HMF_2d = convolution.convolve_HMF_2obs_fixedkernel(dN_dlnM, kernel, Nbins_obs, Nbins_zeta)
        HMF_2d*= self.Delta_lnM
        # Compress
        HMF_2d = HMF_2d[::self.compression,::self.compression]
        # Make it safe to take log
        HMF_2d[HMF_2d==0] = np.nextafter(0,1)
        return np.log(HMF_2d)


    def get_P_2obs_DES_z(self, obsname, z):
        """Return P(DES_WL, zeta | M, z, p) with correlated scatter."""
        dN_dlnM, = np.exp(self.HMF_interp(np.log(z), np.log(self.HMF['M_arr'])))
        # (Mass-dependent) covariance matrices
        # Main component
        cov_base = np.array([[1, self.scaling['rhoSZWL']*self.scaling['Dsz']],
                             [self.scaling['rhoSZWL']*self.scaling['Dsz'], self.scaling['Dsz']**2]])
        DES_scatter = scaling_relations.WLscatter('main', self.HMF['M_arr'], z, self.scaling)
        covmat_main = cov_base * np.array([DES_scatter**2, DES_scatter, DES_scatter, np.ones(len(DES_scatter))]).T.reshape(len(DES_scatter),2,2)
        # Wide component
        # DES_scatter = scaling_relations.WLscatter('wide', self.HMF['M_arr'], z, self.scaling)
        # covmat_wide = cov_base * np.array([DES_scatter**2, DES_scatter, DES_scatter, 1]).T.reshape(len(DES_scatter),2,2)
        covmat = covmat_main # + covmat_wide
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs(obsname, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Scatter kernels [lnobs, lnzeta]
        kernels = [None]*len(self.HMF['M_arr'])
        Nbins_obs = np.empty((len(self.HMF['M_arr']), 2), dtype=int)
        Nbins_zeta = np.empty((len(self.HMF['M_arr']), 2), dtype=int)
        for i in range(len(self.HMF['M_arr'])):
            # Number of bins and arrays for each observable
            Nbins_obs[i], lnobs_arr = self.get_Nbins_array(msqrt(covmat_lnM[i,0,0]))
            Nbins_zeta[i], lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[i,1,1]))
            # Multivariate Gaussian kernel
            kernels[i] = cy_multivariate_normal.bivariate_normal(lnobs_arr, lnzeta_arr, covmat_lnM[i])
        # Actual convolution
        HMF_2d = convolution.convolve_HMF_2obs_varkernel(dN_dlnM, kernels, Nbins_obs, Nbins_zeta)
        HMF_2d*= self.Delta_lnM
        # Compress
        HMF_2d = HMF_2d[::self.compression,::self.compression]
        # Make it safe to take log
        HMF_2d[HMF_2d==0] = np.nextafter(0,1)
        return np.log(HMF_2d)


    def get_P_3obs_DES_z(self, obsnames, z):
        """Return P(DES_WL, obs_1, zeta | M, z, p) with correlated scatter."""
        # Mass function at this redshift
        dN_dlnM, = np.exp(self.HMF_interp(np.log(z), np.log(self.HMF['M_arr'])))
        # (Mass-dependent) covariance matrices
        if obsnames[1]=='richness':
            cov_base = np.array([[1, self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['rhoSZWL']*self.scaling['Dsz']],
                                 [self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['Drichness']**2, self.scaling['rhoSZrichness']*self.scaling['Dsz']],
                                 [self.scaling['rhoSZWL']*self.scaling['Dsz'], self.scaling['rhoSZrichness']*self.scaling['Dsz'], self.scaling['Dsz']**2]])
        DES_scatter = scaling_relations.WLscatter('main', self.HMF['M_arr'], z, self.scaling)
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
        kernels = [None]*len(self.HMF['M_arr'])
        Nbins_obs0 = np.empty((len(self.HMF['M_arr']), 2), dtype=int)
        Nbins_obs1 = np.empty((len(self.HMF['M_arr']), 2), dtype=int)
        Nbins_zeta = np.empty((len(self.HMF['M_arr']), 2), dtype=int)
        for i in range(len(self.HMF['M_arr'])):
            # Number of bins and arrays for each observable
            Nbins_obs0[i], lnobs0_arr = self.get_Nbins_array(msqrt(covmat_lnM[i,0,0]))
            Nbins_obs1[i], lnobs1_arr = self.get_Nbins_array(msqrt(covmat_lnM[i,1,1]))
            Nbins_zeta[i], lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[i,2,2]))
            # Multivariate Gaussian kernel
            kernels[i] = cy_multivariate_normal.trivariate_normal(lnobs0_arr, lnobs1_arr, lnzeta_arr, cov=covmat_lnM[i])
        # Actual convolution
        HMF_3d = convolution.convolve_HMF_3obs_varkernel(dN_dlnM, kernels, Nbins_obs0, Nbins_obs1, Nbins_zeta)
        HMF_3d*= self.Delta_lnM
        # Compress
        HMF_3d = HMF_3d[::self.compression,::self.compression,::self.compression]
        # Make it safe to take log
        HMF_3d[HMF_3d==0] = np.nextafter(0,1)
        return np.log(HMF_3d)


    def get_P_3obs_z(self, obsnames, covmat, z):
        """Return P(obs0, obs1, zeta | M, z(z_id), p) for constant correlated
        scatter."""
        dN_dlnM, = np.exp(self.HMF_interp(np.log(z), np.log(self.HMF['M_arr'])))
        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = [scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames]
        Jacobian = np.array([[dlnM_dlnobs[0]**2,             dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[0]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[1]**2,             dlnM_dlnobs[1]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnzeta,   dlnM_dlnobs[1]*dlnM_dlnzeta,   dlnM_dlnzeta**2]])
        covmat_lnM = covmat * Jacobian
        # Number of bins and observable arrays
        Nbins_obs0, lnobs0_arr = self.get_Nbins_array(msqrt(covmat_lnM[0,0]))
        Nbins_obs1, lnobs1_arr = self.get_Nbins_array(msqrt(covmat_lnM[1,1]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[2,2]))
        # Get the scatter kernel [lnobs, lnzeta]
        kernel = cy_multivariate_normal.trivariate_normal(lnobs0_arr, lnobs1_arr, lnzeta_arr, covmat)
        # Convolution
        HMF_3d = convolution.convolve_HMF_3obs_fixedkernel(dN_dlnM, kernel, Nbins_obs0, Nbins_obs1, Nbins_zeta)
        HMF_3d*= self.Delta_lnM
        # Compress
        HMF_3d = HMF_3d[::self.compression,::self.compression,::self.compression]
        # Make it safe to take log
        HMF_3d[HMF_3d==0] = np.nextafter(0,1)
        return np.log(HMF_3d)
