from __future__ import division
import numpy as np
import imp
from multiprocessing import Pool
from scipy.interpolate import RectBivariateSpline
from scipy.stats import multivariate_normal

import convolution, scaling_relations


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MultiObsConvolution.get_P_multiobs_z_fixedkernel(*arg)

################################################################################
class MultiObsConvolution:

    def __init__(self, WLsimcalibfile,
                 observable_pairs, pairs_zmin, pairs_zmax, pairs_Nz,
                 NPROC):
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

        self.observable_pairs = observable_pairs
        self.pairs_zmin = pairs_zmin
        self.pairs_zmax = pairs_zmax
        self.pairs_Nz = pairs_Nz
        self.NPROC = NPROC

        self.pairnames_2d = ['HST_SZ',]
        self.pairnames_3d = ['HST_Yx_SZ', 'HST_Mgas_SZ']
        self.obsnames_dict = {'HST_SZ': 'WLHST',
                              'HST_Yx_SZ': ['WLHST', 'Yx'],
                              'HST_Mgas_SZ': ['WLHST', 'Mgas'],
                              }
        # Sigma-clipping in convolutions
        self.N_sigma = 4



    ############################################################################
    def execute(self):
        ##### Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in))

        ##### Pre-compute the intrinsic scatter convolutions
        output_dict = {}
        for pair_idx,pair_name in enumerate(self.observable_pairs):
            z_arr = np.linspace(self.pairs_zmin[pair_idx], self.pairs_zmax[pair_idx], self.pairs_Nz[pair_idx])
            this_covmat_ = self.covmat['cov_%s'%pair_name]
            obsname_s_ = self.obsnames_dict[pair_name]
            this_grid_ = self.get_P_2obs_allz(obsname=obsname_s_,
                                              covmat=this_covmat_,
                                              z_arr=z_arr)
            output_dict[pair_name] = this_grid_
            output_dict['%s_z'%pair_name] = z_arr

        return output_dict



    def get_P_multiobs_allz(self, obsname, covmat, z_arr):
        """Return P(obs, xi | M, z, p) for each redshift in z_arr. Optional
        multiprocess."""
        # Write to self to make function pickleable for multiprocessing
        self.obsname = obsname
        self.covmat = covmat

        if self.NPROC==0:
            # Iterate through redshift array
            P_obs_grid = np.array([self.get_P_multiobs_z_fixedkernel(z) for z in z_arr])
        else:
            # Launch and execute a multiprocessing pool
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len(z_arr), z_arr)
            P_obs_grid = pool.map(unwrap_self_f, argin, chunksize=len_z//self.NPROC)
            pool.close()
        return P_obs_grid


    def get_P_multiobs_z_fixedkernel(self, z):
        """Decide whether it's a 2D or 3D observable array."""
        # Unpack self (again, because of multiprocessing)
        covmat = self.covmat
        obsname = self.obsname
        # Compute 2D or 3D multi-obs HMF convolution
        if pairname in self.pairnames_2d:
            return self.get_P_2obs_z_fixedkernel(obsname, covmat, z)
        elif pairname in self.pairnames_3d:
            return self.get_P_3obs_z_fixedkernel(obsname, covmat, z)



    def get_P_2obs_z_fixedkernel(self, obsname, covmat, z):
        """Return P(obs, zeta | M, z(z_id), p) for constant correlated
        scatter."""
        dN_dlnM, = self.HMF_interp(z_arr)

        # Convert observable covmat into covmat in mass
        Delta_lnM = np.log(self.HMF['M_arr'][1]/self.HMF['M_arr'][0])
        dlnzeta_dlnM = 1/scaling_relations.dlnM_dlnobs('zeta')
        dlnobs_dlnM = 1/scaling_relations.dlnM_dlnobs(obsname)
        Jacobian = np.array([[dlnobs_dlnM**2, dlnobs_dlnM*dlnzeta_dlnM],
                             [dlnobs_dlnM*dlnzeta_dlnM, dlnzeta_dlnM**2]])
        covmat_lnM = covmat * Jacobian

        Nbins_obs = int(2 * self.N_sigma * covmat_lnM[0,0]**.5 / Delta_lnM)
        if Nbins_obs%2 != 0:
            Nbins_obs+= 1
        minmax_ = (Nbins_obs-1)/2 * Delta_lnM
        lnobs_arr = np.linspace(-minmax_, minmax_, Nbins_obs)

        Nbins_zeta = int(2 * self.N_sigma * covmat_lnM[1,1]**.5 / Delta_lnM)
        if Nbins_zeta%2 != 0:
            Nbins_zeta+= 1
        minmax_ = (Nbins_zeta-1)/2 * Delta_lnM
        lnzeta_arr = np.linspace(-minmax_, minmax_, Nbins_zeta)

        # Get the scatter kernel [lnobs, lnzeta]
        pos = np.empty((len_obs, len_zeta, 2))
        pos[:,:,0], pos[:,:,1] = np.meshgrid(lnobs_arr, lnzeta_arr, indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(0,0), cov=covmat_lnM)

        HMF_2d = self.convolve_HMF_2obs_fixedkernel(dN_dlnM, kernel)

        return HMF_2d


    def get_P_3obs_z_fixedkernel(self, obsnames, covmat, z):
        """Return P(obs0, obs1, zeta | M, z(z_id), p) for constant correlated
        scatter."""
        dN_dlnM, = self.HMF_interp(z_arr)

        # Convert observable covmat into covmat in mass
        Delta_lnM = np.log(self.HMF['M_arr'][1]/self.HMF['M_arr'][0])
        dlnzeta_dlnM = 1/scaling_relations.dlnM_dlnobs('zeta')
        dlnobs_dlnM = [1/scaling_relations.dlnM_dlnobs(obs) for obs in obsnames]

        Jacobian = np.array([[dlnobs_dlnM[0]**2,             dlnobs_dlnM[0]*dlnobs_dlnM[1], dlnobs_dlnM[0]*dlnzeta_dlnM,
                             [dlnobs_dlnM[0]*dlnobs_dlnM[1], dlnobs_dlnM[1]**2,             dlnobs_dlnM[1]*dlnzeta_dlnM],
                             [dlnobs_dlnM[0]*dlnzeta_dlnM,   dlnobs_dlnM[1]*dlnzeta_dlnM,   dlnzeta_dlnM**2]])
        covmat_lnM = covmat * Jacobian

        Nbins_obs = [int(2 * self.N_sigma * covmat_lnM[i,i]**.5 / Delta_lnM) for i in range(2)]
        for i in range(2):
            if Nbins_obs[i]%2 != 0:
                Nbins_obs[i]+= 1
        minmax_ = [(Nbins_obs[i]-1)/2 * Delta_lnM for i in range(2)]
        lnobs_arr = [np.linspace(-minmax_[i], minmax_[i], Nbins_obs[i]) for i in range(2)]

        Nbins_zeta = int(2 * self.N_sigma * covmat_lnM[2,2]**.5 / Delta_lnM)
        if Nbins_zeta%2 != 0:
            Nbins_zeta+= 1
        minmax_ = (Nbins_zeta-1)/2 * Delta_lnM
        lnzeta_arr = np.linspace(-minmax_, minmax_, Nbins_zeta)

        # Get the scatter kernel [lnobs, lnzeta]
        pos = np.empty((len_obs[0], len_obs[1], len_zeta, 3))
        pos[:,:,0], pos[:,:,1], pos[:,:,2] = np.meshgrid(lnobs_arr[0], lnobs_arr[1], lnzeta_arr, indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(0,0,0), cov=covmat)

        HMF_3d = self.convolve_HMF_3obs_fixedkernel(dN_dlnM, kernel)

        return HMF_3d
