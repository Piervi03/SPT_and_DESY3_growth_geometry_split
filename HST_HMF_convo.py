from __future__ import division
import numpy as np
from math import sqrt as msqrt
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
                 observable_pairs, pairs_zmin, pairs_zmax, pairs_Nz):
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

        self.pairnames_2d = ['HST_SZ',]
        self.pairnames_3d = ['HST_Yx_SZ', 'HST_Mgas_SZ']
        self.obsnames_dict = {'HST_SZ': 'WLHST',
                              'HST_Yx_SZ': ['WLHST', 'Yx'],
                              'HST_Mgas_SZ': ['WLHST', 'Mgas'],
                              }
        # Sigma-clipping in convolutions
        self.N_sigma = 4
        self.compression = 10

        self.observable_pairs = []#, self.pairs_zmin, self.pairs_zmax, self.pairs_Nz = [], [], [], []
        for pair, zmin, zmax, Nz in zip(observable_pairs, pairs_zmin, pairs_zmax, pairs_Nz):
            if (pair in self.pairnames_2d) | (pair in self.pairnames_3d):
                self.observable_pairs.append(pair)
                #self.pairs_zmin.append(zmin)
                #self.pairs_zmax.append(zmax)
                #self.pairs_Nz.append(Nz)




    ############################################################################
    def execute(self):
        ##### Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in))
        self.Delta_lnM = np.log(self.HMF['M_arr'][1]/self.HMF['M_arr'][0])

        ##### Pre-compute the intrinsic scatter convolutions
        output_dict = {}
        for pair_name in self.observable_pairs:
            output_dict[pair_name] = {}
            for name in self.WLcalib['HSTsim'].keys():
                this_grid_ = self.get_P_multiobs_z_fixedkernel(pair_name,
                                                               self.obsnames_dict[pair_name],
                                                               self.covmat['cov_%s_%s'%(pair_name, name)],
                                                               self.WLcalib['HSTsim'][name]['z'])
                output_dict[pair_name][name] = this_grid_

        return output_dict



    def get_P_multiobs_z_fixedkernel(self, pair_name, obs_name, covmat, z):
        """Decide whether it's a 2D or 3D observable array."""
        # Compute 2D or 3D multi-obs HMF convolution
        if pair_name in self.pairnames_2d:
            return self.get_P_2obs_z(obs_name, covmat, z)
        elif pair_name in self.pairnames_3d:
            return self.get_P_3obs_z(obs_name, covmat, z)



    def get_Nbins_array(self, std):
        Nbins_obs = int(2 * self.N_sigma * std / self.Delta_lnM)
        if Nbins_obs%2 != 0:
            Nbins_obs+= 1
        minmax_ = (Nbins_obs-1)/2 * self.Delta_lnM
        lnobs_arr = np.linspace(-minmax_, minmax_, Nbins_obs)

        return Nbins_obs, lnobs_arr


    def get_P_2obs_z(self, obsname, covmat, z): 
        """Return P(obs, zeta | M, z(z_id), p) for constant correlated
        scatter."""
        dN_dlnM, = np.exp(self.HMF_interp(np.log(z), np.log(self.HMF['M_arr'])))

        # Convert observable covmat into covmat in mass
        dlnzeta_dlnM = 1/scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnobs_dlnM = 1/scaling_relations.dlnM_dlnobs(obsname, self.scaling)
        Jacobian = np.array([[dlnobs_dlnM**2, dlnobs_dlnM*dlnzeta_dlnM],
                             [dlnobs_dlnM*dlnzeta_dlnM, dlnzeta_dlnM**2]])
        covmat_lnM = covmat * Jacobian

        Nbins_obs, lnobs_arr = self.get_Nbins_array(msqrt(covmat_lnM[0,0]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[1,1]))


        # Get the scatter kernel [lnobs, lnzeta]
        pos = np.empty((Nbins_obs, Nbins_zeta, 2))
        pos[:,:,0], pos[:,:,1] = np.meshgrid(lnobs_arr, lnzeta_arr, indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(0,0), cov=covmat_lnM)

        HMF_2d = convolution.convolve_HMF_2obs_fixedkernel(dN_dlnM, kernel)
        HMF_2d*= self.Delta_lnM

        # Compress
        HMF_2d = HMF_2d[::self.compression,::self.compression]

        # Remove 0
        HMF_2d[(HMF_2d==0).nonzero()] = np.nextafter(0, 1)

        return np.log(HMF_2d)


    def get_P_3obs_z(self, obsnames, covmat, z): 
        """Return P(obs0, obs1, zeta | M, z(z_id), p) for constant correlated
        scatter."""
        dN_dlnM, = np.exp(self.HMF_interp(np.log(z), np.log(self.HMF['M_arr'])))

        # Convert observable covmat into covmat in mass
        dlnzeta_dlnM = 1/scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnobs_dlnM = [1/scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames]

        Jacobian = np.array([[dlnobs_dlnM[0]**2,             dlnobs_dlnM[0]*dlnobs_dlnM[1], dlnobs_dlnM[0]*dlnzeta_dlnM],
                             [dlnobs_dlnM[0]*dlnobs_dlnM[1], dlnobs_dlnM[1]**2,             dlnobs_dlnM[1]*dlnzeta_dlnM],
                             [dlnobs_dlnM[0]*dlnzeta_dlnM,   dlnobs_dlnM[1]*dlnzeta_dlnM,   dlnzeta_dlnM**2]])
        covmat_lnM = covmat * Jacobian


        Nbins_obs0, lnobs0_arr = self.get_Nbins_array(msqrt(covmat_lnM[0,0]))
        Nbins_obs1, lnobs1_arr = self.get_Nbins_array(msqrt(covmat_lnM[1,1]))
        Nbins_zeta, lnzeta_arr = self.get_Nbins_array(msqrt(covmat_lnM[2,2]))


        # Get the scatter kernel [lnobs, lnzeta]
        pos = np.empty((Nbins_obs0, Nbins_obs1, Nbins_zeta, 3)) 
        pos[:,:,:,0], pos[:,:,:,1], pos[:,:,:,2] = np.meshgrid(lnobs0_arr, lnobs1_arr, lnzeta_arr, indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(0,0,0), cov=covmat)

        HMF_3d = convolution.convolve_HMF_3obs_fixedkernel(dN_dlnM, kernel)
        HMF_3d*= self.Delta_lnM

        # Compress
        HMF_3d = HMF_3d[::self.compression,::self.compression,::self.compression]

        # Remove 0
        HMF_3d[(HMF_3d==0).nonzero()] = np.nextafter(0, 1)

        return np.log(HMF_3d)
     
