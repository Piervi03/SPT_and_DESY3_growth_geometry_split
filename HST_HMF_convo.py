import numpy as np
from math import sqrt as msqrt
from multiprocessing import Pool
from scipy.interpolate import RectBivariateSpline
from scipy.stats import multivariate_normal
from astropy.table import Table

import convolution, scaling_relations


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MultiObsConvolution.get_P_multiobs_z_fixedkernel(*arg)

################################################################################
class MultiObsConvolution:

    def __init__(self, HSTcalibfile,
                 observable_pairs, pairs_zmin, pairs_zmax, pairs_Nz):
        self.HSTcalib = Table.read(HSTcalibfile, format='ascii.commented_header')

        self.pairnames_2d = ['HST_SZ',]
        self.pairnames_3d = ['HST_Yx_SZ', 'HST_Mgas_SZ']
        self.obsnames_dict = {'HST_SZ': 'WLHST',
                              'HST_Yx_SZ': ['WLHST', 'Yx'],
                              'HST_Mgas_SZ': ['WLHST', 'Mgas'],
                              }
        # Sigma-clipping in convolutions
        self.N_sigma = 4
        self.compression = 10

        self.observable_pairs = []  # , self.pairs_zmin, self.pairs_zmax, self.pairs_Nz = [], [], [], []
        for pair, zmin, zmax, Nz in zip(observable_pairs, pairs_zmin, pairs_zmax, pairs_Nz):
            if (pair in self.pairnames_2d) | (pair in self.pairnames_3d):
                self.observable_pairs.append(pair)
                #self.pairs_zmin.append(zmin)
                #self.pairs_zmax.append(zmax)
                #self.pairs_Nz.append(Nz)




    ############################################################################
    def execute(self):
        ##### Set up interpolation for HMF
        with np.errstate(divide='ignore'):
            lnHMF_in = np.log(self.HMF['dNdlnM'])
        self.HMF_interp = RectBivariateSpline(self.HMF['z_arr'], np.log(self.HMF['M_arr']), lnHMF_in, kx=1, ky=1)
        self.Delta_lnM = np.log(self.HMF['M_arr'][1]/self.HMF['M_arr'][0])

        ##### Pre-compute the intrinsic scatter convolutions
        output_dict = {}
        for pair_name in self.observable_pairs:
            output_dict[pair_name] = {}
            for n,name in enumerate(self.HSTcalib['SPT_ID']):
                this_grid_ = self.get_P_multiobs_z_fixedkernel(pair_name,
                                                               self.obsnames_dict[pair_name],
                                                               self.covmat['cov_%s_%s'%(pair_name, name)],
                                                               self.HSTcalib['redshift'][n])
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
        dN_dlnM, = np.exp(self.HMF_interp(z, np.log(self.HMF['M_arr'])))

        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs(obsname, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
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

        # Ignore ln(0) warnings
        with np.errstate(divide='ignore'):
            lnHMF_2d = np.log(HMF_2d)

        return lnHMF_2d


    def get_P_3obs_z(self, obsnames, covmat, z):
        """Return P(obs0, obs1, zeta | M, z(z_id), p) for constant correlated
        scatter."""
        dN_dlnM, = np.exp(self.HMF_interp(z, np.log(self.HMF['M_arr'])))

        # Convert observable covmat into covmat in mass
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = [scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames]

        Jacobian = np.array([[dlnM_dlnobs[0]**2,             dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[0]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[1]**2,             dlnM_dlnobs[1]*dlnM_dlnzeta],
                             [dlnM_dlnobs[0]*dlnM_dlnzeta,   dlnM_dlnobs[1]*dlnM_dlnzeta,   dlnM_dlnzeta**2]])
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

        # Ignore ln(0) warnings
        with np.errstate(divide='ignore'):
            lnHMF_3d = np.log(HMF_3d)

        return lnHMF_3d
