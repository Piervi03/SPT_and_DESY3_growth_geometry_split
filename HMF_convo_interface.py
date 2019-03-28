from __future__ import division
from cosmosis.datablock import option_section
import HMF_convo

def setup(options):
    multi_obs_convolution = HMF_convo.MultiObsConvolution()

    multi_obs_convolution.observable_pairs = options.get_string(option_section, 'observable_pairs').split()
    for pair in multi_obs_convolution.observable_pairs:
        assert (pair in multi_obs_convolution.pairnames_2d) or (pair in multi_obs_convolution.pairnames_3d), "Unknown pair of observables %s"%pair
    if len(multi_obs_convolution.observable_pairs)==1:
        multi_obs_convolution.pairs_zmin = [options.get_double(option_section, 'pairs_zmin')]
        multi_obs_convolution.pairs_zmax = [options.get_double(option_section, 'pairs_zmax')]
        multi_obs_convolution.pairs_Nz = [options.get_int(option_section, 'pairs_Nz')]
    else:
        multi_obs_convolution.pairs_zmin = options.get_double_array_1d(option_section, 'pairs_zmin')
        multi_obs_convolution.pairs_zmax = options.get_double_array_1d(option_section, 'pairs_zmax')
        multi_obs_convolution.pairs_Nz = options.get_int_array_1d(option_section, 'pairs_Nz')
        assert len(multi_obs_convolution.pairs_zmin)==len(multi_obs_convolution.observable_pairs), "Bad length of pairs_zmin"
        assert len(multi_obs_convolution.pairs_zmax)==len(multi_obs_convolution.observable_pairs), "Bad length of pairs_zmax"
        assert len(multi_obs_convolution.pairs_Nz)==len(multi_obs_convolution.observable_pairs), "Bad length of pairs_Nz"

    # Number of multi-processes
    multi_obs_convolution.NPROC = options.get_int(option_section, 'NPROC')

    return multi_obs_convolution

def execute(block, multi_obs_convolution):
    ##### Extract from datablock
    for p in ['Bsz', 'Bx', 'Brichness']:
        multi_obs_convolution.scaling[p] = block.get_double('mor_parameters', p)
    # Covariance matrices
    for c in ['cov_X_SZ', 'cov_Megacam_SZ', 'cov_DES_SZ', 'cov_richness_SZ', 'cov_Megacam_X_SZ', 'cov_DES_X_SZ']:
        multi_obs_convolution.covmat[c] = block.get_double_array_nd('mor_parameters', c)
    # Halo mass function
    multi_obs_convolution.HMF = {
        'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
        'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
        'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    multi_obs_convolution.HMF['len_z'] = len(multi_obs_convolution.HMF['z_arr'])

    ##### Compute the convolutions
    dN_dmultiobs_dict = multi_obs_convolution.execute()
    for pair_name in multi_obs_convolution.observable_pairs:
        block.put_double_array_nd('dN_dmultiobs', pair_name, dN_dmultiobs_dict[pair_name])
        block.put_double_array_nd('dN_dmultiobs', '%s_z'%pair_name, dN_dmultiobs_dict['%s_z'%pair_name])

    return 0

def cleanup(config):
    pass
