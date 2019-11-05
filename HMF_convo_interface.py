from __future__ import division
from cosmosis.datablock import option_section
import HMF_convo

def setup(options):
    observable_pairs = options.get_string(option_section, 'observable_pairs').split()
    if len(observable_pairs)==1:
        pairs_zmin = [options.get_double(option_section, 'pairs_zmin')]
        pairs_zmax = [options.get_double(option_section, 'pairs_zmax')]
        pairs_Nz = [options.get_int(option_section, 'pairs_Nz')]
    else:
        pairs_zmin = options.get_double_array_1d(option_section, 'pairs_zmin')
        pairs_zmax = options.get_double_array_1d(option_section, 'pairs_zmax')
        pairs_Nz = options.get_int_array_1d(option_section, 'pairs_Nz')
        assert len(pairs_zmin)==len(observable_pairs), "Bad length of pairs_zmin"
        assert len(pairs_zmax)==len(observable_pairs), "Bad length of pairs_zmax"
        assert len(pairs_Nz)==len(observable_pairs), "Bad length of pairs_Nz"
    NPROC = options.get_int(option_section, 'NPROC')

    multi_obs_convolution = HMF_convo.MultiObsConvolution(observable_pairs,
                                                          pairs_zmin, pairs_zmax, pairs_Nz,
                                                          NPROC)

    return multi_obs_convolution

def execute(block, multi_obs_convolution):
    ##### Extract from datablock
    for p in ['Bsz', 'Bx', 'Brichness']:
        multi_obs_convolution.scaling[p] = block.get_double('mor_parameters', p)
    # Covariance matrices
    for c in ['cov_X_SZ', 'cov_Megacam_SZ', 'cov_DES_SZ', 'cov_richness_SZ', 'cov_Megacam_X_SZ', 'cov_DES_X_SZ', 'cov_DES_richness_SZ']:
        multi_obs_convolution.covmat[c] = block.get_double_array_nd('mor_parameters', c)
    # Halo mass function
    multi_obs_convolution.HMF = {
        'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
        'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
        'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    multi_obs_convolution.HMF['len_z'] = len(multi_obs_convolution.HMF['z_arr'])

    ##### Compute the convolutions
    dN_dmultiobs_dict = multi_obs_convolution.execute()
    block.put_double_array_1d('dN_dmultiobs', 'M_arr', dN_dmultiobs_dict['M_arr'])
    for pair_name in multi_obs_convolution.observable_pairs:
        block.put_double_array_nd('dN_dmultiobs', pair_name, dN_dmultiobs_dict[pair_name])
        block.put_double_array_1d('dN_dmultiobs', '%s_z'%pair_name, dN_dmultiobs_dict['%s_z'%pair_name])

    return 0

def cleanup(config):
    pass
