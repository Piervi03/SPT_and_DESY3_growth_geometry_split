from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
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

    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.loadtxt(surveyCutLambda_file, unpack=True)
    surveyCutLambda = {'shallow': interp1d(tmp[0], tmp[1], kind='linear'),
                       'deep': interp1d(tmp[0], tmp[2], kind='linear')}
    richness_scatter_model = options.get_string(option_section, 'richness_scatter_model')

    multi_obs_convolution = HMF_convo.MultiObsConvolution(observable_pairs,
                                                          pairs_zmin, pairs_zmax, pairs_Nz,
                                                          surveyCutLambda, richness_scatter_model,
                                                          NPROC)

    return multi_obs_convolution

def execute(block, multi_obs_convolution):
    ##### Extract from datablock
    scaling = {}
    for p in ['Bsz', 'Bx', 'Dsz',
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'DES_b_dev_0', 'DES_b_dev_1', 'DES_b_dev_2',
              'DES_s_dev_0', 'DES_s_dev_1', 'DES_s_dev_2',
              'DES_b_m', 'DES_s_M', 'DES_m_piv',
              'rhoSZWL', 'rhoSZrichness', 'rhoWLrichness']:
        scaling[p] = block.get_double('mor_parameters', p)
    for p in ['DESwl_z', 'DESwl_scatter_mean', 'DESwl_scatter_std']:
        scaling[p] = block.get_double_array_1d('mor_parameters', p)
    # Covariance matrices
    covmat = {}
    for c in ['cov_X_SZ', 'cov_Megacam_SZ', 'cov_richness_SZ', 'cov_Megacam_X_SZ', ]:
        covmat[c] = block.get_double_array_nd('mor_parameters', c)
    # Halo mass function
    HMF = {
        'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
        'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
        'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    HMF['len_z'] = len(HMF['z_arr'])

    ##### Compute the convolutions
    dN_dmultiobs_dict = multi_obs_convolution.execute(HMF, scaling, covmat)
    block.put_double_array_1d('dN_dmultiobs', 'M_arr', dN_dmultiobs_dict['M_arr'])
    for pair_name in multi_obs_convolution.observable_pairs:
        block.put_double_array_nd('dN_dmultiobs', pair_name, dN_dmultiobs_dict[pair_name])
        block.put_double_array_1d('dN_dmultiobs', '%s_z'%pair_name, dN_dmultiobs_dict['%s_z'%pair_name])

    return 0

def cleanup(config):
    pass
