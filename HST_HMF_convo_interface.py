from __future__ import division
import imp
from cosmosis.datablock import option_section
import HST_HMF_convo

def setup(options):
    multi_obs_convolution = HST_HMF_convo.MultiObsConvolution(options)

    # WL simulation calibration data
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
    WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
    multi_obs_convolution.WLcalib = WLsimcalib.WLcalibration

    multi_obs_convolution.observable_pairs = options.get_string(option_section, 'observable_pairs').split()
    for pair in multi_obs_convolution.observable_pairs:
        assert pair in multi_obs_convolution.pairnames_2d or in multi_obs_convolution.pairnames_3d, "Unknown pair of observables %s"%pair
    multi_obs_convolution.pairs_zmin = options.get_double_array_1d(option_section, 'pairs_zmin')
    multi_obs_convolution.pairs_zmax = options.get_double_array_1d(option_section, 'pairs_zmax')
    multi_obs_convolution.pairs_Nz = options.get_double_array_1d(option_section, 'pairs_Nz')
    assert len(multi_obs_convolution.pairs_zmin)==len(multi_obs_convolution.observable_pairs), "Bad length of pairs_zmin"
    assert len(multi_obs_convolution.pairs_zmax)==len(multi_obs_convolution.observable_pairs), "Bad length of pairs_zmax"
    assert len(multi_obs_convolution.pairs_Nz)==len(multi_obs_convolution.observable_pairs), "Bad length of pairs_Nz"

    # Number of multi-processes
    multi_obs_convolution.NPROC = options.get_int(option_section, 'NPROC')

    return multi_obs_convolution

def execute(block, multi_obs_convolution):
    ##### Extract from datablock
    # Scaling relation parameters
    multi_obs_convolution.scaling = {}
    for p in ['Bsz',]:
        multi_obs_convolution.scaling[p] = block.get_double('mor_parameters', p)
    # Covariance matrices
    multi_obs_convolution.covmat = {}
    for name in multi_obs_convolution.WLcalib['HSTsim'].keys():
        cov_name = 'cov_HST_SZ_%s'%name
        multi_obs_convolution.covmat[cov_name] = block.get_double_array_nd('scaling', covname)
        cov_name = 'cov_HST_X_SZ_%s'%name
        multi_obs_convolution.covmat[cov_name] = block.get_double_array_nd('scaling', covname)
    # Halo mass function
    multi_obs_convolution.HMF = {
        'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
        'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
        'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    multi_obs_convolution.HMF['len_z'] = len(multi_obs_convolution.HMF['z_arr'])

    ##### Execute
    HST_convo_dict = multi_obs_convolution.execute()

    ##### Put back into block
    for pair_name in multi_obs_convolution.observable_pairs:
        block.put_double_array_nd('dN_dmultiobs', pair_name, HST_convo_dict[pair_name])
        block.put_double_array_nd('dN_dmultiobs', '%s_z'%pair_name, HST_convo_dict['%s_z'%pair_name])

    return 0

def cleanup(config):
    pass
