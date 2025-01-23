from cosmosis.datablock import option_section
import HST_HMF_convo


def setup(options):
    # WL simulation calibration data
    HSTcalibfile = options.get_string(option_section, 'HSTcalibfile')

    observable_pairs = options.get_string(option_section, 'observable_pairs').split()
    pairs_zmin = options.get_double_array_1d(option_section, 'pairs_zmin')
    pairs_zmax = options.get_double_array_1d(option_section, 'pairs_zmax')
    pairs_Nz = options.get_int_array_1d(option_section, 'pairs_Nz')
    assert len(pairs_zmin)==len(observable_pairs), "Bad length of pairs_zmin"
    assert len(pairs_zmax)==len(observable_pairs), "Bad length of pairs_zmax"
    assert len(pairs_Nz)==len(observable_pairs), "Bad length of pairs_Nz"

    multi_obs_convolution = HST_HMF_convo.MultiObsConvolution(HSTcalibfile,
                                                              observable_pairs,
                                                              pairs_zmin, pairs_zmax, pairs_Nz)

    return multi_obs_convolution


def execute(block, multi_obs_convolution):
    ##### Extract from datablock
    # Scaling relation parameters
    multi_obs_convolution.scaling = {}
    for p in ['Bsz',]:
        multi_obs_convolution.scaling[p] = block.get_double('mor_parameters', p)
    # Covariance matrices
    multi_obs_convolution.covmat = {}
    for name in multi_obs_convolution.HSTcalib['SPT_ID']:
        cov_name = 'cov_HST_SZ_%s'%name
        multi_obs_convolution.covmat[cov_name] = block.get_double_array_nd('mor_parameters', cov_name)
        cov_name = 'cov_HST_X_SZ_%s'%name
        multi_obs_convolution.covmat[cov_name] = block.get_double_array_nd('mor_parameters', cov_name)
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
        for name in multi_obs_convolution.HSTcalib['SPT_ID']:
            block.put_double_array_nd('dN_dmultiobs', '%s_%s'%(pair_name, name), HST_convo_dict[pair_name][name])

    return 0


def cleanup(config):
    pass
