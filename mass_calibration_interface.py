import numpy as np
from scipy.interpolate import make_interp_spline
from cosmosis.datablock import option_section
import mass_calibration, lensing


def setup(options):
    ##### Config parameters
    todo = {}
    for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
        todo[opt[2:]] = options.get_bool(option_section, opt)
    mcType = options.get_string(option_section, 'mcType')
    z_cl_min_max = options.get_double_array_1d(option_section, 'z_cl_min_max')
    # lambda_min(z)
    lambda_min_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.genfromtxt(lambda_min_file, names=True, dtype=None)
    lambda_min = {}
    for name in tmp.dtype.names[1:]:
        lambda_min[name] = make_interp_spline(tmp['z'], tmp[name], k=1)
    NPROC = options.get_int(option_section, 'NPROC')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    ##### Multi-obs HMF convolution names
    observable_pairs = options.get_string(option_section, 'observable_pairs').split()
    # WL param file
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')

    masscalibration = mass_calibration.MassCalibration(todo, mcType,
                                                       z_cl_min_max, lambda_min,
                                                       SPT_survey_fields, SPTcatalogfile,
                                                       observable_pairs,
                                                       WLsimcalibfile,
                                                       NPROC)
    masscalibration.YXPARAM = options.get_string(option_section, 'YXPARAM')

    # Set up lensing code
    if todo['WL']:
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        HSTfile = options.get_string(option_section, 'HSTfile')
        MegacamFile = options.get_string(option_section, 'MegacamFile')
        DESfile = options.get_string(option_section, 'DESfile')
        masscalibration.WL = lensing.SPTlensing(masscalibration.catalog,
                                                WLsimcalibfile,
                                                HSTfile, MegacamFile, DESfile,
                                                mcType,
                                                NPROC)

    return masscalibration


def execute(block, masscalibration):
    ##### Extract from datablock
    cosmology = {
        'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
        'h': block.get_double('cosmological_parameters', 'hubble')/100,
        'ns': block.get_double('cosmological_parameters', 'n_s'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa'),
        'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
    for p in ['Omega_m', 'Omega_b', 'wa']:
        cosmology[p] = block.get_double('cosmological_parameters', p)

    scaling = {'YXPARAM': masscalibration.YXPARAM}
    for p in ['Asz', 'Bsz', 'Csz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',
              'Ax', 'Bx', 'Cx', 'Ex', 'dlnMg_dlnr', 'XraymPivot',
              'DES_b_dev_0', 'DES_b_dev_1', 'DES_b_dev_2',
              'DES_b_m', 'DES_m_piv',
              'HSTscatterLSS', 'MegacamScatterLSS',
              'Arichness', 'Brichness', 'Crichness', 'richmPivot',
              'Adisp', 'Bdisp', 'Cdisp',]:
        scaling[p] = block.get_double('mor_parameters', p)
    for p in ['DESwl_z', 'DESwl_bias_mean', 'DESwl_bias_std']:
        scaling[p] = block.get_double_array_1d('mor_parameters', p)

    scaling['bWL_HST'] = {}
    for name in masscalibration.WLcalib['HSTsim'].keys():
        scaling['bWL_HST'][name] = block.get_double('mor_parameters', 'bWL_HST_%s'%name)

    # Get multi-obs HMF convolutions
    HMF_convos = {}
    HMF_convos['lnM_arr'] = block.get_double_array_1d('dN_dmultiobs', 'lnM_arr')
    for pair_name in masscalibration.observable_pairs:
        if pair_name[:3]=='HST':
            HMF_convos[pair_name] = {}
            for name in masscalibration.WLcalib['HSTsim'].keys():
                HMF_convos[pair_name][name] = block.get_double_array_nd('dN_dmultiobs', '%s_%s'%(pair_name, name))
        else:
            HMF_convos[pair_name] = block.get_double_array_nd('dN_dmultiobs', pair_name)
            HMF_convos['%s_z'%pair_name] = block.get_double_array_1d('dN_dmultiobs', '%s_z'%pair_name)

    ##### Compute lensing likelihoods
    if masscalibration.todo['WL']:
        masscalibration.WL.lnlike_all(masscalibration.catalog,
                                      cosmology,
                                      scaling)

    ##### Compute likelihood
    lnlike = masscalibration.lnlike(HMF_convos, cosmology, scaling)
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)

    return 0


def cleanup(config):
    pass
