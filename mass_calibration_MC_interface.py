from __future__ import division
import numpy as np
from astropy.table import Table
from cosmosis.datablock import option_section
import mass_calibration_MC as mass_calibration
import lensing

def setup(options):
    ##### Config parameters
    todo = {}
    for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
        todo[opt[2:]] = options.get_bool(option_section, opt)
    mcType = options.get_string(option_section, 'mcType')
    surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
    surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
    NPROC = options.get_int(option_section, 'NPROC')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    # Double counted clusters
    SPT_doublecounts = options.get_string(option_section, 'SPT_doublecounts')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    # WL param file
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')

    masscalibration = mass_calibration.MassCalibration(todo, mcType,
                                                       surveyCutSZ, surveyCutRedshift,
                                                       SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
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
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Bsz2', 'Csz2', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',
              'Ax', 'Bx', 'Cx', 'Ex', 'dlnMg_dlnr', 'XraymPivot',
              'DES_b_dev', 'DES_b_m', 'DES_s_dev', 'DES_s_M', 'DES_m_piv',
              'HSTscatterLSS', 'MegacamScatterLSS', 
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'rhoSZrichness', 'rhoWLrichness', 'rhoSZWL',
              'Adisp', 'Bdisp', 'Cdisp',]:
        scaling[p] = block.get_double('mor_parameters', p)
    for p in ['DESwl_z', 'DESwl_bias_mean', 'DESwl_bias_std']:
        scaling[p] = block.get_double_array_1d('mor_parameters', p)

    scaling['bWL_HST'] = {}
    for name in masscalibration.WLcalib['HSTsim'].keys():
        scaling['bWL_HST'][name] = block.get_double('mor_parameters', 'bWL_HST_%s'%name)
    for p in ['DESwl_z', 'DESwl_scatter_mean', 'DESwl_scatter_std']:
        scaling[p] = block.get_double_array_1d('mor_parameters', p)

    # Halo mass function
    HMF = {
        'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
          'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
          'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    HMF['len_z'] = len(HMF['z_arr'])

    ##### Compute lensing likelihoods
    if masscalibration.todo['WL']:
        masscalibration.WL.lnlike_all(masscalibration.catalog,
                                      cosmology,
                                      scaling)

    ##### Compute likelihood
    lnlike = masscalibration.lnlike(HMF, cosmology, scaling)
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)

    return 0

def cleanup(config):
    pass
