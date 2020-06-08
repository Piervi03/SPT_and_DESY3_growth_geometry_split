from __future__ import division
import numpy as np
from astropy.table import Table
from cosmosis.datablock import option_section
import mass_calibration, lensing

def setup(options):
    ##### Config parameters
    todo = {}
    for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
        todo[opt[2:]] = options.get_bool(option_section, opt)
    scaling = {}
    for opt in ['SZmPivot', 'XraymPivot', 'richmPivot']:
        scaling[opt] = options.get_double(option_section, opt)
    scaling['YXPARAM'] = options.get_string(option_section, 'YXPARAM')
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
    ##### Multi-obs HMF convolution names
    observable_pairs = options.get_string(option_section, 'observable_pairs').split()
    # WL param file
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')

    masscalibration = mass_calibration.MassCalibration(todo, scaling, mcType,
                                                       surveyCutSZ, surveyCutRedshift,
                                                       SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
                                                       observable_pairs,
                                                       WLsimcalibfile,
                                                       NPROC)

    # Set up lensing code
    if todo['WL']:
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        HSTfile = options.get_string(option_section, 'HSTfile')
        MegacamFile = options.get_string(option_section, 'MegacamFile')
        DESfile = options.get_string(option_section, 'DESfile')
        masscalibration.WL = lensing.SPTlensing(masscalibration.catalog,
                                                WLsimcalibfile,
                                                HSTfile, MegacamFile, DESfile,
                                                mcType)

    return masscalibration


def execute(block, masscalibration):
    ##### Extract from datablock
    masscalibration.cosmology = {
        'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
        'h': block.get_double('cosmological_parameters', 'hubble')/100,
        'ns': block.get_double('cosmological_parameters', 'n_s'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa'),
        'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
    for p in ['Omega_m', 'Omega_b', 'wa']:
        masscalibration.cosmology[p] = block.get_double('cosmological_parameters', p)
    # SZ
    for p in ['Asz', 'Bsz', 'Csz', 'Bsz2', 'Csz2', 'Esz', 'SPECS_calib']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # X-ray
    for p in ['Ax', 'Bx', 'Cx', 'Ex', 'dlnMg_dlnr']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # WL
    for p in ['HSTscatterLSS', 'MegacamScatterLSS', 'DESscatterLSS']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    masscalibration.scaling['bWL_HST'] = {}
    for name in masscalibration.WLcalib['HSTsim'].keys():
        masscalibration.scaling['bWL_HST'][name] = block.get_double('mor_parameters', 'bWL_HST_%s'%name)
    # Richness
    for p in ['Arichness', 'Brichness', 'Crichness']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # dispersion
    for p in ['Adisp', 'Bdisp', 'Cdisp']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # Get multi-obs HMF convolutions
    masscalibration.HMF_convos = {}
    masscalibration.HMF_convos['M_arr'] = block.get_double_array_1d('dN_dmultiobs', 'M_arr')
    for pair_name in masscalibration.observable_pairs:
        if pair_name[:3]=='HST':
            masscalibration.HMF_convos[pair_name] = {}
            for name in masscalibration.WLcalib['HSTsim'].keys():
                masscalibration.HMF_convos[pair_name][name] = block.get_double_array_nd('dN_dmultiobs', '%s_%s'%(pair_name, name))
        else:
            masscalibration.HMF_convos[pair_name] = block.get_double_array_nd('dN_dmultiobs', pair_name)
            masscalibration.HMF_convos['%s_z'%pair_name] = block.get_double_array_1d('dN_dmultiobs', '%s_z'%pair_name)

    ##### Compute lensing likelihoods
    if masscalibration.todo['WL']:
        masscalibration.WL.like_all(masscalibration.catalog,
                                    masscalibration.cosmology)

    ##### Compute likelihood
    lnlike = masscalibration.lnlike()
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)

    return 0

def cleanup(config):
    pass
