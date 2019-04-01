from __future__ import division
import numpy as np
import imp
from astropy.table import Table
from cosmosis.datablock import option_section
import mass_calibration

def setup(options):
    masscalibration = mass_calibration.MassCalibration()

    ##### Config parameters
    masscalibration.todo = {}
    for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
        masscalibration.todo[opt[2:]] = options.get_bool(option_section, opt)
    masscalibration.scaling = {}
    for opt in ['SZmPivot', 'XraymPivot', 'richmPivot']:
        masscalibration.scaling[opt] = options.get_double(option_section, opt)
    masscalibration.scaling['YXPARAM'] = options.get_string(option_section, 'YXPARAM')
    masscalibration.mcType = options.get_string(option_section, 'mcType')
    masscalibration.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
    masscalibration.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
    masscalibration.NPROC = options.get_int(option_section, 'NPROC')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    masscalibration.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
    # Double counted clusters
    SPT_doublecounts = options.get_string(option_section, 'SPT_doublecounts')
    SPTdata = imp.load_source('SPTdata', SPT_doublecounts)
    masscalibration.SPTdoubleCount = SPTdata.SPTdoubleCount
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    masscalibration.catalog = Table.read(SPTcatalogfile)
    ##### WL simulation calibration
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
    WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
    masscalibration.WLcalib = WLsimcalib.WLcalibration
    ##### Multi-obs HMF convolution names
    masscalibration.observable_pairs = options.get_string(option_section, 'observable_pairs').split()

    if masscalibration.todo['WL']:
        # WL simulation calibration data
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        DES_betabias_file = options.get_string(option_section, 'DES_betabias_file')
        # Lensing data
        HSTfile = options.get_string(option_section, 'HSTfile')
        MegacamFile = options.get_string(option_section, 'MegacamFile')
        DESfile = options.get_string(option_section, 'DESfile')

        masscalibration.init_WL(WLsimcalibfile, HSTfile, MegacamFile, DESfile, DES_betabias_file)

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
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Bsz2', 'Csz2', 'Esz', 'DszM']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # X-ray
    for p in ['Ax', 'Bx', 'Cx', 'Dx', 'Ex', 'dlnMg_dlnr']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # WL
    for p in ['bWL_Megacam', 'bWL_DES', 'DESbias', 'HSTscatterLSS', 'MegacamScatterLSS', 'DESscatterLSS']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # Richness
    for p in ['Arichness', 'Brichness', 'Crichness', 'Drichness']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # dispersion
    for p in ['Adisp', 'Bdisp', 'Cdisp', 'Ddisp0', 'DdispN']:
        masscalibration.scaling[p] = block.get_double('mor_parameters', p)
    # Get multi-obs HMF convolutions
    masscalibration.HMF_convos = {}
    for pair_name in masscalibration.observable_pairs:
        masscalibration.HMF_convos[pair_name] = block.get_double_array_nd('dN_dmultiobs', pair_name)
        masscalibration.HMF_convos['%s_z'%pair_name] = block.get_double_array_nd('dN_dmultiobs', '%s_z'%pair_name)
    # Halo mass function
    masscalibration.HMF = {
        'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
        'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
        'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    masscalibration.HMF['len_z'] = len(masscalibration.HMF['z_arr'])

    ##### Compute likelihood
    lnlike = masscalibration.lnlike()
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)

    return 0

def cleanup(config):
    pass
