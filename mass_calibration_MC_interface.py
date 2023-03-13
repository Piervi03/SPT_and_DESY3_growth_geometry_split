from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
import h5py
from cosmosis.datablock import option_section
import mass_calibration_MC as mass_calibration
import lensing


def setup(options):
    ##### Config parameters
    todo = {}
    for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
        todo[opt[2:]] = options.get_bool(option_section, opt)
    todo['lambda_min'] = options.get_bool(option_section, 'lambda_min')
    mcType = options.get_string(option_section, 'mcType')
    surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
    # Data for optical cleaning
    if todo['lambda_min']:
        surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
        tmp = np.loadtxt(surveyCutLambda_file, unpack=True)
        surveyCutLambda = {'shallow': interp1d(tmp[0], tmp[1], kind='linear'),
                           'deep': interp1d(tmp[0], tmp[2], kind='linear')}
    else:
        surveyCutLambda = None
    richness_scatter_model = options.get_string(option_section, 'richness_scatter_model')
    NPROC = options.get_int(option_section, 'NPROC')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    # WL param file
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
    # HST file
    HSTcalibfile = options.get_string(option_section, 'HSTcalibfile')
    # Do stack lensing for validation
    get_stacked_DES = options.get_bool(option_section, 'get_stacked_DES', False)

    masscalibration = mass_calibration.MassCalibration(todo, mcType,
                                                       surveyCutRedshift, surveyCutLambda, richness_scatter_model,
                                                       SPT_survey_fields, SPTcatalogfile,
                                                       HSTcalibfile,
                                                       NPROC, get_stacked_DES=get_stacked_DES)
    masscalibration.YXPARAM = options.get_string(option_section, 'YXPARAM')

    # Set up lensing code
    if todo['WL']:
        # Lensing data
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        HSTfile = options.get_string(option_section, 'HSTfile', default='None')
        MegacamFile = options.get_string(option_section, 'MegacamFile', default='None')
        DESfile = options.get_string(option_section, 'DESfile', default='None')
        DESboostfile = options.get_string(option_section, 'DESboostfile')
        DESboost_z_arr = options.get_double_array_1d(option_section, 'DESboost_z_arr')
        DESmiscenterfile = options.get_string(option_section, 'DESmiscenterfile')
        DEScentertype = options.get_string(option_section, 'DEScentertype')
        masscalibration.WL = lensing.SPTlensing(masscalibration.catalog,
                                                WLsimcalibfile,
                                                HSTfile, MegacamFile, DESfile,
                                                DESboostfile, DESboost_z_arr,
                                                DESmiscenterfile, DEScentertype,
                                                mcType,
                                                NPROC, save_shear_profiles=get_stacked_DES)
        # DES lensing priors
        if DESfile=='None':
            DES_WL_prior = None
        else:
            DES_WL_priors_file = options.get_string(option_section, 'DES_WL_priors_file')
            with h5py.File(DES_WL_priors_file, 'r') as f:
                DES_WL_prior = {}
                for k in f.keys():
                    DES_WL_prior[k] = f[k][()]
            # Additional hydro uncertainty
            DES_WL_prior['deltab_pc1'] = -np.sqrt(DES_WL_prior['deltab_pc1']**2 + .02**2)
            DES_WL_prior['bias_slope'][1] = np.sqrt(DES_WL_prior['bias_slope'][1]**2 + .018**2)
            DES_WL_prior['delta_lnsigma2'] = np.sqrt(DES_WL_prior['delta_lnsigma2']**2 + .25**2)
            DES_WL_prior['lnsigma2_slope'][1] = np.sqrt(DES_WL_prior['lnsigma2_slope'][1] + .59**2)

    return masscalibration, DES_WL_prior


def execute(block, setup_stuff):
    masscalibration, DES_WL_prior = setup_stuff
    ##### Extract from datablock
    cosmology = {
        'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
        'h': block.get_double('cosmological_parameters', 'hubble')/100,
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa'),
        'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
    for p in ['Omega_m', 'Omega_b', 'n_s', 'wa']:
        cosmology[p] = block.get_double('cosmological_parameters', p)

    scaling = {'YXPARAM': masscalibration.YXPARAM}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Bsz2', 'Csz2', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',
              'Ax', 'Bx', 'Cx', 'Ex', 'dlnMg_dlnr', 'XraymPivot',
              'HSTscatterLSS', 'MegacamScatterLSS',
              'bWL_Megacam', 'DWL_Megacam',
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'rhoSZrichness', 'rhoWLrichness', 'rhoSZWL',
              'Adisp', 'Bdisp', 'Cdisp',]:
        scaling[p] = block.get_double('mor_parameters', p)
    # DES
    if DES_WL_prior is not None:
        for p in ['DES_b_dev_1', 'DES_b_dev_2', 'DES_b_dev_m',
                  'DES_s_dev', 'DES_s_dev_m',
                  'DES_m_piv']:
            scaling[p] = block.get_double('mor_parameters', p)
        for p in DES_WL_prior.keys():
            scaling['DES_%s'%p] = DES_WL_prior[p]
    # HST
    scaling['bWL_HST'], scaling['DWL_HST'] = {}, {}
    for name in masscalibration.HSTcalib['SPT_ID']:
        scaling['bWL_HST'][name] = block.get_double('mor_parameters', 'bWL_HST_%s'%name)
        scaling['DWL_HST'][name] = block.get_double('mor_parameters', 'DWL_HST_%s'%name)
        for c in ['cov_HST_SZ_%s'%name, 'cov_HST_X_SZ_%s'%name, 'cov_HST_richness_SZ_%s'%name]:
            scaling[c] = block.get_double_array_nd('mor_parameters', c)
    # Covariance matrices
    for p in ['cov_X_SZ', 'cov_Megacam_SZ', 'cov_richness_SZ', 'cov_Megacam_X_SZ', ]:
        scaling[p] = block.get_double_array_nd('mor_parameters', p)

    # Halo mass function
    z, M, N = block.get_grid('HMF', 'z_arr', 'M_arr', 'dNdlnM')
    HMF = {'z_arr': z, 'lnM_arr': np.log(M), 'dNdlnM': N}

    ##### Setup lensing likelihoods
    if masscalibration.todo['WL']:
        masscalibration.WL.setup_one_cluster_mode(cosmology)

    ##### Compute likelihood
    lnlike, DES_stack = masscalibration.lnlike(HMF, cosmology, scaling)
    if np.isfinite(lnlike):
        block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)
        if masscalibration.get_stacked_DES:
            for name in ['zloxilo', 'zloxihi', 'zmidxilo', 'zmidxihi', 'zhixilo', 'zhixihi']:
                for i,n in enumerate(DES_stack['shear_%s'%name]):
                    block.put_double('DES_stack', 'shear_%s_%d'%(name,i), n)
                for i,n in enumerate(DES_stack['DeltaSigma_%s'%name]):
                    block.put_double('DES_stack', 'DeltaSigma_%s_%d'%(name,i), n)
                for i,n in enumerate(DES_stack['DeltaSigma_data_%s'%name]):
                    block.put_double('DES_stack', 'DeltaSigma_data_%s_%d'%(name,i), n)
        return 0
    else:
        return 1


def cleanup(config):
    pass
