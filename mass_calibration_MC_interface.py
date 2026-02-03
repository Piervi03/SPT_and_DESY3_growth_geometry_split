import numpy as np
from scipy.interpolate import make_interp_spline
import h5py
from cosmosis.datablock import option_section
import mass_calibration_MC as mass_calibration
import lensing


def setup(options):
    # Config parameters
    todo = {}
    for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
        todo[opt[2:]] = options.get_bool(option_section, opt, False)
    todo['lambda_min'] = options.get_bool(option_section, 'lambda_min')
    mcType = options.get_string(option_section, 'mcType', 'None')
    z_cl_min_max = options.get_double_array_1d(option_section, 'z_cl_min_max')
    # Data for optical cleaning
    if todo['lambda_min']:
        lambda_min_file = options.get_string(option_section, 'MCMF_lambda_min')
        tmp = np.genfromtxt(lambda_min_file, names=True, dtype=None)
        lambda_min = {}
        for name in tmp.dtype.names[1:]:
            lambda_min[name] = make_interp_spline(tmp['z'], tmp[name], k=1)
    else:
        lambda_min = None
    richness_scatter_model = options.get_string(option_section, 'richness_scatter_model')
    NPROC = options.get_int(option_section, 'NPROC', default=0)
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    # HST file
    HSTcalibfile = options.get_string(option_section, 'HSTcalibfile', default='None')
    # Do stack lensing for validation
    get_stacked_DES = options.get_bool(option_section, 'get_stacked_DES', default=False)

    masscalibration = mass_calibration.MassCalibration(todo=todo,
                                                       mcType=mcType,
                                                       z_cl_min_max=z_cl_min_max, lambda_min=lambda_min, richness_scatter_model=richness_scatter_model,
                                                       SPT_survey_fields=SPT_survey_fields, SPTcatalogfile=SPTcatalogfile,
                                                       HSTcalibfile=HSTcalibfile,
                                                       NPROC=NPROC, get_stacked_DES=get_stacked_DES)
    masscalibration.YXPARAM = options.get_string(option_section, 'YXPARAM', 'None')

    # Set up lensing code
    if todo['WL']:
        # Data files
        lensing_dict = {'HSTfile': options.get_string(option_section, 'HSTfile', default='None'),
                        'MegacamFile': options.get_string(option_section, 'MegacamFile', default='None'),
                        'DESfile': options.get_string(option_section, 'DESfile', default='None'),
                        'NPROC': NPROC,
                        'save_shear_profiles': get_stacked_DES,
                        }
        # DES specific
        if lensing_dict['DESfile'] != 'None':
            for name in ['DESboostfile', 'DESmiscenterfile', 'DEScentertype']:
                lensing_dict[name] = options.get_string(option_section, name)
            lensing_dict['DESboost_z_arr'] = options.get_double_array_1d(option_section, 'DESboost_z_arr')
        # HST and Megacam specific
        if (lensing_dict['HSTfile'] != 'None') or (lensing_dict['MegacamFile'] != 'None'):
            lensing_dict['mcType'] = mcType
            lensing_dict['Delta_crit'] = options.get_double(option_section, 'Delta_crit')
        # Set up lensing module
        masscalibration.WL = lensing.SPTlensing(masscalibration.catalog, **lensing_dict)
        # DES lensing priors
        if lensing_dict['DESfile'] == 'None':
            DES_WL_prior = None
        else:
            DES_WL_priors_file = options.get_string(option_section, 'DES_WL_priors_file')
            with h5py.File(DES_WL_priors_file, 'r') as f:
                DES_WL_prior = {}
                for k in f.keys():
                    DES_WL_prior[k] = f[k][()]
    else:
        DES_WL_prior = None

    return masscalibration, DES_WL_prior


def execute(block, setup_stuff):
    masscalibration, DES_WL_prior = setup_stuff
    # Extract from datablock
    cosmology = {
        'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
        'h': block.get_double('cosmological_parameters', 'hubble')/100,
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa'),
        'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
    for p in ['Omega_m', 'Omega_b', 'n_s', 'wa']:
        cosmology[p] = block.get_double('cosmological_parameters', p)

    scaling = {'YXPARAM': masscalibration.YXPARAM}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'zeta_min', 'SZmPivot']:
        scaling[p] = block.get_double('mor_parameters', p)
    for p in ['SPECS_calib', 'Delta_Csz_ECS', 'Delta_Csz_500d']:
        if block.has_value('mor_parameters', p):
            scaling[p] = block.get_double('mor_parameters', p)
    if masscalibration.todo['Yx'] or masscalibration.todo['Mgas']:
        for p in ['Ax', 'Bx', 'Cx', 'Ex', 'dlnMg_dlnr', 'XraymPivot', 'rhoSZX']:
            scaling[p] = block.get_double('mor_parameters', p)
    if masscalibration.todo['richness']:
        for p in ['Arichness', 'Brichness', 'Crichness', 'Drichness',
                  'Arichness_ext', 'Brichness_ext', 'Crichness_ext', 'Drichness_ext',
                  'z_DESWISE',
                  'richmPivot', 'rhoSZrichness']:
            scaling[p] = block.get_double('mor_parameters', p)
    if masscalibration.todo['veldisp']:
        for p in ['Adisp', 'Bdisp', 'Cdisp', 'rhoSZdisp']:
            scaling[p] = block.get_double('mor_parameters', p)
    if masscalibration.todo['WL']:
        for p in ['MegacamScatterLSS', 'bWL_Megacam', 'DWL_Megacam', 'rhoSZWL']:
            scaling[p] = block.get_double('mor_parameters', p)
        if masscalibration.todo['richness']:
            scaling['rhoWLrichness'] = block.get_double('mor_parameters', 'rhoWLrichness')
    # DES
    if DES_WL_prior is not None:
        for p in ['DES_b_dev_1', 'DES_b_dev_2', 'DES_b_dev_m',
                  'DES_s_dev', 'DES_s_dev_m',
                  'DES_m_piv']:
            scaling[p] = block.get_double('mor_parameters', p)
        for p in DES_WL_prior.keys():
            scaling['DES_%s' % p] = DES_WL_prior[p]
    # HST
    if masscalibration.todo['WL']:
        scaling['bWL_HST'], scaling['DWL_HST'] = {}, {}
        for name in masscalibration.HSTcalib['SPT_ID']:
            scaling['bWL_HST'][name] = block.get_double('mor_parameters', 'bWL_HST_%s' % name)
            scaling['DWL_HST'][name] = block.get_double('mor_parameters', 'DWL_HST_%s' % name)

    # Halo mass function
    z, M, N = block.get_grid('HMF', 'z_arr', 'M_arr', 'dNdlnM')
    HMF = {'z_arr': z, 'lnM_arr': np.log(M), 'dNdlnM': N}

    # Setup lensing likelihoods
    if masscalibration.todo['WL']:
        masscalibration.WL.setup_one_cluster_mode(cosmology)

    # lndN/dxi (computed in abundance)
    if block.has_section('cat') and block.has_value('cat', 'lndNdxi'):
        masscalibration.catalog['lndNdxi'] = block.get_double_array_1d('cat', 'lndNdxi')

    # Compute likelihood
    lnlike, DES_stack = masscalibration.lnlike(HMF, cosmology, scaling)
    if np.isfinite(lnlike):
        block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)
        if masscalibration.get_stacked_DES:
            for name in DES_stack.keys():
                for i, n in enumerate(DES_stack[name]):
                    block.put_double('DES_stack', '%s_%d' % (name, i), n)
        return 0
    else:
        print("mass calibration", flush=True)
        return 1


def cleanup(config):
    pass
