import numpy as np

import xarray as xr
import h5py
from astropy.table import Table

import set_scaling
import HMF_convo
import abundance
import abundance_lambdaselect
import mass_calibration
import lensing

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read HMF
HMF = xr.open_dataset('HMF.nc')

WLsimcalibfile = 'WLsimcalib_data.py'

# set_scaling
scaling_setter = set_scaling.SetScaling(WLsimcalibfile)

# HMF_convo
observable_pairs = ['DES_SZ', 'richness_SZ']
pairs_zmin = [.25, .25]
pairs_zmax = [1., 2]
pairs_Nz = [4, 8]

NPROC = 0
multi_obs_convolution = HMF_convo.MultiObsConvolution(observable_pairs,
                                                      pairs_zmin, pairs_zmax, pairs_Nz,
                                                      NPROC)

# abundance
NPROC = 0
surveyCutSZ = [5., 47.]
z_cl_min_max = [0.25, 2.]
surveyCutLambda = [40, 220]
# SPT survey
SPT_survey = Table.read('/Users/sbocquet/codeandstuff/SPT_cluster_data/SPT_SZ_survey_190418.txt', format='ascii.commented_header')
# Cluster catalog
catalog = Table.read('mock_191029-154550.fits')

number_count = abundance.NumberCount(catalog, SPT_survey, {},
                                     surveyCutSZ, z_cl_min_max,
                                     NPROC)
number_count_lambda = abundance_lambdaselect.NumberCount(catalog, SPT_survey, {},
                                                         surveyCutSZ, surveyCutLambda, z_cl_min_max,
                                                         NPROC)

# mass_calibration
todo = {'WL': True,
        'Yx': False,
        'Mgas': False,
        'veldisp': False,
        'richness': False}
mcType = 'None'
surveyCutSZ = [5., 47.]
z_cl_min_max = [0.25, 1.]
NPROC = 0
# SPT survey
SPT_survey_fields = '/Users/sbocquet/codeandstuff/SPT_cluster_data/SPT_SZ_survey_190418.txt'
# Double counted clusters
SPT_doublecounts = '/Users/sbocquet/codeandstuff/SPT_cluster_data/SPT_doublecounts.py'
# Cluster catalog
SPTcatalogfile = 'mock_191029-154550.fits'
##### Multi-obs HMF convolution names
observable_pairs = ['DES_SZ',]

masscalibration = mass_calibration.MassCalibration(todo,
                                                   {'SZmPivot': 3e14,
                                                    'XraymPivot': 5e14,
                                                    'richmPivot':3e14,
                                                    'YXPARAM': 'XVP',},
                                                   mcType,
                                                   surveyCutSZ, z_cl_min_max,
                                                   SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
                                                   observable_pairs,
                                                   WLsimcalibfile,# DES_betabias_file, HSTfile, MegacamFile, DESfile,
                                                   NPROC)

if todo['WL']:
    HSTfile = 'None'
    MegacamFile = 'None'
    DESfile = 'mock_WL_191029-154631.hdf5'
    DES_betabias_file = '/Users/sbocquet/codeandstuff/SPT_cluster_data/sci_uncertainty_zdiff_01_40bins_new.dat'
    masscalibration.WL = lensing.SPTlensing(masscalibration.catalog,
                                            WLsimcalibfile,
                                            HSTfile, MegacamFile, DESfile,
                                            DES_betabias_file,
                                            mcType)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cosmology = {'Omega_m': .3, 'Omega_l': .7, 'Omega_b': .04,
             'h': 0.7,
             'w0': -1., 'wa': 0.,
             'sigma8': .8, 'ns': .96,}

scaling = {'Asz': 4., 'Bsz': 1.34, 'Csz': .49, 'Dsz': .2,
           'Bsz2': 0, 'Csz2':0, 'DszM':0, 'Esz':0,
           'WLbias': 0., 'WLscatter': 0.,
           'HSTbias': 0., 'HSTscatterLSS':5.6e13,
           'MegacamBias': 0., 'MegacamScatterLSS': 6.3e13,
           'DWL_Megacam': .3, 'bWL_Megacam': 1,
           'DESbias': 0., 'DESscatterLSS': 6.3e13,
           'Adisp':939., 'Bdisp':2.91, 'Cdisp':.33, 'Ddisp0':.2, 'DdispN':3.,
           'Arichness': 70., 'Brichness': 1., 'Crichness': 1., 'Drichness': .2,
           'Ax': 6.5, 'Bx': .57, 'Cx': -.4, 'Dx': .12, 'Ex':0,
           'slope_MgR': 1.16, 'slope_MgR_std': .016,
           'rhoSZrichness': 0., 'rhoSZdisp': 0., 'rhoSZX': 0., 'rhoSZWL': 0.,
           'rhoWLX': 0., 'rhoWLrichness':0.,
           'rhoXrichness': 0,
           'SZmPivot': 3e14,
           'XraymPivot': 5e14,
           'richmPivot': 3e14,
           'YXPARAM': 'SPT_XVP',
           }

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set scaling params
scaling_setter.execute(scaling)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HMF convo

for p in ['Bsz', 'Bx', 'Brichness']:
    multi_obs_convolution.scaling[p] = scaling[p]
# Covariance matrices
for c in ['cov_X_SZ', 'cov_Megacam_SZ', 'cov_DES_SZ', 'cov_richness_SZ', 'cov_Megacam_X_SZ', 'cov_DES_X_SZ', 'cov_DES_richness_SZ']:
    multi_obs_convolution.covmat[c] = scaling[c]

# Halo mass function
multi_obs_convolution.HMF = {
    'M_arr': HMF['m'].values,
    'z_arr': HMF['z'].values,
    'dNdlnM': HMF['__xarray_dataarray_variable__'].values}
multi_obs_convolution.HMF['len_z'] = len(multi_obs_convolution.HMF['z_arr'])
dN_dmultiobs_dict = multi_obs_convolution.execute()
print(dN_dmultiobs_dict.keys())


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Abundance

number_count.cosmology = cosmology
number_count.scaling = scaling
# Halo mass function
number_count.HMF = {
    'M_arr': HMF['m'].values,
    'z_arr': HMF['z'].values,
    'dNdlnM': HMF['__xarray_dataarray_variable__'].values}
number_count.HMF['len_z'] = len(number_count.HMF['z_arr'])
# Compute the likelihood
lnlike = float(number_count.lnlike())
print('lnlike', lnlike)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

number_count_lambda.cosmology = cosmology
number_count_lambda.scaling = scaling
# zeta-lambda HMF
number_count_lambda.HMF_zetalambda = {'dN_dlnM': dN_dmultiobs_dict['richness_SZ'],
                                      'M_arr': dN_dmultiobs_dict['M_arr'],
                                      'z_arr': dN_dmultiobs_dict['richness_SZ_z']}
number_count_lambda.HMF_zetalambda['len_z'] = len(number_count_lambda.HMF_zetalambda['z_arr'])

# Compute the likelihood
lnlike = float(number_count_lambda.lnlike())
print('lnlike', lnlike)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Lensing likelihoods

if masscalibration.todo['WL']:
    masscalibration.WL.like_all(masscalibration.catalog,
                                cosmology, scaling)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mass calibration

masscalibration.cosmology = cosmology
masscalibration.scaling = scaling
masscalibration.HMF_convos = {}
masscalibration.HMF_convos['M_arr'] = dN_dmultiobs_dict['M_arr']
for pair_name in masscalibration.observable_pairs:
    masscalibration.HMF_convos[pair_name] = dN_dmultiobs_dict[pair_name]
    masscalibration.HMF_convos['%s_z'%pair_name] = dN_dmultiobs_dict['%s_z'%pair_name]


##### Compute likelihood
lnlike = masscalibration.lnlike()
print('lnlike', lnlike)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
