from __future__ import division
import numpy as np
import os
import imp
from multiprocessing import Pool
import scipy.ndimage
import scipy.special as ss
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
from scipy import signal
from scipy.stats import norm
from scipy.stats import multivariate_normal

from cosmosis.datablock import option_section
import cosmo, Mconversion_concentration, observablecovmat, scaling_relations
import convolution


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlike(*arg)

################################################################################
class Convolution:

    def __init__(self, options):
        ##### Config parameters
        self.todo = {
            'WL': options.get_bool(option_section, 'doWL'),
            'Yx': options.get_bool(option_section, 'doYx'),
            'Mgas': options.get_bool(option_section, 'doMgas'),
            'veldisp': options.get_bool(option_section, 'doveldisp'),
            'richness': options.get_bool(option_section, 'dorichness'),
            }
        self.scaling = {}
        for opt in ['SZmPivot', 'XraymPivot', 'richmPivot', 'YXPARAM']:
            self.scaling[opt] = options.get_double(option_section, opt)
        self.mcType = options.get_string(option_section, 'mcType')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        self.NPROC = options.get_int(option_section, 'NPROC')
        # SPT survey
        SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
        assert os.path.isfile(SPT_survey_fields), "SPT survey table does not exist"
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        # Double counted clusters
        SPT_doublecounts = options.get_string(option_section, 'SPT_doublecounts')
        SPTdata = imp.load_source('SPTdata', SPT_doublecounts)
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        # Cluster catalog
        SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        ##### WL simulation calibration
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

        # Weak lensing
        if self.todo['WL']:
            self.WL = lensing.SPTlensing(options, self.catalog)


    ############################################################################
    def compute(self, block):
        ##### Extract from datablock
        self.cosmology = {
            'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
            'Omega_b': block.get_double('cosmological_parameters', 'Omega_b'),
            'h': block.get_double('cosmological_parameters', 'hubble')/100,
            'ns': block.get_double('cosmological_parameters', 'n_s'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa'),
            'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
        # SZ
        for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Bsz2', 'Csz2', 'Esz', 'DszM']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # X-ray
        for p in ['Ax', 'Bx', 'Cx', 'Dx', 'Ex', 'dlnMg_dlnr']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # WL
        for p in ['WLbias', 'WLscatter', 'HSTbias', 'HSTscatterLSS', 'MegacamBias', 'MegacamScatterLSS', 'DESbias', 'DESscatterLSS']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # Richness
        for p in ['Arichness', 'Brichness', 'Crichness', 'Drichness']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # dispersion
        for p in ['Adisp', 'Bdisp', 'Cdisp', 'Ddisp0', 'DdispN']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # Correlation coefficients
        for p in ['rhoSZWL', 'rhoSZX', 'rhoWLX', 'rhoSZrichness', 'rhoXdisp', 'rhoSZdisp']:
            self.scaling[p] = block.get_double('mor_parameters', p)

        # Halo mass function
        self.HMF = {
            'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
            'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
            'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
        self.HMF['len_z'] = len(self.HMF['z_arr'])

        ##### Setup stuff for WL
        if self.todo['WL']:
            # Set bias and scatter for Megacam and DES from sim calibration
            # and nuissance parameters
            self.WL.set_scaling(self.scaling)


        ##### Populate and check observable covariance matrices
        self.covmat = {'invertible': True}
        if not observablecovmat.set_covmats(self.todo, self.scaling, self.covmat):
            self.covmat['invertible'] = False
            return -np.inf

        ##### Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in))


        ##### Pre-compute the intrinsic scatter convolutions
        self.P_obszeta_M_grid = {}
        if self.todo['WL']:
            self.P_obszeta_M_grid['WLDES'] = self.get_P_2obs_allz(obsname='WLDES')
        if self.todo['Yx']:
            self.P_obszeta_M_grid['Yx'] = self.get_P_2obs_allz(obsname='Yx')
        if self.todo['Yx']:
            self.P_obszeta_M_grid['Mgas'] = self.get_P_2obs_allz(obsname='Mgas')

        if self.todo['WL'] and self.todo['Yx']:
            self.P_obszeta_M_grid['WLYx'] = self.get_P_3obs_allz(obsnames=['WL', 'Yx'], covname='WLYx')
        if self.todo['WL'] and self.todo['Mgas']:
            self.P_obszeta_M_grid['WLMgas'] = self.get_P_3obs_allz(obsnames=['WL', 'Mgas'], covname='WLMgas')




    def get_P_2obs_allz(self, obsname):
            """Return P(obs, xi | M, z, p) for each redshift in HMF. Optional
            multiprocess."""

            if self.NPROC==0:
                # Iterate through redshift array
                P_2obs_grid = np.array([self.get_P_2obs_z_fixedkernel(i, obsname) for i in range(self.HMF['len_z'])])
            else:
                # Launch a multiprocessing pool and get the likelihoods
                pool = Pool(processes=self.NPROC)
                argin = zip([self]*self.HMF['len_z'], range(self.HMF['len_z']), obsname)
                P_2obs_grid = pool.map(unwrap_self_P2obs, argin)
                pool.close()
            return P_2obs_grid


        def get_P_3obs_allz(self, obsnames, covname):
            """Return P(obs0, obs1, xi | M, z, p) for each redshift in HMF. Optional
            multiprocess."""

            if self.NPROC==0:
                # Iterate through redshift array
                P_3obs_grid = np.array([self.get_P_3obs_z_fixedkernel(i, obsnames, covname) for i in range(self.HMF['len_z'])])
            else:
                # Launch a multiprocessing pool and get the likelihoods
                pool = Pool(processes=self.NPROC)
                argin = zip([self]*self.HMF['len_z'], range(self.HMF['len_z']), obsnames, covname)
                P_3obs_grid = pool.map(unwrap_self_P3obs, argin)
                pool.close()
            return P_3obs_grid
