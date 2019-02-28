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
import cosmo, Mconversion_concentration, observablecovmat
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
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        self.XraymPivot = options.get_double(option_section, 'XraymPivot')
        self.richmPivot = options.get_double(option_section, 'richmPivot')
        self.YXPARAM = options.get_string(option_section, 'YXPARAM')
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
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        ##### Extract from datablock
        self.cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
            'Omega_b': block.get_double('cosmological_parameters', 'Omega_b'),
            'h': block.get_double('cosmological_parameters', 'hubble')/100,
            'ns': block.get_double('cosmological_parameters', 'n_s'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa'),
            'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
        self.scaling = {
            # SZ
            'Asz': block.get_double('mor_parameters', 'Asz'),
            'Bsz': block.get_double('mor_parameters', 'Bsz'),
            'Csz': block.get_double('mor_parameters', 'Csz'),
            'Dsz': block.get_double('mor_parameters', 'Dsz'),
            'Bsz2': block.get_double('mor_parameters', 'Bsz2'),
            'Csz2': block.get_double('mor_parameters', 'Csz2'),
            'Esz': block.get_double('mor_parameters', 'Esz'),
            'DszM': block.get_double('mor_parameters', 'DszM'),
            # X-ray
            'Ax': block.get_double('mor_parameters', 'Ax'),
            'Bx': block.get_double('mor_parameters', 'Bx'),
            'Cx': block.get_double('mor_parameters', 'Cx'),
            'Dx': block.get_double('mor_parameters', 'Dx'),
            'Ex': block.get_double('mor_parameters', 'Ex'),
            'dlnMg_dlnr': block.get_double('mor_parameters', 'dlnMg_dlnr'),
            # WL
            'WLbias': block.get_double('mor_parameters', 'WLbias'),
            'WLscatter': block.get_double('mor_parameters', 'WLscatter'),
            'HSTbias': block.get_double('mor_parameters', 'HSTbias'),
            'HSTscatterLSS': block.get_double('mor_parameters', 'HSTscatterLSS'),
            'MegacamBias': block.get_double('mor_parameters', 'MegacamBias'),
            'MegacamScatterLSS': block.get_double('mor_parameters', 'MegacamScatterLSS'),
            'DESbias': block.get_double('mor_parameters', 'DESbias'),
            'DESscatterLSS': block.get_double('mor_parameters', 'DESscatterLSS'),
            # Richness
            'Arichness': block.get_double('mor_parameters', 'Arichness'),
            'Brichness': block.get_double('mor_parameters', 'Brichness'),
            'Crichness': block.get_double('mor_parameters', 'Crichness'),
            'Drichness': block.get_double('mor_parameters', 'Drichness'),
            # dispersion
            'Adisp': block.get_double('mor_parameters', 'Adisp'),
            'Bdisp': block.get_double('mor_parameters', 'Bdisp'),
            'Cdisp': block.get_double('mor_parameters', 'Cdisp'),
            'Ddisp0': block.get_double('mor_parameters', 'Ddisp0'),
            'DdispN': block.get_double('mor_parameters', 'DdispN'),
            # Correlation coefficients
            'rhoSZWL': block.get_double('mor_parameters', 'rhoSZWL'),
            'rhoWLX': block.get_double('mor_parameters', 'rhoWLX'),
            'rhoSZX': block.get_double('mor_parameters', 'rhoSZX'),
            'rhoSZrichness': block.get_double('mor_parameters', 'rhoSZrichness'),
            'rhoXdisp': block.get_double('mor_parameters', 'rhoXdisp'),
            'rhoSZdisp': block.get_double('mor_parameters', 'rhoSZdisp'),
            }
        # Halo mass function
        self.HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
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



    ####################
    def obs2mass(self, name, obs, z):
        """Returns mass given (observable, z) using scaling relation."""
        if name=='zeta':
            Asz = self.thisSPTfield_gamma * self.scaling['Asz']
            lnM = np.log(self.SZmPivot) + (np.log(obs) - np.log(Asz)\
                - self.scaling['Csz']*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology)))/(self.scaling['Bsz']\
                + self.scaling['Esz']*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology)))
            return np.exp(lnM)
        elif name=='Yx':
            if self.YXPARAM=='SPT_XVP':
                return 1e14 * self.scaling['Ax'] * self.cosmology['h']**1.5\
                    * (self.cosmology['h']/.72)**(2.5*self.scaling['Bx']-1.5)\
                    * (obs/3.)**self.scaling['Bx'] * cosmo.Ez(z, self.cosmology)**self.scaling['Cx']
            elif self.YXPARAM=='Munich':
                return self.XraymPivot * self.cosmology['h']**1.5 * (obs/(self.scaling['Ax']
                    *(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx']))**(1/self.scaling['Bx'])
        elif name=='Mgas':
            return self.XraymPivot * self.cosmology['h'] * (obs/self.XraymPivot/self.scaling['Ax']
                /(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx'])**(1./self.scaling['Bx'])
        elif name=='disp':
            h70z = self.cosmology['h']/.7*cosmo.Ez(z, self.cosmology)
            M200c = 1e15*self.cosmology['h'] * (obs/self.scaling['Adisp']/h70z**self.scaling['Cdisp'])**self.scaling['Bdisp']
            return np.exp(self.lnM200_to_lnM500(z, np.log(M200c)))
        elif name=='richness':
            return self.richmPivot* (obs/self.scaling['Arichness']
                /(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Crichness'])**(1/self.scaling['Brichness'])
        elif name=='WLMegacam':
            return obs/self.scaling['bWL_Megacam']
        elif name=='WLHST':
            return obs/self.scaling['bWL_HST']
        elif name=='WLDES':
            return obs/self.scaling['bWL_DES']
        else:
            raise ValueError("Observable not known:",name)


    ####################
    def mass2obs(self, name, mass, z):
        """Returns observable given (mass, z) using scaling relation."""
        if name=='zeta':
            lnzeta = np.log(self.scaling['Asz']*self.thisSPTfield_gamma)\
                + self.scaling['Bsz'] * np.log(mass/self.SZmPivot)\
                + self.scaling['Csz'] * np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))\
                + self.scaling['Esz'] * np.log(mass/self.SZmPivot)*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))
            return np.exp(lnzeta)
        elif name=='Yx':
            if self.YXPARAM=='SPT_XVP':
                return 3.*(mass*1e-14/(self.scaling['Ax'] * self.cosmology['h']**1.5
                    * (self.cosmology['h']/.72)**(2.5*self.scaling['Bx']-1.5)
                    * cosmo.Ez(z, self.cosmology)**self.scaling['Cx']))**(1/self.scaling['Bx'])
            elif self.YXPARAM=='Munich':
                return self.scaling['Ax']* (mass/self.cosmology['h']**1.5/self.XraymPivot)**self.scaling['Bx']\
                    * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx']
        elif name=='Mgas':
            lnMgas = np.log(self.XraymPivot * self.scaling['Ax']) + self.scaling['Bx']*np.log(mass/self.XraymPivot/self.cosmology['h'])\
                + self.scaling['Cx']*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))\
                + self.scaling['Ex']*np.log(mass/self.XraymPivot/self.cosmology['h'])*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))
            return np.exp(lnMgas)
        elif name=='disp':
            h70z = self.cosmology['h']/.7*cosmo.Ez(z, self.cosmology)
            M200c = np.exp(self.lnM500_to_lnM200(z, np.log(mass)))
            if len(M200c)==1: M200c = M200c[0]
            return self.scaling['Adisp'] * (M200c/1e15/self.cosmology['h'])**(1/self.scaling['Bdisp']) * h70z**self.scaling['Cdisp']
        elif name=='richness':
            return self.scaling['Arichness'] * (mass/self.richmPivot)**self.scaling['Brichness']\
                * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Crichness']
        elif name=='WLMegacam':
            return self.scaling['bWL_Megacam'] * mass
        elif name=='WLHST':
            return self.scaling['bWL_HST'] * mass
        elif name=='WLDES':
            return self.scaling['bWL_DES'] * mass
        else:
            raise ValueError("Observable not known:",name)


    ####################
    def dlnM_dlnobs(self, name, M0_arr=None, z=None):
        """Returns dlnM/dln(obs) for a given observable."""
        if name=='zeta': return 1/self.scaling['Bsz']
        elif name=='richness': return 1/self.scaling['Brichness']
        elif name=='Yx':
            if self.YXPARAM=='SPT_XVP': return 1/(1/self.scaling['Bx'] - self.scaling['dlnMg_dlnr']/3)
            elif self.YXPARAM=='Munich': return 1/(self.scaling['Bx'] - self.scaling['dlnMg_dlnr']/3)
        elif name=='Mgas': return 1/(self.scaling['Bx'] - self.scaling['dlnMg_dlnr']/3)
        elif (name=='WLMegacam')|(name=='WLHST')|(name=='WLDES'): return 1.
        elif name=='disp':
            dlnM = np.log(1.01)
            dlnobs = np.log(self.mass2obs('disp', 1.01*M0_arr, z)/self.mass2obs('disp', M0_arr, z))
            if np.any(dlnobs==0.):
                if dlnobs[-1]==0: dlnobs[-1] = dlnobs[-2]
            return dlnM/dlnobs
