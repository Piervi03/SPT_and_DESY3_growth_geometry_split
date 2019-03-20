from __future__ import division
import numpy as np
import os
from scipy.interpolate import RectBivariateSpline
from astropy.table import Table

from cosmosis.datablock import option_section
import cosmo, Mconversion_concentration

################################################################################
class MarginalizeMass:
    def __init__(self, options):
        ##### Global variables
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        # SPT survey
        SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
        assert os.path.isfile(SPT_survey_fields), "SPT survey table does not exist"
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        # Cluster catalog
        SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        ##### M-c relation for M200
        self.MCrel = Mconversion_concentration.ConcentrationConversion('Duffy08')


    ##########
    def do_it(self, block):
        """Return ln-likelihood for SPT cluster abundance."""
        # Only need cosmo for E(z)-type stuff
        self.cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa')}
        # SZ scaling relation parameters
        self.Asz = block.get_double('mor_parameters', 'Asz')
        self.Bsz = block.get_double('mor_parameters', 'Bsz')
        self.Csz = block.get_double('mor_parameters', 'Csz')
        self.Dsz = block.get_double('mor_parameters', 'Dsz')
        # Halo mass function
        self.HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
            'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
            'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
        self.HMF['len_z'] = len(self.HMF['z_arr'])

        ##### Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in))


        M500, M200, weight = [], [], []
        for i,name in enumerate(self.catalog['SPT_ID']):
            if self.catalog['REDSHIFT_LIMIT'][i]>0:
                continue
            if self.catalog['REDSHIFT'][i]==0:
                continue

            # Normalized HMF
            HMF_z_ = np.exp(HMF_interp(np.log(self.catalog['REDSHIFT'][i]), np.log(self.HMF['M_arr'])))[0]
            HMF_z_/= np.sum(np.diff(self.HMF['M_arr']) * np.exp(.5*np.log(HMF_z_[1:]*HMF_z_[:-1])))

            xi = 0
            while xi<=2.65:
                # Measurement error
                xi = np.random.normal(self.catalog['XI'][i], 1)
            # Go to zeta
            zeta = self.xi2zeta(xi)
            # Intrinsic scatter
            zeta_true = np.random.lognormal(np.log(zeta), self.Dsz)
            # Weight with HMF
            field_factor = self.SPT_survey['GAMMA'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]]
            M500_ = self.zeta2mass(zeta_true, self.catalog['REDSHIFT'][i], field_factor)
            M200_ = self.MCrel.MDelta_to_M200(M500_, 500., self.catalog['REDSHIFT'][i])
            # P(M,z) a.k.a. the halo mass function
            weight_ = np.exp(np.interp(np.log(M500_), np.log(self.HMF['M_arr']), np.log(HMF_z_)))

            M500.append(M500_)
            M200.append(M200_)
            weight.append(weight_)

        for i in range(len(M500)):
            block.put_double('marge_mass', 'M500_%d'%i, M500[i])
            block.put_double('marge_mass', 'weight_%d'%i, weight[i])
            block.put_double('marge_mass', 'M200_%d'%i, M200[i])




    ########## Utility functions

    def xi2zeta(self, xi):
        if xi>2.65:
            return (xi**2 - 3)**.5
        else:
            return 0

    def zeta2mass(self, zeta, z, field_factor):
        Asz = self.Asz * field_factor
        zterm = (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.Csz
        return self.SZmPivot * (zeta / Asz / zterm)**(1/self.Bsz)
