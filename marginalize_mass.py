import numpy as np
import os
from scipy.interpolate import RectBivariateSpline
from astropy.table import Table

import cosmo, Mconversion_concentration

################################################################################
class MarginalizeMass:
    def __init__(self, SPTcatalogfile, SPT_survey_fields, SZmPivot):
        self.catalog = Table.read(SPTcatalogfile)
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        self.SZmPivot = SZmPivot
        # M-c relation for M200
        self.MCrel = Mconversion_concentration.ConcentrationConversion('Duffy08')


    ##########
    def do_it(self):
        """Return ln-likelihood for SPT cluster abundance."""
        ##### Set up interpolation for HMF
        with np.errstate(divide='ignore'):
            lnHMF_in = np.log(self.HMF['dNdlnM'])
        HMF_interp = RectBivariateSpline(self.HMF['z_arr'], np.log(self.HMF['M_arr']), lnHMF_in, kx=1, ky=1)

        ##### Now go through cluster catalog
        M500, M200, weight = [], [], []
        for i,name in enumerate(self.catalog['SPT_ID']):
            if self.catalog['REDSHIFT_LIMIT'][i]>0:
                continue
            if self.catalog['REDSHIFT'][i]==0:
                continue

            # Normalized HMF
            lnHMF_z_ = HMF_interp(self.catalog['REDSHIFT'][i], np.log(self.HMF['M_arr']))[0]
            Ntot_ = np.sum(np.diff(self.HMF['M_arr']) * np.exp(.5*(lnHMF_z_[1:]+lnHMF_z_[:-1])))

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
            weight_ = np.exp(HMF_interp(self.catalog['REDSHIFT'][i], np.log(M500_)))[0,0] / Ntot_

            M500.append(M500_)
            M200.append(M200_)
            weight.append(weight_)

        output_arr = np.vstack((M500, M200, weight))

        return output_arr



    ########## Utility functions
    def xi2zeta(self, xi):
        if xi>2.65:
            return np.sqrt(xi**2 - 3)
        else:
            return 0

    def zeta2mass(self, zeta, z, field_factor):
        Asz = self.Asz * field_factor
        zterm = (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.Csz
        return self.SZmPivot * (zeta / Asz / zterm)**(1/self.Bsz)
