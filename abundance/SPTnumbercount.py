from __future__ import division
import numpy as np
import scipy.ndimage
from cosmosis.datablock import option_section
import os
from astropy.table import Table
from scipy.stats import norm
from scipy import interpolate
import imp

class NumberCount:
    def __init__(self, options):
        ##### Global variables
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        ##### SPT survey
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        SPTcatalogfile = SPTdata.SPTcatalogfile
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        self.SPTfieldNames = SPTdata.SPTfieldNames
        self.SPTfieldCorrection = SPTdata.SPTfieldCorrection
        self.SPTfieldSize = SPTdata.SPTfieldSize
        self.SPTnFalse_alpha = SPTdata.SPTnFalse_alpha
        self.SPTnFalse_beta = SPTdata.SPTnFalse_beta
        ##### Various observable arrays
        # Lin spaced for convo with unit scatter
        self.xi_bins = np.linspace(2.7, 50, 474)
        self.dxi = self.xi_bins[1] - self.xi_bins[0]
        # ln(zeta(xi_bins))
        self.ln_zeta_xi_arr = np.log(self.xi2zeta(self.xi_bins))
        # dlnzeta/dxi (xi_bins)
        self.dlnzeta_dxi_arr = self.dlnzeta_dxi(self.xi_bins)
        # Arrays over which we'll integrate (survey cuts applied)
        Nxi = int(np.log10(self.surveyCutSZ[1]/self.surveyCutSZ[0])/.005 + 1)
        self.xi_arr = np.logspace(np.log10(self.surveyCutSZ[0]), np.log10(self.surveyCutSZ[1]), Nxi)
        dz = .01
        Nz = int((self.surveyCutRedshift[1]-self.surveyCutRedshift[0])/dz + 1)
        self.z_arr = np.linspace(self.surveyCutRedshift[0], self.surveyCutRedshift[1], Nz)



    ########## lnlikelihood
    def lnlike(self, block):
        # Need cosmo params for E(z)
        self.Omega_m = block['cosmological_parameters', 'Omega_m']
        self.Omega_l = block['cosmological_parameters', 'omega_lambda']
        self.w0 = block['cosmological_parameters', 'w']
        # SZ scaling relation parameters
        self.Asz = block['mor_parameters', 'Asz']
        self.Bsz = block['mor_parameters', 'Bsz']
        self.Csz = block['mor_parameters', 'Csz']
        self.Dsz = block['mor_parameters', 'Dsz']
        # Advanced SZ scaling parameters
        self.Bsz2 = block['mor_parameters', 'Bsz2']
        self.Csz2 = block['mor_parameters', 'Csz2']
        self.Esz = block['mor_parameters', 'Esz']
        self.DszM = block['mor_parameters', 'DszM']

        # Initialize ln-likelihood to 0.
        lnlike = 0.



        # ##### Convert HMF to dN/dln(zeta) = dN/dlog10(M) * dlog10(M)/dln(zeta)
        # if ((self.Bsz2!=0.)|(self.Csz2!=0)|(self.Esz!=0.)|(self.DszM!=0.)):
        #     lnzetaM = np.log(self.mass2zeta(HMF['M_arr'], HMF['z_arr']))
        #
        # # dln(M)/dln(zeta)
        # if ((self.Bsz2==0.)&(self.Esz==0.)):
        #     dlnM_dlnzeta = 1/self.Bsz
        # else:
        #     lnEz_E0p6 = np.log(self.Ez(HMF['z_arr'])/self.Ez(.6))
        #     lnmassRatio = np.log(HMF['M_arr']/self.SZmPivot)
        #     bLin = self.Bsz + self.Esz*lnEz_E0p6
        #     cEff = self.Csz*lnEz_E0p6 + self.Csz2*lnEz_E0p6**2
        #     # [z,M]
        #     dlnzeta_dlnmRatio = bLin[:,None] + 2*self.Bsz2*lnmassRatio[None,:]
        #     if np.any(dlnzeta_dlnmRatio<=0.):
        #         return -np.inf
        #     # [z,M]
        #     sqrtTerm = bLin[:,None]**2 - 4.*self.Bsz2* (cEff[:,None] + np.log(self.Asz) - lnzetaM)
        #     if np.any(sqrtTerm<0.):
        #         return -np.inf
        #     dlnM_dlnzeta = sqrtTerm**-.5
        #
        # dN_dlnzeta_noScatter = HMF['dNdlnM'] * dlnM_dlnzeta
        #
        # # Concolve with intrinsic scatter
        # if((self.Bsz2==0.)&(self.Csz2==0)&(self.Esz==0.)&(self.DszM==0)):
        #     dlnzeta = self.Bsz*np.log(HMF['M_arr'][1]/HMF['M_arr'][0])
        #     Nbin = self.Dsz / dlnzeta
        #     dN_dlnzeta_unitSolidAng = scipy.ndimage.gaussian_filter1d(dN_dlnzeta_noScatter, Nbin, axis=1, mode='constant')
        # else:
        #     scatter = (self.Dsz**2 + self.DszM**2*(HMF['M_arr']/3e14)**(2*scaling['DszMslope']))**.5
        #     scatter[np.where(scatter<.01)[0]] = .01
        #     dN_dlnzeta_unitSolidAng = np.empty((HMF['len_z'],HMF['len_M']))
        #     for i in range(HMF['len_z']):
        #         lnzetaArr = lnzetaM[i]
        #         integrand = dN_dlnzeta_noScatter[i,None,:] * norm.pdf(lnzetaArr[:,None], lnzetaArr[None,:], scatter)
        #         dN_dlnzeta_unitSolidAng[i] = np.trapz(integrand, lnzetaArr, axis=1)
        #
        #
        # Ntotal = 0.
        # ##### Now go and get lnlike for each SPT field
        # for fieldidx,field in enumerate(self.SPTfieldNames):
        #     ##### dN/dln(zeta)
        #     dN_dlnzeta = dN_dlnzeta_unitSolidAng * self.SPTfieldSize[fieldidx] * (np.pi/180)**2.
        #
        #     ##### zeta[z,M]
        #     zeta_m = self.mass2zeta(HMF['M_arr'], HMF['z_arr'])
        #
        #     #### Apply field scaling factor
        #     zeta_m*= self.SPTfieldCorrection[fieldidx]
        #
        #     ##### dN/dxi = dN/dlnzeta dlnzeta/dxi (unconvolved)
        #     # Unfortunately, the zeta_m table is not regular
        #     # and repeated spline interp is way too slow (1.6sec per field)
        #     # So we do linear interpolation (in ln(M), and for ln(dN/dlnzeta))
        #     dN_dxi = self.dlnzeta_dxi_arr * np.exp(np.array([np.interp(self.ln_zeta_xi_arr, np.log(zeta_m[i]), np.log(dN_dlnzeta[i])) for i in range(HMF['len_z'])]))
        #
        #     # Convolve with unit scatter (measurement uncertainty)
        #     dN_dxi = scipy.ndimage.gaussian_filter1d(dN_dxi, 1./dxi, axis=1, mode='constant')
        #
        #     # Set up interpolation for cluster list below
        #     lndNdxi = interpolate.interp2d(np.log(xi_bins), np.log(HMF['z_arr'][1:]), np.log(dN_dxi[1:,:]), kind='cubic')
        #
        #     # Ntotal
        #     dNdz = np.array([np.sum(np.exp(.5*(lndNdxi(np.log(xi_arr[1:]), np.log(z)) + lndNdxi(np.log(xi_arr[:-1]), np.log(z)))) * (xi_arr[1:]-xi_arr[:-1])) for z in z_arr])
        #     Ntotal+= np.trapz(dNdz, z_arr)
        #
        #     ##### cluster contributions
        #     thisfield_conf = np.where((catalog['field']==field) & (catalog['xi']>=surveyCutSZ[0]) & (catalog['xi']<=surveyCutSZ[1]) & (catalog['redshift']>=surveyCutRedshift[0]) & (catalog['redshift']<=surveyCutRedshift[1]))[0]
        #     for i in thisfield_conf:
        #
        #         ##### spec-z
        #         if catalog['redshift_err'][i]==0.:
        #             thislnlike = lndNdxi(np.log(catalog['xi'][i]), np.log(catalog['redshift'][i]))
        #
        #         ##### photo-z
        #         elif catalog['redshift_err'][i]>0.:
        #             zlo = min((.25, catalog['redshift'][i]-4*catalog['redshift_err'][i]))
        #             zhi = max((HMF['z_arr'][-1], catalog['redshift'][i]+4*catalog['redshift_err'][i]))
        #             zarr = np.linspace(zlo, zhi, 15)
        #             integrand = np.exp(lndNdxi(np.log(catalog['xi'][i]), np.log(zarr))[:,0]) * norm.pdf(zarr, catalog['redshift'][i], catalog['redshift_err'][i])
        #             thislnlike = np.log(np.trapz(integrand, zarr))
        #
        #
        #         lnlike+= thislnlike
        #
        #
        #     ##### unconfirmed candidates
        #     thisfield_unconf = np.where((catalog['field']==field) & (catalog['xi']>=surveyCutSZ[0]) & (catalog['xi']<=surveyCutSZ[1]) & (catalog['redshift']==0.) & (catalog['redshift_lim']<=surveyCutRedshift[1]))[0]
        #     for i in thisfield_unconf:
        #         dNdxifalse = SPTnFalse_beta[fieldidx] * SPTfieldSize[fieldidx]/2500. * SPTnFalse_alpha[fieldidx] * np.exp(-SPTnFalse_beta[fieldidx]*(catalog['xi'][i]-5.))
        #         zarr = np.linspace(catalog['redshift_lim'][i], HMF['z_arr'][-1], 25)
        #         thislnlike = np.log(dNdxifalse + np.trapz(np.exp(lndNdxi(np.log(catalog['xi'][i]), np.log(zarr))[:,0]), zarr))
        #
        #         lnlike+= thislnlike
        #
        #
        # ##### Add total number of clusters contribution
        # lnlike-= Ntotal
        #
        # #print 'Ntotal',Ntotal
        # #print 'sigma8', cosmology['sigma8']
        lnlike = 12.3
        return lnlike



    ########## Utility functions

    def Ez(self, z):
        return (self.Omega_m*(1+z)**3. + self.Omega_l*(1+z)**(3.*(1+self.w0)))**.5

    def dlnzeta_dxi(self, xi):
        return xi/(xi**2-3.)

    def xi2zeta(self, xi):
        return (xi**2 - 3.)**.5

    def mass2zeta(self, mass, z):
        # [redshift][mass]
        massterm = (mass/SZmPivot)**self.Bsz
        zterm = (self.Ez(z, cosmology)/self.Ez(.6, cosmology))**self.Csz
        return self.Asz * massterm[None,:] * zterm[:,None]
