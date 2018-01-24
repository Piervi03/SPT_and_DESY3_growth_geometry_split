from __future__ import division
import numpy as np
import os
import imp
from multiprocessing import Pool
import scipy.ndimage
from scipy.stats import norm
from scipy import interpolate
from astropy.table import Table

from cosmosis.datablock import option_section
import cosmo


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return NumberCount.lnlike_field(*arg)


################################################################################
class NumberCount:
    def __init__(self, options):
        ##### Global variables
        self.NPROC = options.get_int(option_section, 'NPROC')
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        ##### SPT survey
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
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



    ##########
    def lnlike(self, block):
        """Return ln-likelihood for SPT cluster abundance."""
        # Only need cosmo for E(z)-type stuff
        self.cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
            'w0': block.get_double('cosmological_parameters', 'w')}
        # SZ scaling relation parameters
        self.Asz = block.get_double('mor_parameters', 'Asz')
        self.Bsz = block.get_double('mor_parameters', 'Bsz')
        self.Csz = block.get_double('mor_parameters', 'Csz')
        self.Dsz = block.get_double('mor_parameters', 'Dsz')
        # Advanced SZ scaling parameters
        self.Bsz2 = block.get_double('mor_parameters', 'Bsz2')
        self.Csz2 = block.get_double('mor_parameters', 'Csz2')
        self.Esz = block.get_double('mor_parameters', 'Esz')
        self.DszM = block.get_double('mor_parameters', 'DszM')
        # Halo mass function
        self.HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
            'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
            'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
        self.HMF['len_z'] = len(self.HMF['z_arr'])

        ##### Convert HMF to dN/dln(zeta) = dN/dlog10(M) * dlog10(M)/dln(zeta)
        if ((self.Bsz2!=0.)|(self.Csz2!=0)|(self.Esz!=0.)|(self.DszM!=0.)):
            lnzetaM = np.log(self.mass2zeta(self.HMF['M_arr'], self.HMF['z_arr']))

        # dln(M)/dln(zeta)
        if ((self.Bsz2==0.)&(self.Esz==0.)):
            dlnM_dlnzeta = 1/self.Bsz
        else:
            lnEz_E0p6 = np.log(cosmo.Ez(self.HMF['z_arr'], self.cosmology)/cosmo.Ez(.6, self.cosmology))
            lnmassRatio = np.log(self.HMF['M_arr']/self.SZmPivot)
            bLin = self.Bsz + self.Esz*lnEz_E0p6
            cEff = self.Csz*lnEz_E0p6 + self.Csz2*lnEz_E0p6**2
            # [z,M]
            dlnzeta_dlnmRatio = bLin[:,None] + 2*self.Bsz2*lnmassRatio[None,:]
            if np.any(dlnzeta_dlnmRatio<=0.):
                return -np.inf
            # [z,M]
            sqrtTerm = bLin[:,None]**2 - 4.*self.Bsz2* (cEff[:,None] + np.log(self.Asz) - lnzetaM)
            if np.any(sqrtTerm<0.):
                return -np.inf
            dlnM_dlnzeta = sqrtTerm**-.5

        dN_dlnzeta_noScatter = self.HMF['dNdlnM'] * dlnM_dlnzeta

        # Concolve with intrinsic scatter
        if((self.Bsz2==0.)&(self.Csz2==0)&(self.Esz==0.)&(self.DszM==0)):
            dlnzeta = self.Bsz*np.log(self.HMF['M_arr'][1]/self.HMF['M_arr'][0])
            Nbin = self.Dsz / dlnzeta
            self.dN_dlnzeta_unitSolidAng = scipy.ndimage.gaussian_filter1d(dN_dlnzeta_noScatter, Nbin, axis=1, mode='constant')
        else:
            scatter = (self.Dsz**2 + self.DszM**2*(self.HMF['M_arr']/3e14)**(2*scaling['DszMslope']))**.5
            scatter[np.where(scatter<.01)[0]] = .01
            self.dN_dlnzeta_unitSolidAng = np.empty((self.HMF['len_z'],self.HMF['len_M']))
            for i in range(self.HMF['len_z']):
                lnzetaArr = lnzetaM[i]
                integrand = dN_dlnzeta_noScatter[i,None,:] * norm.pdf(lnzetaArr[:,None], lnzetaArr[None,:], scatter)
                self.dN_dlnzeta_unitSolidAng[i] = np.trapz(integrand, lnzetaArr, axis=1)

        ##### Evaluate (log)-likelihood for each SPT field (optional multiprocessing)
        num_fields = len(self.SPTfieldNames)
        if self.NPROC==0:
            field_results = [self.lnlike_field(fieldidx) for fieldidx in range(num_fields)]
        else:
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*num_fields, range(num_fields))
            field_results = pool.map(unwrap_self_f, argin)
            pool.close()
        field_results = np.array(field_results)
        lnlike = np.sum(field_results[:,0])
        Ntotal = np.sum(field_results[:,1])

        if __debug__:
            print 'lnlike %.3f, Ntotal %.2f'%(lnlike, Ntotal)

        return lnlike


    ##########
    def lnlike_field(self, fieldidx):
        """Returns (ln-likelihood, Ntotal) for a given SPT field (index)."""
        # dN/dln(zeta)
        dN_dlnzeta = self.dN_dlnzeta_unitSolidAng * self.SPTfieldSize[fieldidx] * (np.pi/180)**2

        # zeta[z,M]
        zeta_m = self.mass2zeta(self.HMF['M_arr'], self.HMF['z_arr'])

        # Apply field scaling factor
        zeta_m*= self.SPTfieldCorrection[fieldidx]

        # dN/dxi = dN/dlnzeta dlnzeta/dxi (unconvolved)
        # Unfortunately, the zeta_m table is not regular
        # and repeated spline interp is way too slow (1.6sec per field)
        # So we do linear interpolation (in ln(M), and for ln(dN/dlnzeta))
        dN_dxi = self.dlnzeta_dxi_arr\
            * np.exp(np.array([np.interp(self.ln_zeta_xi_arr, np.log(zeta_m[i]), np.log(dN_dlnzeta[i]))
            for i in range(self.HMF['len_z'])]))

        # Convolve with unit scatter (measurement uncertainty)
        dN_dxi = scipy.ndimage.gaussian_filter1d(dN_dxi, 1/self.dxi, axis=1, mode='constant')

        # Set up interpolation for cluster list below
        lndNdxi = interpolate.interp2d(np.log(self.xi_bins), np.log(self.HMF['z_arr'][1:]), np.log(dN_dxi[1:,:]), kind='cubic')

        # Ntotal
        dNdz = np.array([np.sum(np.exp(
            .5*(lndNdxi(np.log(self.xi_arr[1:]), np.log(z)) + lndNdxi(np.log(self.xi_arr[:-1]), np.log(z))))\
            * (self.xi_arr[1:]-self.xi_arr[:-1])) for z in self.z_arr])
        Ntotal = np.trapz(dNdz, self.z_arr)

        # Likelihood contribution from Ntotal
        this_lnlike = -Ntotal

        ##### confirmed clusters
        thisfield_conf = np.where((self.catalog['field']==self.SPTfieldNames[fieldidx])
            & (self.catalog['xi']>=self.surveyCutSZ[0]) & (self.catalog['xi']<=self.surveyCutSZ[1])
            & (self.catalog['redshift']>=self.surveyCutRedshift[0]) & (self.catalog['redshift']<=self.surveyCutRedshift[1]))[0]
        for i in thisfield_conf:
            # spec-z
            if self.catalog['redshift_err'][i]==0.:
                this_lnlike+= lndNdxi(np.log(self.catalog['xi'][i]), np.log(self.catalog['redshift'][i]))[0]
            # photo-z
            elif self.catalog['redshift_err'][i]>0.:
                zlo = min((.25, self.catalog['redshift'][i]-4*self.catalog['redshift_err'][i]))
                zhi = max((self.HMF['z_arr'][-1], self.catalog['redshift'][i]+4*self.catalog['redshift_err'][i]))
                zarr = np.linspace(zlo, zhi, 15)
                integrand = np.exp(lndNdxi(np.log(self.catalog['xi'][i]), np.log(zarr))[:,0]) * norm.pdf(zarr, self.catalog['redshift'][i], self.catalog['redshift_err'][i])
                this_lnlike+= np.log(np.trapz(integrand, zarr))

        ##### unconfirmed candidates
        thisfield_unconf = np.where((self.catalog['field']==self.SPTfieldNames[fieldidx])
            & (self.catalog['xi']>=self.surveyCutSZ[0]) & (self.catalog['xi']<=self.surveyCutSZ[1])
            & (self.catalog['redshift']==0.) & (self.catalog['redshift_lim']<=self.surveyCutRedshift[1]))[0]
        for i in thisfield_unconf:
            dNdxifalse = self.SPTnFalse_beta[fieldidx] * self.SPTfieldSize[fieldidx]/2500 * self.SPTnFalse_alpha[fieldidx]\
                * np.exp(-self.SPTnFalse_beta[fieldidx]*(self.catalog['xi'][i]-5.))
            zarr = np.linspace(self.catalog['redshift_lim'][i], self.HMF['z_arr'][-1], 25)
            this_lnlike+= np.log(dNdxifalse + np.trapz(np.exp(lndNdxi(np.log(self.catalog['xi'][i]), np.log(zarr))[:,0]), zarr))

        return this_lnlike, Ntotal


    ########## Utility functions

    def dlnzeta_dxi(self, xi):
        return xi/(xi**2 - 3)

    def xi2zeta(self, xi):
        return (xi**2 - 3)**.5

    def mass2zeta(self, mass, z):
        # [redshift][mass]
        massterm = (mass/self.SZmPivot)**self.Bsz
        zterm = (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.Csz
        return self.Asz * massterm[None,:] * zterm[:,None]
