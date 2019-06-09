from __future__ import division
import numpy as np
from multiprocessing import Pool
import scipy.ndimage
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline

import cosmo
import scaling_relations

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return NumberCount.lnlike_field(*arg)

################################################################################
class NumberCount:

    def __init__(self, catalog, SPT_survey, scaling,
                 surveyCutSZ, surveyCutRedshift, NPROC):
        self.catalog = catalog
        self.SPT_survey = SPT_survey
        self.scaling = scaling
        self.surveyCutSZ = surveyCutSZ
        self.surveyCutLambda = surveyCutLambda
        self.surveyCutRedshift = surveyCutRedshift
        self.NPROC = NPROC

        ##### Observable arrays
        # Lin spaced for convo with unit scatter (+3 sigma margin)
        Nxi = int((self.surveyCutSZ[1]+3 - 2.7)/.1 + 1)
        self.xi_bins = np.linspace(2.7, self.surveyCutSZ[1]+3, Nxi)
        self.dxi = self.xi_bins[1] - self.xi_bins[0]
        # ln(zeta(xi_bins))
        self.ln_zeta_xi_arr = np.log(scaling_relations.xi2zeta(self.xi_bins))
        # dlnzeta/dxi (xi_bins)
        self.dlnzeta_dxi_arr = scaling_relations.dlnzeta_dxi(self.xi_bins)
        # Arrays over which we'll integrate (survey cuts applied)
        Nlambda = int(np.log10(self.surveyCutLambda[1]/self.surveyCutLambda[0])/.1 + 1)
        self.lambda_arr = np.logspace(np.log10(self.surveyCutLambda[0]), np.log10(self.surveyCutLambda[1]), Nlambda)
        self.ln_lambda_arr = np.log(self.lambda_arr)
        Nxi = int(np.log10(self.surveyCutSZ[1]/self.surveyCutSZ[0])/.005 + 1)
        self.xi_arr = np.logspace(np.log10(self.surveyCutSZ[0]), np.log10(self.surveyCutSZ[1]), Nxi)
        dz = .01
        Nz = int((self.surveyCutRedshift[1]-self.surveyCutRedshift[0])/dz + 1)
        self.z_arr = np.linspace(self.surveyCutRedshift[0], self.surveyCutRedshift[1], Nz)



    def lnlike(self):
        """Return ln-likelihood for SPT cluster abundance."""
        ##### Convert HMF to dN/dln(zeta)/dln(lambda) = dN/dlog10(M) * dlog10(M)/dln(zeta) * dlog10(M)/dln(lambda)
        dlnM_dlnlambda_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling) * scaling_relations.dlnM_dlnobs('richness', self.scaling)
        self.dN_dlnlambda_dlnzeta_unitSolidAng = self.HMF_zetalambda['dN_dlnM'] * dlnM_dlnlambda_dlnzeta

        ##### Evaluate (log)-likelihood for each SPT field (optional multiprocessing)
        num_fields = len(self.SPT_survey)
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

        # print 'abundance lnlike %.3f, Ntotal %.2f'%(lnlike, Ntotal)

        return lnlike


    ##########
    def lnlike_field(self, fieldidx):
        """Returns (ln-likelihood, Ntotal) for a given SPT field (index)."""
        # dN/dln(zeta)/dln(lambda)
        dN_dlnlambda_dlnzeta = self.dN_dlnlambda_dlnzeta_unitSolidAng * self.SPT_survey['AREA'][fieldidx] * (np.pi/180)**2
        if np.any(dN_dlnlambda_dlnzeta==0):
            dN_dlnlambda_dlnzeta[(dN_dlnlambda_dlnzeta==0).nonzero()] = np.nextafter(0, 1)

        # obs[z,M]
        zeta_m = scaling_relations.mass2obs('zeta', self.HMF_zetalambda['M_arr'][None,:], self.HMF_zetalambda['z_arr'][:,None], self.scaling, self.cosmology)
        lambda_m = scaling_relations.mass2obs('richness', self.HMF_zetalambda['M_arr'][None,:], self.HMF_zetalambda['z_arr'][:,None], self.scaling, self.cosmology)

        # Apply field scaling factor
        zeta_m*= self.SPT_survey['GAMMA'][fieldidx]
        if self.SPT_survey['SURVEY'][fieldidx]=='SPECS':
            zeta_m*= self.scaling['SPECS_calib']

        # dN/dxi = dN/dlnzeta dlnzeta/dxi (unconvolved)
        # Unfortunately, the zeta_m table is not regular
        # and repeated spline interp is way too slow (1.6sec per field)
        # So we do linear interpolation (in ln(M), and for ln(dN/dlnzeta))
        dN_dlnlambda_dlnzeta_interp = [RectBivariateSpline(np.log(lambda_m[i]), np.log(zeta_m[i]), np.log(dN_dlnlambda_dlnzeta[i]))
                                       for i in range(self.HMF_zetalambda['len_z'])]
        dN_dlnlambda_dlnzeta_grid = np.array([dN_dlnlambda_dlnzeta_interp[i](self.ln_lambda_arr, self.ln_zeta_xi_arr)
                                              for i in range(self.HMF_zetalambda['len_z'])])
        dN_dlambda_dxi = dN_dlnlambda_dlnzeta_grid * self.dlnzeta_dxi_arr / self.lambda_arr

        # Convolve with unit scatter in xi (measurement uncertainty)
        dN_dlambda_dxi = scipy.ndimage.gaussian_filter1d(dN_dlambda_dxi, 1/self.dxi, axis=2, mode='constant')

        # Integrate above lambda selection threshold
        dN_dxi = np.trapz(dN_dlambda_dxi, self.lambda_arr, axis=1)
        if np.any(dN_dxi==0):
            dN_dxi[(dN_dxi==0).nonzero()] = np.nextafter(0, 1)

        # Set up interpolation for cluster list below
        lndNdxi = RectBivariateSpline(np.log(self.HMF_zetalambda['z_arr']), np.log(self.xi_bins), np.log(dN_dxi))

        # Ntotal (trapz except that we sum in log-space)
        integrand = np.exp(.5*(lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[1:])) + lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[:-1]))))\
             * (self.xi_arr[1:]-self.xi_arr[:-1])
        dNdz = np.sum(integrand, axis=1)
        Ntotal = np.trapz(dNdz, self.z_arr)

        # Likelihood contribution from Ntotal
        lnlike_this_field = -Ntotal

        ##### confirmed clusters
        thisfield_conf = np.where((self.catalog['FIELD']==self.SPT_survey['FIELD'][fieldidx])
            & (self.catalog['XI']>=self.surveyCutSZ[0]) & (self.catalog['XI']<=self.surveyCutSZ[1])
            & (self.catalog['REDSHIFT']>=self.surveyCutRedshift[0]) & (self.catalog['REDSHIFT']<=self.surveyCutRedshift[1]))[0]
        for i in thisfield_conf:
            # spec-z: Evaluate dN/dxi/dz at exact location
            if self.catalog['REDSHIFT_UNC'][i]==0.:
                this_lnlike = lndNdxi(np.log(self.catalog['REDSHIFT'][i]), np.log(self.catalog['XI'][i]))[0,0]
                lnlike_this_field+= this_lnlike
            # photo-z: \int dz dN/dxi/dz, choose limits to encompass +/- 4 sigma of photo-z error
            elif self.catalog['REDSHIFT_UNC'][i]>0.:
                zlo = min((.25, self.catalog['REDSHIFT'][i]-4*self.catalog['REDSHIFT_UNC'][i]))
                zhi = max((self.HMF_zetalambda['z_arr'][-1], self.catalog['REDSHIFT'][i]+4*self.catalog['REDSHIFT_UNC'][i]))
                zarr = np.linspace(zlo, zhi, 15)
                integrand = np.exp(lndNdxi(np.log(zarr), np.log(self.catalog['XI'][i])))[:,0] * norm.pdf(zarr, self.catalog['REDSHIFT'][i], self.catalog['REDSHIFT_UNC'][i])
                this_lnlike = np.log(np.trapz(integrand, zarr))
                lnlike_this_field+= this_lnlike

        ##### unconfirmed candidates
        thisfield_unconf = np.where((self.catalog['FIELD']==self.SPT_survey['FIELD'][fieldidx])
            & (self.catalog['XI']>=self.surveyCutSZ[0]) & (self.catalog['XI']<=self.surveyCutSZ[1])
            & (self.catalog['REDSHIFT']==0.) & (self.catalog['REDSHIFT_LIMIT']<=self.surveyCutRedshift[1]))[0]
        for i in thisfield_unconf:
            # If it's a false detection, it's drawn from dN_false/dxi
            dNdxifalse = self.SPT_survey['BETA'][fieldidx] * self.SPT_survey['AREA'][fieldidx]/2500 * self.SPT_survey['ALPHA'][fieldidx]\
                * np.exp(-self.SPT_survey['BETA'][fieldidx]*(self.catalog['XI'][i]-5.))
            # If it's a true, unconfirmed cluster, it's drawn from \int_redshift_lim^inf dz dN/dxi/dz
            zarr = np.linspace(self.catalog['REDSHIFT_LIMIT'][i], self.HMF_zetalambda['z_arr'][-1], 25)
            dNdxitrue = np.trapz(np.exp(lndNdxi(np.log(zarr), np.log(self.catalog['XI'][i])))[:,0], zarr)
            # Either way, it's drawn from one of these
            this_lnlike = np.log(dNdxifalse + dNdxitrue)
            lnlike_this_field+= this_lnlike

        return lnlike_this_field, Ntotal
