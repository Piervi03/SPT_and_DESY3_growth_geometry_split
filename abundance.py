from __future__ import division
import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage import gaussian_filter1d
import scaling_relations


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return NumberCount.lnlike_field(*arg)

################################################################################

class NumberCount:

    def __init__(self, catalog, SPT_survey,
                 surveyCutSZmax, surveyCutRedshift,
                 NPROC):
        self.catalog = catalog
        self.SPT_survey = SPT_survey
        self.surveyCutSZmax = surveyCutSZmax
        self.surveyCutRedshift = surveyCutRedshift
        self.NPROC = NPROC

        ##### Observable arrays
        # Arrays over which we'll integrate (survey cuts applied)
        xi_min = np.amin(self.SPT_survey['XI_MIN'])
        Nxi = int(np.log10(self.surveyCutSZmax/xi_min)/.005 + 1)
        self.xi_arr = np.logspace(np.log10(xi_min), np.log10(self.surveyCutSZmax), Nxi)
        dz = .01
        Nz = int((self.surveyCutRedshift[1]-self.surveyCutRedshift[0])/dz + 1)
        self.z_arr = np.linspace(self.surveyCutRedshift[0], self.surveyCutRedshift[1], Nz)
        # For output
        dz = .1
        Nz = int((self.surveyCutRedshift[1]-self.surveyCutRedshift[0])/dz + 1)
        self.z_bins_output = np.linspace(self.surveyCutRedshift[0], self.surveyCutRedshift[1], Nz)
        self.xi_bins_output = np.logspace(np.log10(4.25), np.log10(self.surveyCutSZmax), 11)
        self.xi_bins_survey = {'SPTPOL_500d': self.xi_bins_output,
                               'SZ': np.logspace(np.log10(4.5), np.log10(self.surveyCutSZmax), 11),
                               'SPECS': np.logspace(np.log10(5), np.log10(self.surveyCutSZmax), 11)}


    def lnlike(self, HMF, cosmology, scaling):
        """Return ln-likelihood for SPT cluster abundance."""
        self.HMF = HMF
        self.cosmology = cosmology
        self.scaling = scaling

        # Lin spaced array in xi for convo with unit scatter (+3 sigma margin)
        xi_min = scaling_relations.zeta2xi(self.scaling['zeta_min'])
        Nxi = int((self.surveyCutSZmax+3 - xi_min)/.1 + 1)
        self.xi_bins = np.linspace(xi_min, self.surveyCutSZmax+3, Nxi)
        self.dxi = self.xi_bins[1] - self.xi_bins[0]
        self.ln_zeta_xi_arr = np.log(scaling_relations.xi2zeta(self.xi_bins))
        self.dlnzeta_dxi_arr = scaling_relations.dlnzeta_dxi_given_xi(self.xi_bins)

        self.dN_dlnzeta_unitSolidAng = {}
        for tmp in ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep', 'SZ']:
            self.dN_dlnzeta_unitSolidAng[tmp] = scaling_relations.dlnM_dlnobs('zeta', self.scaling) * np.exp(self.HMF['%s_dNdlnM'%tmp])

        # zeta[z,M]
        self.lnzeta_m = scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling, self.cosmology)

        ##### Evaluate (log)-likelihood for each SPT field (optional multiprocessing)
        num_fields = len(self.SPT_survey)
        if self.NPROC==0:
            field_results = [self.lnlike_field(fieldidx) for fieldidx in range(num_fields)]
        else:
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*num_fields, range(num_fields))
                field_results = pool.map(unwrap_self_f, argin)
        lnlike = np.sum([field_results[i][0] for i in range(num_fields)])
        Ntotal = np.sum([field_results[i][1] for i in range(num_fields)])
        dN_dz = np.array([field_results[i][2] for i in range(num_fields)]).sum(axis=0)
        dN_dxi = np.array([field_results[i][3] for i in range(num_fields)]).sum(axis=0)
        # dN_dxi_survey = np.array([field_results[i][4] for i in range(num_fields)])
        # subsurveys = np.array([field_results[i][5] for i in range(num_fields)])
        # dN_dz_500d = dN_dz[subsurveys=='SPTPOL_500d',:].sum(axis=0)
        # dN_dz_SZ = dN_dz[subsurveys=='SZ',:].sum(axis=0)
        # dN_dz_SPECS = dN_dz[subsurveys=='SPECS',:].sum(axis=0)
        # dN_dxi_500d = dN_dxi_survey[subsurveys=='SPTPOL_500d',:].sum(axis=0)
        # dN_dxi_SZ = dN_dxi_survey[subsurveys=='SZ',:].sum(axis=0)
        # dN_dxi_SPECS = dN_dxi_survey[subsurveys=='SPECS',:].sum(axis=0)

        # print 'abundance lnlike %.3f, Ntotal %.2f'%(lnlike, Ntotal)

        return lnlike, dN_dz, dN_dxi, Ntotal

    ##########

    def lnlike_field(self, fieldidx):
        """Returns (ln-likelihood, Ntotal) for a given SPT field (index)."""
        # dN/dln(zeta)
        if self.SPT_survey['FIELD'][fieldidx]=='sptpol_500d_MCMF':
            tmp = 'SZ_lambdacut_deep'
        elif '_MCMF' in self.SPT_survey['FIELD'][fieldidx]:
            tmp = 'SZ_lambdacut_shallow'
        else:
            tmp = 'SZ'
        dN_dlnzeta = self.dN_dlnzeta_unitSolidAng[tmp] * self.SPT_survey['AREA'][fieldidx] * (np.pi/180)**2
        with np.errstate(divide='ignore'):
            lndN_dlnzeta = np.log(dN_dlnzeta)

        # Apply field scaling factor
        this_lnzeta_m = self.lnzeta_m + np.log(self.SPT_survey['GAMMA'][fieldidx])
        if '_sptpol' in self.SPT_survey['FIELD'][fieldidx]:
            this_lnzeta_m+= np.log(self.scaling['SPECS_calib'])

        # dN/dxi = dN/dlnzeta dlnzeta/dxi (unconvolved)
        # Unfortunately, the zeta_m table is not regular
        # and repeated spline interp is way too slow (1.6sec per field)
        # So we do linear interpolation (in ln(M), and for ln(dN/dlnzeta))
        dN_dxi = (self.dlnzeta_dxi_arr
                  * np.exp(np.array([np.interp(self.ln_zeta_xi_arr, this_lnzeta_m[i], lndN_dlnzeta[i])
                                     for i in range(self.HMF['len_z'])])))

        # Convolve with unit scatter (measurement uncertainty)
        dN_dxi = gaussian_filter1d(dN_dxi, 1/self.dxi, axis=1, mode='constant')

        # Set up interpolation for cluster list below
        with np.errstate(divide='ignore'):
            lndN_dxi = np.log(dN_dxi)
        lndNdxi = RectBivariateSpline(np.log(self.HMF['z_arr']), np.log(self.xi_bins), lndN_dxi)

        # Ntotal (trapz except that we sum in log-space)
        Nxi = int(np.log10(self.surveyCutSZmax/self.SPT_survey['XI_MIN'][fieldidx])/.005 + 1)
        self.xi_arr = np.logspace(np.log10(self.SPT_survey['XI_MIN'][fieldidx]), np.log10(self.surveyCutSZmax), Nxi)
        integrand = (np.exp(.5*(lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[1:])) + lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[:-1]))))
                     * (self.xi_arr[1:]-self.xi_arr[:-1]))
        dNdz = np.sum(integrand, axis=1)
        Ntotal = np.trapz(dNdz, self.z_arr)

        # dN_dxi and dN_dz for output
        dNdz_interp = interp1d(self.z_arr, dNdz, kind='cubic')
        dN_dz_out = dNdz_interp(self.z_bins_output)
        integrand = np.exp(lndNdxi(np.log(self.z_arr), np.log(self.xi_bins_output)))
        dN_dxi_out = np.trapz(integrand, self.z_arr, axis=0)
        integrand = np.exp(lndNdxi(np.log(self.z_arr), np.linspace(np.log(self.SPT_survey['XI_MIN'][fieldidx]), np.log(50), 11)))
        dN_dxi_out_survey = np.trapz(integrand, self.z_arr, axis=0)

        # Likelihood contribution from Ntotal
        lnlike_this_field = -Ntotal

        ##### confirmed clusters
        thisfield_conf = np.nonzero((self.catalog['FIELD']==self.SPT_survey['FIELD'][fieldidx])
            & (self.catalog['COSMO_SAMPLE']==1)
            & (self.catalog['XI']>=self.SPT_survey['XI_MIN'][fieldidx]) & (self.catalog['XI']<=self.surveyCutSZmax)
            & (self.catalog['REDSHIFT']>=self.surveyCutRedshift[0]) & (self.catalog['REDSHIFT']<=self.surveyCutRedshift[1]))[0]
        for i in thisfield_conf:
            # spec-z: Evaluate dN/dxi/dz at exact location
            if self.catalog['REDSHIFT_UNC'][i]==0.:
                this_lnlike = lndNdxi(np.log(self.catalog['REDSHIFT'][i]), np.log(self.catalog['XI'][i]))[0,0]
                lnlike_this_field+= this_lnlike
            # photo-z: \int dz dN/dxi/dz, choose limits to encompass +/- 4 sigma of photo-z error
            elif self.catalog['REDSHIFT_UNC'][i]>0.:
                zlo = min((.25, self.catalog['REDSHIFT'][i]-4*self.catalog['REDSHIFT_UNC'][i]))
                zhi = max((self.HMF['z_arr'][-1], self.catalog['REDSHIFT'][i]+4*self.catalog['REDSHIFT_UNC'][i]))
                zarr = np.linspace(zlo, zhi, 15)
                integrand = np.exp(lndNdxi(np.log(zarr), np.log(self.catalog['XI'][i])))[:,0] * norm.pdf(zarr, self.catalog['REDSHIFT'][i], self.catalog['REDSHIFT_UNC'][i])
                this_lnlike = np.log(np.trapz(integrand, zarr))
                lnlike_this_field+= this_lnlike

        return lnlike_this_field, Ntotal, dN_dz_out, dN_dxi_out, dN_dxi_out_survey, self.SPT_survey['FIELD'][fieldidx]
