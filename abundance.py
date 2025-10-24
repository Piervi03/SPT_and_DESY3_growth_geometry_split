import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d
import scaling_relations


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return NumberCount.lnlike_field(*arg)

################################################################################


class NumberCount:

    def __init__(self, catalog, SPT_survey,
                 surveyCutSZmax, z_cl_min_max,
                 NPROC):
        self.catalog = catalog
        self.SPT_survey = SPT_survey
        self.surveyCutSZmax = surveyCutSZmax
        self.z_cl_min_max = z_cl_min_max
        self.NPROC = NPROC

        # Observable arrays
        # Arrays over which we'll integrate (survey cuts applied)
        dz = .01
        Nz = int((self.z_cl_min_max[1]-self.z_cl_min_max[0])/dz + 1)
        self.z_arr = np.linspace(self.z_cl_min_max[0], self.z_cl_min_max[1], Nz)
        # For output
        dz = .1
        Nz = int((self.z_cl_min_max[1]-self.z_cl_min_max[0])/dz + 1)
        self.z_bins_output = np.linspace(self.z_cl_min_max[0], self.z_cl_min_max[1], Nz)
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

        # Evaluate (log)-likelihood for each SPT field (optional multiprocessing)
        num_fields = len(self.SPT_survey)
        if self.NPROC == 0:
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
        # dN/dxi
        all_lndNdxi = np.zeros(len(self.catalog))
        for i in range(num_fields):
            all_lndNdxi[field_results[i][6]] = field_results[i][7]

        # print('abundance lnlike %.3f, Ntotal %.2f'%(lnlike, Ntotal))

        return lnlike, dN_dz, dN_dxi, Ntotal, all_lndNdxi

    ##########

    def lnlike_field(self, fieldidx):
        """Returns (ln-likelihood, Ntotal) for a given SPT field (index)."""
        # dN/dln(zeta)
        if self.SPT_survey['LAMBDA_MIN'][fieldidx] not in ['none', 'None', 'NONE']:
            tmp = 'SZ_lambdacut_' + self.SPT_survey['LAMBDA_MIN'][fieldidx]
        else:
            tmp = 'SZ'
        lndN_dlnzeta = (self.HMF['%s_lndNdlnM' % tmp]
                        + np.log(scaling_relations.dlnM_dlnobs('zeta', self.scaling)
                                 * self.SPT_survey['AREA'][fieldidx] * (np.pi/180)**2))

        # Scaling relation (depends on survey)
        lnzeta_m = scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None, :], self.HMF['z_arr'][:, None],
                                                  self.scaling, self.cosmology, SPTfield=self.SPT_survey[fieldidx])

        # dN/dxi = dN/dlnzeta dlnzeta/dxi (unconvolved)
        # Unfortunately, the zeta_m table is not regular
        # and repeated spline interp is way too slow (1.6sec per field)
        # So we do linear interpolation (in ln(M), and for ln(dN/dlnzeta))
        dN_dxi = (self.dlnzeta_dxi_arr
                  * np.exp(np.array([np.interp(self.ln_zeta_xi_arr, lnzeta_m[i], lndN_dlnzeta[i])
                                     for i in range(len(self.HMF['z_arr']))])))

        # Convolve with unit scatter (measurement uncertainty)
        dN_dxi = gaussian_filter1d(dN_dxi, 1/self.dxi, axis=1, mode='constant')

        # Set up interpolation for cluster list below
        with np.errstate(divide='ignore'):
            lndN_dxi = np.log(dN_dxi)
        lndNdxi = RectBivariateSpline(np.log(self.HMF['z_arr']), np.log(self.xi_bins), lndN_dxi, kx=2, ky=2)

        # dN/dz = int dlnxi d2N/dz/dlnxi (trapezoid in ln-space because it's essentially a power law)
        lnxi_arr = np.arange(np.log(self.SPT_survey['XI_MIN'][fieldidx]),
                             np.log(self.surveyCutSZmax),
                             .005)
        integrand = (np.exp(.5*(lndNdxi(np.log(self.z_arr), lnxi_arr[1:])+lnxi_arr[1:]
                                + lndNdxi(np.log(self.z_arr), lnxi_arr[:-1])+lnxi_arr[:-1]))
                     * (lnxi_arr[1:]-lnxi_arr[:-1]))
        dNdz = np.sum(integrand, axis=1)
        # N = int dz dN/dz (approximately quadratic in z)
        Ntotal = simpson(dNdz, self.z_arr)

        # dN_dxi and N(z) for output
        f = InterpolatedUnivariateSpline(self.z_arr, dNdz)
        N_z_out = np.array([f.integral(self.z_bins_output[i], self.z_bins_output[i+1]) for i in range(len(self.z_bins_output)-1)])
        f = RectBivariateSpline(self.HMF['z_arr'], self.xi_bins, dN_dxi)
        N_xi_out = np.array([f.integral(self.z_arr[0], self.z_arr[-1], self.xi_bins_output[i], self.xi_bins_output[i+1])
                             for i in range(len(self.xi_bins_output)-1)])
        if self.SPT_survey['XI_MIN'][fieldidx] != self.xi_bins_output[0]:
            N_xi_out[0] = f.integral(self.z_arr[0], self.z_arr[-1], self.SPT_survey['XI_MIN'][fieldidx], self.xi_bins_output[1])
        # integrand = np.exp(lndNdxi(np.log(self.z_arr), np.linspace(np.log(self.SPT_survey['XI_MIN'][fieldidx]), np.log(50), 11)))
        dN_dxi_out_survey = None  # np.trapezoid(integrand, self.z_arr, axis=0)

        # Likelihood contribution from Ntotal
        lnlike_this_field = -Ntotal

        # confirmed clusters
        thisfield_conf = ((self.catalog['FIELD'] == self.SPT_survey['FIELD'][fieldidx])
                          & (self.catalog['COSMO_SAMPLE'] == 1)
                          & (self.catalog['XI'] >= self.SPT_survey['XI_MIN'][fieldidx])
                          & (self.catalog['XI'] <= self.surveyCutSZmax)
                          & (self.catalog['REDSHIFT'] >= self.z_cl_min_max[0])
                          & (self.catalog['REDSHIFT'] <= self.z_cl_min_max[1]))
        these_lndNdxi = (lndNdxi(np.log(self.catalog['REDSHIFT'][thisfield_conf]), np.log(self.catalog['XI'][thisfield_conf]), grid=False)
                         - np.log(self.SPT_survey['AREA'][fieldidx] * (np.pi/180.)**2.))
        for n, i in enumerate(thisfield_conf.nonzero()[0]):
            # spec-z: Evaluate dN/dxi/dz at exact location
            if self.catalog['REDSHIFT_UNC'][i] == 0.:
                lnlike_this_field += these_lndNdxi[n]
            # photo-z: \int dz dN/dxi/dz, choose limits to encompass +/- 4 sigma of photo-z error
            elif self.catalog['REDSHIFT_UNC'][i] > 0.:
                zlo = min((.25, self.catalog['REDSHIFT'][i]-4*self.catalog['REDSHIFT_UNC'][i]))
                zhi = max((self.HMF['z_arr'][-1], self.catalog['REDSHIFT'][i]+4*self.catalog['REDSHIFT_UNC'][i]))
                zarr = np.linspace(zlo, zhi, 15)
                integrand = np.exp(lndNdxi(np.log(zarr), np.log(self.catalog['XI'][i])))[:, 0] * norm.pdf(zarr, self.catalog['REDSHIFT'][i], self.catalog['REDSHIFT_UNC'][i])
                this_lnlike = np.log(np.trapezoid(integrand, zarr))
                lnlike_this_field += this_lnlike

        return lnlike_this_field, Ntotal, N_z_out, N_xi_out, dN_dxi_out_survey, self.SPT_survey['FIELD'][fieldidx], thisfield_conf, these_lndNdxi
