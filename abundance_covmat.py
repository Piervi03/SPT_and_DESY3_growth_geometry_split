import numpy as np
from multiprocessing import Pool
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage import gaussian_filter1d
import scaling_relations


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return NumberCount.process_field(*arg)

################################################################################

class NumberCount:

    def __init__(self, **kwargs):
        catalog = kwargs.pop('catalog')
        self.SPT_survey = kwargs.pop('SPT_survey')
        self.NPROC = kwargs.pop('NPROC', 0)
        self.covmat_sv = kwargs.pop('covmat_sv')
        z_cl_min_max = kwargs.pop('z_cl_min_max')

        SPT_survey = kwargs.pop('SPT_survey')
        # Arrays over which we'll integrate (survey cuts applied)
        xi_min = np.amin(self.SPT_survey['XI_MIN'])
        Nxi = int(np.log10(self.surveyCutSZmax/xi_min)/.005 + 1)
        self.xi_arr = np.logspace(np.log10(xi_min), np.log10(self.surveyCutSZmax), Nxi)
        dz = .01
        Nz = int((z_cl_min_max[1]-z_cl_min_max[0])/dz + 1)
        self.z_arr = np.linspace(z_cl_min_max[0], z_cl_min_max[1], Nz)
        # For output
        dz = .1
        Nz = int((z_cl_min_max[1]-z_cl_min_max[0])/dz + 1)
        self.z_bins_output = np.linspace(z_cl_min_max[0], z_cl_min_max[1], Nz)
        # self.xi_bins_output = np.logspace(np.log10(4.25), np.log10(self.surveyCutSZmax), 11)
        # self.xi_bins_survey = {'SPTPOL_500d': self.xi_bins_output,
        #                        'SZ': np.logspace(np.log10(4.5), np.log10(self.surveyCutSZmax), 11),
        #                        'SPECS': np.logspace(np.log10(5), np.log10(self.surveyCutSZmax), 11)}
        self.obs_bins_z = np.array([.25, .5, .95, 1.79])
        self.obs_bins_xi = np.array([4.25, 5.6, 50.])
        self.obs_bin_shape = [len(self.obs_bins_z)-1, len(self.obs_bins_xi)-1]
        self.binned_cat,_,_ = np.histogram2d(catalog['REDSHIFT'], catalog['XI'],
                                             bins=(self.obs_bins_z, self.obs_bins_xi))
        # self.binned_cat,_,_ = np.histogram2d(catalog['REDSHIFT'], np.sqrt(catalog['XI']**2-3)/catalog['GAMMA_FIELD'],
        #                                      bins=(self.z_bins_4, self.snr_bins_3))


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

        ##### Evaluate (log)-likelihood for each SPT field (optional multiprocessing)
        num_fields = len(self.SPT_survey)
        if self.NPROC==0:
            field_results = [self.process_field(fieldidx) for fieldidx in range(num_fields)]
        else:
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*num_fields, range(num_fields))
                field_results = pool.map(unwrap_self_f, argin)
        N_bins = np.array([field_results[i][0] for i in range(num_fields)])
        N_total = np.sum(N_bins)
        dN_dz = np.array([field_results[i][1] for i in range(num_fields)]).sum(axis=0)
        dN_dxi = np.array([field_results[i][2] for i in range(num_fields)]).sum(axis=0)
        # dN_dxi_survey = np.array([field_results[i][3] for i in range(num_fields)])
        # subsurveys = np.array([field_results[i][4] for i in range(num_fields)])
        # dN_dz_500d = dN_dz[subsurveys=='SPTPOL_500d',:].sum(axis=0)
        # dN_dz_SZ = dN_dz[subsurveys=='SZ',:].sum(axis=0)
        # dN_dz_SPECS = dN_dz[subsurveys=='SPECS',:].sum(axis=0)
        # dN_dxi_500d = dN_dxi_survey[subsurveys=='SPTPOL_500d',:].sum(axis=0)
        # dN_dxi_SZ = dN_dxi_survey[subsurveys=='SZ',:].sum(axis=0)
        # dN_dxi_SPECS = dN_dxi_survey[subsurveys=='SPECS',:].sum(axis=0)

        # Likelihood
        model = np.sum(N_bins, axis=0).flatten()
        diff = model-self.binned_cat.flatten()
        covmat = self.covmat_sv + np.diag(model)
        chi2 = np.dot(diff, np.linalg.solve(covmat, diff))
        lnlike = -.5*np.log(np.linalg.det(2*np.pi*covmat)) - .5*chi2

        return lnlike, dN_dz, dN_dxi, N_total, None


    ##########

    def process_field(self, fieldidx):
        """Returns (ln-likelihood, Ntotal) for a given SPT field (index)."""
        # dN/dln(zeta)
        if self.SPT_survey['LAMBDA_MIN'][fieldidx] not in ['none', 'None', 'NONE']:
            tmp = 'SZ_lambdacut_' + self.SPT_survey['LAMBDA_MIN'][fieldidx]
        else:
            tmp = 'SZ'
        lndN_dlnzeta = (self.HMF['%s_dNdlnM'%tmp]
                        + np.log(scaling_relations.dlnM_dlnobs('zeta', self.scaling)
                                 * self.SPT_survey['AREA'][fieldidx] * (np.pi/180)**2))

        # Scaling relation (depends on survey)
        lnzeta_m = scaling_relations.lnmass2lnobs('zeta', self.HMF['lnM_arr'][None,:], self.HMF['z_arr'][:,None], self.scaling, self.cosmology, SPTfield=self.SPT_survey[fieldidx])

        # dN/dxi = dN/dlnzeta dlnzeta/dxi (unconvolved)
        # Unfortunately, the zeta_m table is not regular
        # and repeated spline interp is way too slow (1.6sec per field)
        # So we do linear interpolation (in ln(M), and for ln(dN/dlnzeta))
        dN_dxi = (self.dlnzeta_dxi_arr
                  * np.exp(np.array([np.interp(self.ln_zeta_xi_arr, lnzeta_m[i], lndN_dlnzeta[i])
                                     for i in range(len(self.HMF['z_arr']))])))

        # Convolve with unit scatter (measurement uncertainty)
        dN_dxi = gaussian_filter1d(dN_dxi, 1/self.dxi, axis=1, mode='constant')

        # Set up interpolation
        with np.errstate(divide='ignore'):
            lndN_dxi = np.log(dN_dxi)
        lndNdxi = RectBivariateSpline(np.log(self.HMF['z_arr']), np.log(self.xi_bins), lndN_dxi)

        # Ntotal (trapezoid except that we sum in log-space)
        Nxi = int(np.log10(self.surveyCutSZmax/self.SPT_survey['XI_MIN'][fieldidx])/.005 + 1)
        self.xi_arr = np.logspace(np.log10(self.SPT_survey['XI_MIN'][fieldidx]), np.log10(self.surveyCutSZmax), Nxi)
        integrand = (np.exp(.5*(lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[1:])) + lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[:-1]))))
                     * (self.xi_arr[1:]-self.xi_arr[:-1]))
        dNdz = np.sum(integrand, axis=1)

        # dN_dxi and dN_dz for output
        dNdz_interp = interp1d(self.z_arr, dNdz, kind='cubic', assume_sorted=True)
        dN_dz_out = dNdz_interp(self.z_bins_output)
        integrand = np.exp(lndNdxi(np.log(self.z_arr), np.log(self.xi_bins_output)))
        dN_dxi_out = np.trapezoid(integrand, self.z_arr, axis=0)
        integrand = np.exp(lndNdxi(np.log(self.z_arr), np.linspace(np.log(self.SPT_survey['XI_MIN'][fieldidx]), np.log(50), 11)))
        dN_dxi_out_survey = np.trapezoid(integrand, self.z_arr, axis=0)

        # In bins
        N_bins = np.zeros(self.obs_bin_shape)
        for i in range(self.obs_bin_shape[0]):
            z_arr = np.linspace(self.obs_bins_z[i], self.obs_bins_z[i+1], 50)
            for j in range(self.obs_bin_shape[1]):
                xi_min = self.obs_bins_xi[j]
                if self.obs_bins_xi[j]<self.SPT_survey['XI_MIN'][fieldidx]:
                    xi_min = self.SPT_survey['XI_MIN'][fieldidx]
                xi_max = self.obs_bins_xi[j+1]
                xi_arr = np.logspace(np.log10(xi_min), np.log10(xi_max), 200)
                integrand = np.exp(.5*(lndNdxi(np.log(z_arr), np.log(xi_arr[1:])) + lndNdxi(np.log(z_arr), np.log(xi_arr[:-1])))) * (xi_arr[1:]-xi_arr[:-1])
                N_bins[i,j] = np.trapezoid(np.sum(integrand, axis=1), z_arr)
        return N_bins, dN_dz_out, dN_dxi_out, dN_dxi_out_survey, self.SPT_survey['FIELD'][fieldidx]

        N_bins = np.zeros((4,3))
        for i in range(4):
            z_arr = np.linspace(self.z_bins_4[i], self.z_bins_4[i+1], 50)
            for j in range(3):
                xi_min = np.sqrt((self.SPT_survey['GAMMA'][fieldidx]*self.snr_bins_3[j])**2 +3)
                if xi_min<self.SPT_survey['XI_MIN'][fieldidx]:
                    xi_min = self.SPT_survey['XI_MIN'][fieldidx]
                if j==2:
                    xi_max = self.surveyCutSZmax
                else:
                    xi_max = np.sqrt((self.SPT_survey['GAMMA'][fieldidx]*self.snr_bins_3[j+1])**2 +3)
                if xi_min>xi_max:
                    continue
                xi_arr = np.logspace(np.log10(xi_min), np.log10(xi_max), 200)
                integrand = np.exp(.5*(lndNdxi(np.log(z_arr), np.log(xi_arr[1:])) + lndNdxi(np.log(z_arr), np.log(xi_arr[:-1])))) * (xi_arr[1:]-xi_arr[:-1])
                N_bins[i,j] = np.trapezoid(np.sum(integrand, axis=1), z_arr)

        return N_bins, dN_dz_out, dN_dxi_out, dN_dxi_out_survey, self.SPT_survey['FIELD'][fieldidx]
