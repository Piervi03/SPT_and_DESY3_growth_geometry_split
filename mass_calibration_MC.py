import numpy as np
import warnings
from math import sqrt as msqrt
from multiprocessing import Pool
from astropy.table import Table
from scipy.special import log_ndtr
from scipy.special import ndtr as ndtr_sp
from scipy.special import ndtri
from scipy.interpolate import interp1d, RectBivariateSpline

import Mconversion_concentration
import scaling_relations

cosmologyRef = {'Omega_m': .272, 'Omega_l': .728, 'h': .702, 'w0': -1, 'wa': 0}
GETPULL = False
Ndraw_ini = 2**11
Ndraw = 2**15
ln2pi = np.log(2.*np.pi)

# ndtr can be interpolated accurately and is only needed in finite range (-4, 3)
# (This is not true for log_ndtr and ndtri.)
x = np.logspace(0, np.log10(8), 1024) - 5.
y = ndtr_sp(x)
ndtr_max = y[-1]
ndtr = interp1d(x, y, kind='linear', fill_value=(y[0], y[-1]), bounds_error=False, assume_sorted=True)

# Limits for stack
z_names = ['zlo', 'zhi']
xi_names = ['xilo', 'xihi']
z_lims = [.25, .5, .95]
xi_lims = [4.25, 5.6, 1e10]
lnr_r200c_stack = np.linspace(np.log(.3), np.log(5), 16)

scatter_dict = {'zeta': 'Dsz',
                'richness_base': 'Drichness',
                'richness_ext': 'Drichness_ext',
                'Mgas': 'Dx', 'Yx': 'Dx',
                'WLMegacam': 'DWL_Megacam', 'WLDES': 'one', 'WLHST': 'one'}
rho_dict = {'zeta': 'SZ',
            'richness_base': 'richness',
            'richness_ext': 'richness',
            'Mgas': 'X', 'Yx': 'X',
            'WLDES': 'WL', 'WLHST': 'WL', 'WLMegacam': 'WL'}


# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlnlike(*arg)


################################################################################

class MassCalibration:

    def __init__(self, **kwargs):
        # General setup
        self.todo = kwargs.pop('todo')
        self.NPROC = kwargs.pop('NPROC', 0)
        self.get_stacked_DES = kwargs.pop('get_stacked_DES', False)
        self.mcType = kwargs.pop('mcType')
        self.z_cl_min_max = kwargs.pop('z_cl_min_max')
        self.lambda_min = kwargs.pop('lambda_min')
        self.richness_scatter_model = kwargs.pop('richness_scatter_model')
        # Read input files
        self.SPT_survey = Table.read(kwargs.pop('SPT_survey_fields'), format='ascii.commented_header')
        self.catalog = Table.read(kwargs.pop('SPTcatalogfile'))
        HSTcalibfile = kwargs.pop('HSTcalibfile', 'None')
        if HSTcalibfile != 'None':
            self.HSTcalib = Table.read(HSTcalibfile, format='ascii.commented_header')
        # Init data structures for lensing stacks
        if self.get_stacked_DES:
            self.catalog['DES_shear_profile_mean'] = [None for i in range(len(self.catalog))]
        # Safety first
        if not kwargs == {}:
            print("Unknown keyword arguments in", kwargs)

    ############################################################################

    def lnlike(self, HMF, cosmology, scaling):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        self.cosmology = cosmology
        self.scaling = scaling
        self.scaling['one'] = 1.
        self.xi_min = scaling_relations.zeta2xi(self.scaling['zeta_min'])

        # Set up interpolation for HMF
        HMF_in = HMF['dNdlnM']
        if np.any(HMF_in == 0):
            HMF_in[HMF_in == 0] = np.nextafter(0, 1)
        self.lnM_arr = HMF['lnM_arr']
        self.HMF_interp = RectBivariateSpline(HMF['z_arr'], self.lnM_arr, np.log(HMF_in), kx=1, ky=1)

        # Initialize mass-concentration relation class (for WL and dispersions)
        if self.todo['veldisp']:
            self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology,
                                                                           setup_interp=True, interp_massdef=500)

        # Evaluate the individual likelihoods
        len_data = len(self.catalog['SPT_ID'])
        if self.NPROC == 0:
            # Iterate through cluster list
            lnlikelihoods = np.zeros(len_data)
            for i in range(len_data):
                lnlikelihoods[i] = self.clusterlnlike(i)
                if not np.isfinite(lnlikelihoods[i]):
                    break
        else:
            # Launch a multiprocessing pool and get the likelihoods
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*len_data, range(len_data))
                lnlikelihoods = pool.map(unwrap_self_f, argin)
        # Check if they are all finite
        if np.any(np.isinf(lnlikelihoods)):
            return -np.inf, None
        lnlike = np.sum(lnlikelihoods)

        # DES stacked shear profile
        if self.get_stacked_DES:
            stack = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # All clusters with DES lensing data
                notNone = [this is not None for this in self.catalog['DES_shear_profile_mean']]
                for i in range(len(z_lims)-1):
                    for j in range(len(xi_lims)-1):
                        idx = (notNone
                               & (self.catalog['REDSHIFT'] >= z_lims[i]) & (self.catalog['REDSHIFT'] < z_lims[i+1])
                               & (self.catalog['XI'] >= xi_lims[j]) & (self.catalog['XI'] < xi_lims[j+1]))
                        stack['shear_%s%s' % (z_names[i], xi_names[j])] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
            return lnlike, stack
        else:
            return lnlike, None

    ############################################################################

    def clusterlnlike(self, i):
        """Return multi-wavelength mass-calibration ln-likelihood for a
        given cluster (index) by calling `get_lnP_obs_given_xi` or
        returning 0 if no follow-up data is available."""
        # t0 = time.time()
        # Do we actually want this guy?
        if self.catalog['COSMO_SAMPLE'][i] == 0:
            return 0.
        if (self.catalog['XI'][i] < self.SPT_survey['XI_MIN'][self.SPT_survey['FIELD'] == self.catalog['FIELD'][i]]):
            return 0.
        if not (self.z_cl_min_max[0] < self.catalog['REDSHIFT'][i] < self.z_cl_min_max[1]):
            return 0.

        # Check if follow-up is available
        nobs = 0
        obsnames = ['zeta',]
        if self.todo['WL'] and self.catalog['WLdata'][i] is not None:
            nobs += 1
            if self.catalog['WLdata'][i]['datatype'] == 'Megacam':
                obsnames.append('WLMegacam')
            elif self.catalog['WLdata'][i]['datatype'] == 'DES':
                obsnames.append('WLDES')
            elif self.catalog['WLdata'][i]['datatype'] == 'HST':
                obsnames.append('WLHST')
        if self.todo['veldisp'] and self.catalog['veldisp'][i] != 0.:
            nobs += 1
            obsnames.append('disp')
        if self.todo['Yx'] and self.catalog['Mg_fid'][i] != 0:
            nobs += 1
            obsnames.append('Yx')
        if self.todo['Mgas'] and self.catalog['Mg_fid'][i] != 0:
            nobs += 1
            obsnames.append('Mgas')
        if self.todo['richness'] and self.catalog['richness'][i] > 0.:
            nobs += 1
            if self.catalog['REDSHIFT'][i] < self.scaling['z_DESWISE']:
                obsnames.append('richness_base')
            else:
                obsnames.append('richness_ext')
        if nobs == 0:
            return 0.

        # Set up random number generator and get likelihood
        seed = np.abs(int(123456.*(i+1)*np.prod([self.scaling[key]+1. for key in ['Asz', 'Bsz', 'Csz', 'Dsz']])))
        self.rng = np.random.default_rng(seed)
        lnprobability = self.get_lnP_obs_given_xi(obsnames, i)

        if np.isnan(lnprobability) | np.isinf(lnprobability):
            return -np.inf

        # print(self.catalog['SPT_ID'][i], obsnames, lnprobability,)# time.time()-t0)
        return lnprobability

    ############################################################################

    def get_lnP_obs_given_xi(self, obsnames, dataID):
        """Return lnP(obs|xi,z,p) = ln(P(obs,xi|z,p)/P(xi|z,p)) for `obsnames`
        and cluster number `dataID`."""
        # ln(dN/dxi)
        if 'lndNdxi' in self.catalog.colnames:
            lndNdxi = self.catalog['lndNdxi'][dataID]
        else:
            # Which richness (if any)?
            richness_obs = None
            if self.todo['lambda_min']:
                if 'richness_base' in obsnames:
                    richness_obs = 'richness_base'
                else:
                    richness_obs = 'richness_ext'
            lndNdxi = self.get_lndN_dxi(dataID, richness_obs)
        # ln(dN/dobs)
        self.lensingres = None
        dlnM_dlnobs = np.array([scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames])
        covmat = self.get_covmat_obs(obsnames)
        lndNdallobs = self.get_lndN_dallobs(obsnames, covmat, dlnM_dlnobs, dataID)
        return lndNdallobs-lndNdxi

    def get_lndN_dxi(self, dataID, richness_obs=None):
        """Return lnP(xi|z,p) for cluster `dataID`."""
        # Mass given zeta
        zeta, lnM_zeta, xi_lnweights = self.get_lnM_zeta_given_xi(dataID)
        SZscatter_lnM = self.scaling['Dsz'] * scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        lnM, zeta_lnweights = self.draw_lnm_given_lnzeta(lnM_zeta, SZscatter_lnM)
        # Weight with mass function
        mass_lnweights = self.get_mass_function_lnweights(self.catalog['REDSHIFT'][dataID], lnM)
        # Parameter transformation
        trans_lnweights = np.log(scaling_relations.dlnM_dlnobs('zeta', self.scaling) * scaling_relations.dlnzeta_dxi_given_zeta(zeta))
        # Account for lambda_min
        lnP_lambda_gtr_cut = 0.
        if self.todo['lambda_min']:
            lambda_min_type = self.SPT_survey['LAMBDA_MIN'][self.SPT_survey['FIELD'] == self.catalog['FIELD'][dataID]][0]
            if lambda_min_type not in ['none', 'None', 'NONE']:
                this_lambda_min = self.lambda_min[lambda_min_type](self.catalog['REDSHIFT'][dataID])
            else:
                this_lambda_min = 0.
            if this_lambda_min > 0.:
                if self.richness_scatter_model in ['lognormal', 'lognormalrelPoisson']:
                    # Scatter in richness
                    var_richness = self.scaling[scatter_dict[richness_obs]]**2
                    if self.richness_scatter_model == 'lognormalrelPoisson':
                        richness = np.exp(scaling_relations.lnmass2lnobs(richness_obs, lnM, self.catalog['REDSHIFT'][dataID], self.scaling))
                        var_richness += 1/richness
                    # Variance and covariance in mass space
                    var_richness_lnM = var_richness * scaling_relations.dlnM_dlnobs(richness_obs, self.scaling)**2
                    covar_lnM = self.scaling['rhoSZrichness'] * SZscatter_lnM * np.sqrt(var_richness_lnM)
                    # Conditional P(lambda| M, xi)
                    lnmass_lambda_mean = lnM + (covar_lnM/SZscatter_lnM**2) * (lnM_zeta-lnM)
                    lnlambda_mean = scaling_relations.lnmass2lnobs(richness_obs, lnmass_lambda_mean, self.catalog['REDSHIFT'][dataID], self.scaling)
                    lnmass_lambda_std = np.sqrt(var_richness_lnM - covar_lnM**2/SZscatter_lnM**2)
                    lnlambda_std = lnmass_lambda_std/scaling_relations.dlnM_dlnobs(richness_obs, self.scaling)
                    lnP_lambda_gtr_cut = log_ndtr((lnlambda_mean-np.log(this_lambda_min))/lnlambda_std)
        # lnP(xi)
        lnweights = xi_lnweights + zeta_lnweights + mass_lnweights + trans_lnweights + lnP_lambda_gtr_cut
        shift_lnweights = np.amax(lnweights)
        lnPxi = np.log(np.mean(np.exp(lnweights-shift_lnweights))) + shift_lnweights
        return lnPxi

    def get_lnM_zeta_given_xi(self, dataID):
        """Returns mass draws given xi for cluster `dataID`. Prior P(M) is not
        accounted for."""
        zeta, lnweights = self.get_zeta_draws(self.catalog['XI'][dataID])
        lnM = scaling_relations.obs2lnmass('zeta', zeta, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, SPTfield=self.SPT_survey[self.SPT_survey['FIELD'] == self.catalog['FIELD'][dataID]])
        return zeta, lnM, lnweights

    def get_zeta_draws(self, xi):
        """Draw zetas from `xi`. In practice, draw from offset distribution,
        N(xi+offset, 1) so that there are more low-mass samples which will later
        be up-weighted by the mass function. Return zeta and weights."""
        xi_offset = -3/xi**2
        # xi_draw > xi_min, xi_draw > xi-5, xi_draw < xi+4
        r_min = ndtr(self.xi_min-(xi+xi_offset))
        r = r_min + (ndtr_max-r_min)*self.rng.random(Ndraw)
        # Percent point function (scipy stats is too slow)
        xi0 = ndtri(r) + xi+xi_offset
        # Probability of draws and probability of xi
        lnweights = -.5*(xi0-xi)**2 + .5*(xi0-(xi+xi_offset))**2
        zeta = scaling_relations.xi2zeta(xi0)
        return zeta, lnweights

    def draw_lnm_given_lnzeta(self, lnM_zeta, SZscatter_lnM):
        """Return draws of ln(mass) given mass(zeta). Prior P(M) is not
        accounted for."""
        offset = -3*SZscatter_lnM**2
        r_min = ndtr((self.lnM_arr[0]-(lnM_zeta+offset))/SZscatter_lnM)
        r_max = ndtr((self.lnM_arr[-1]-(lnM_zeta+offset))/SZscatter_lnM)
        r = r_min + (r_max-r_min)*self.rng.random(len(lnM_zeta))
        lnM = ndtri(r)*SZscatter_lnM + lnM_zeta+offset
        # Probability of lnM draws and probability of zeta
        lnweights = -.5/SZscatter_lnM**2 * ((lnM-lnM_zeta)**2 - (lnM-(lnM_zeta+offset))**2)
        return lnM, lnweights

    def get_mass_function_lnweights(self, z, lnM):
        """Return log-probability of halo mass function
        ln(P(lnM)) = ln(dN/dlnM) at given `z` and array `lnM`."""
        # RectBivariateSpline wants sorted inputs, but this is faster than `grid=False`
        idx = np.argsort(lnM)
        mass_lnweights = np.zeros(len(lnM))
        mass_lnweights[idx] = self.HMF_interp(z, lnM[idx])
        return mass_lnweights

    def draw_lnzeta_given_lnM(self, lnM, dataID, SZscatter_lnM):
        """Return draws and associated weights from P(zeta|M,z)."""
        zeta_min_lnM = scaling_relations.obs2lnmass('zeta', self.scaling['zeta_min'], self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, SPTfield=self.SPT_survey[self.SPT_survey['FIELD'] == self.catalog['FIELD'][dataID]])
        r_min = ndtr((zeta_min_lnM-lnM)/SZscatter_lnM)
        r = r_min + (ndtr_max-r_min)*self.rng.random(len(lnM))
        lnM_zeta = ndtri(r)*SZscatter_lnM + lnM
        lnw = log_ndtr((lnM-zeta_min_lnM)/SZscatter_lnM)
        return lnM_zeta, lnw

    def P_xi_zeta(self, lnM_zeta, dataID):
        """Return probability P(xi|zeta) for cluster `dataID`."""
        zeta = np.exp(scaling_relations.lnmass2lnobs('zeta', lnM_zeta, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, SPTfield=self.SPT_survey[self.SPT_survey['FIELD'] == self.catalog['FIELD'][dataID]]))
        xi = scaling_relations.zeta2xi(zeta)
        lnP = -.5*(self.catalog['XI'][dataID]-xi)**2 - .5*ln2pi
        return lnP

    def get_covmat_obs(self, obsnames):
        """Returns covariance matrix for the requested `obsnames`."""
        N_obs = len(obsnames)
        scatter = np.array([self.scaling[scatter_dict[obs]] for obs in obsnames])
        covmat = scatter[:, None]*scatter[None, :]
        for i in range(N_obs):
            for j in range(i+1, N_obs):
                covmat[i, j] *= self.scaling['rho%s%s' % (rho_dict[obsnames[i]], rho_dict[obsnames[j]])]
                covmat[j, i] *= self.scaling['rho%s%s' % (rho_dict[obsnames[i]], rho_dict[obsnames[j]])]
        return covmat

    def get_conditional(self, lnM, lnM_meas, covmat_lnM, obsname_meas, all_obsnames):
        """Return mean and (co-)variance conditioned on
        `obsname_meas`=`lnM_meas` along with list of remaining observable names
        `obsnames_cond`."""
        # Observable names and indices
        idx_meas = all_obsnames.index(obsname_meas)
        obsnames_cond = all_obsnames.copy()
        obsnames_cond.remove(obsname_meas)
        # Mean of conditional distribution mu_1(x_2=x_2) = mu_1 + Sigma_12/Sigma_22 (x_2-mu_2)
        Sigma12 = np.delete(covmat_lnM[:, idx_meas, :], idx_meas, axis=1)
        lnM_cond = np.delete(lnM, idx_meas, axis=1) + Sigma12 / covmat_lnM[:, idx_meas, idx_meas][:, None] * (lnM_meas-lnM[:, idx_meas])[:, None]
        # Variance of conditional distribution var(x_2=x_2) = var_11 - Sigma_12/Sigma_22*Sigma_21
        var = (np.delete(np.delete(covmat_lnM.copy(), idx_meas, axis=1), idx_meas, axis=2)
               - Sigma12[:, :, None]*Sigma12[:, None, :] / covmat_lnM[:, idx_meas, idx_meas][:, None, None])
        return lnM_cond, var, obsnames_cond

    def get_lnlike_richness(self, lnM, richness_std_lnM, dataID, obsname):
        """Return ln-likelihood of richness and the draws of intrinsic
        ln(M_richness)."""
        # Lognormal scatter
        if self.richness_scatter_model == 'lognormal':
            lnM_richness = scaling_relations.obs2lnmass(obsname, self.catalog['richness'][dataID], self.catalog['REDSHIFT'][dataID], self.scaling)
            lnrichness_std = richness_std_lnM/scaling_relations.dlnM_dlnobs(obsname, self.scaling)
            lnrichness = scaling_relations.lnmass2lnobs(obsname, lnM, self.catalog['REDSHIFT'][dataID], self.scaling)
            lnlike = -.5*(np.log(self.catalog['richness'][dataID])-lnrichness)**2/lnrichness_std**2 - np.log(self.catalog['richness'][dataID]*lnrichness_std) - .5*ln2pi
        # In all other cases we need to draw richness
        elif self.richness_scatter_model in ['lognormalrelPoisson', 'lognormalGaussPoisson', 'lognormalGaussmeaserror']:
            lnM_richness = self.rng.normal(lnM, richness_std_lnM)
            lnrichness = scaling_relations.lnmass2lnobs(obsname, lnM_richness, self.catalog['REDSHIFT'][dataID], self.scaling)
            richness = np.exp(lnrichness)
            # Lognormal scatter in richness gets additional 1/lambda for relative shot noise
            if self.richness_scatter_model == 'lognormalrelPoisson':
                lnlike = -.5*(np.log(self.catalog['richness'][dataID])-lnrichness)**2*richness - np.log(self.catalog['richness'][dataID]) - .5*ln2pi + .5*lnrichness
            # Convolve lognormal scatter with Gaussian of width sqrt(richness)
            elif self.richness_scatter_model == 'lognormalGaussPoisson':
                lnlike = -.5*(self.catalog['richness'][dataID]-richness)**2/richness - .5*np.log(2*np.pi*richness)
            # Gaussian measurement error
            elif self.richness_scatter_model == 'lognormalGaussmeaserror':
                lnlike = -.5*(self.catalog['richness'][dataID]-richness)**2/self.catalog['richness_err'][dataID]**2 - .5*ln2pi - np.log(self.catalog['richness_err'][dataID])
        # No valid option
        else:
            raise RuntimeError("richness_scatter_model %s not found" % self.richness_scatter_model)
        return lnlike, lnM_richness

    def get_lnlike_WL(self, lnM, WL_std_lnM, dataID, obsname):
        """Return ln-likelihood of lensing shear profile and the draws of
        intrinsic ln(M_WL)."""
        lnM_lensing = self.rng.normal(lnM, WL_std_lnM)
        lnobs = scaling_relations.lnmass2lnobs(obsname, lnM_lensing, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, self.catalog['SPT_ID'][dataID])
        # Draw from HST large-scale structure scatter
        if obsname == 'WLHST':
            obs = np.exp(lnobs)
            idx = self.HSTcalib['SPT_ID'] == self.catalog['SPT_ID'][dataID]
            std = msqrt(self.HSTcalib['LSS'][idx]**2 + self.HSTcalib['LOS'][idx]**2)
            r_min = ndtr((1-obs)/std)
            r = r_min + (ndtr_max-r_min)*self.rng.random(len(obs))
            lnobs = np.log(obs + ndtri(r)*std)
        # Lensing likelihood
        if self.lensingres is None:
            self.lnMwl = np.linspace(np.amin(lnobs), np.amax(lnobs), 64)
            self.lensingres = self.WL.one_cluster(self.catalog[dataID], np.exp(self.lnMwl))
            self.lensinglikeinterp = interp1d(self.lnMwl, self.lensingres[0], fill_value='extrapolate', assume_sorted=True)
        lnlike = self.lensinglikeinterp(lnobs)
        # Shear profile for DES stacks
        if self.get_stacked_DES & (obsname == 'WLDES'):
            self.DES_shear_profile_MC = interp1d(self.lnMwl, self.lensingres[1], axis=0, fill_value='extrapolate', assume_sorted=True)(lnobs)
        return lnlike, lnM_lensing

    def weights_for_mass_samples(self, lnM, obsnames, covmat, dlnM_dlnobs, dataID, do_obs=True):
        """Return weights for P(xi,obs|M,z,p)"""
        # Draw zeta, and compute P(xi)
        i = obsnames.index('zeta')
        lnM_zeta, zeta_lnweights = self.draw_lnzeta_given_lnM(lnM, dataID, np.sqrt(covmat[i, i])*dlnM_dlnobs[i])
        xi_lnweights = self.P_xi_zeta(lnM_zeta, dataID)
        # Weight with mass function
        mass_lnweights = self.get_mass_function_lnweights(self.catalog['REDSHIFT'][dataID], lnM)
        # Follow-up observables
        if do_obs:
            # Covariance matrix with mass dependent scatter
            N_obs = len(obsnames)
            covmat_lnM = covmat * dlnM_dlnobs[:, None]*dlnM_dlnobs[None, :] * np.ones((len(lnM), N_obs, N_obs))
            if 'WLDES' in obsnames:
                scatter = scaling_relations.WLscatter('WLDES', lnM, self.catalog['REDSHIFT'][dataID], self.scaling)
                covmat_lnM[:, obsnames.index('WLDES'), :] *= scatter[:, None]
                covmat_lnM[:, :, obsnames.index('WLDES')] *= scatter[:, None]
            elif 'WLHST' in obsnames:
                scatter = self.scaling['DWL_HST'][self.catalog['SPT_ID'][dataID]]
                covmat_lnM[:, obsnames.index('WLHST'), :] *= scatter
                covmat_lnM[:, :, obsnames.index('WLHST')] *= scatter
            # Get mean and (co-)variance of follow-up observables conditioned on lnM_zeta
            lnM_remaining = lnM[:, None] * np.ones(len(obsnames))[None, :]
            lnM_remaining, var_remaining, obsnames_remaining = self.get_conditional(lnM_remaining, lnM_zeta, covmat_lnM, 'zeta', obsnames)
            # Likelihood of follow-up observables
            lnlike_obs = []
            # Always pick the first element and then remove it from list
            while True:
                if 'richness' in obsnames_remaining[0]:
                    tmp, lnM_meas = self.get_lnlike_richness(lnM_remaining[:, 0], np.sqrt(var_remaining[:, 0, 0]), dataID, obsnames_remaining[0])
                elif 'WL' in obsnames_remaining[0]:
                    tmp, lnM_meas = self.get_lnlike_WL(lnM_remaining[:, 0], np.sqrt(var_remaining[:, 0, 0]), dataID, obsnames_remaining[0])
                lnlike_obs.append(tmp)
                # Condition on this follow-up observable or finish
                if len(obsnames_remaining) > 1:
                    lnM_remaining, var_remaining, obsnames_remaining = self.get_conditional(lnM_remaining, lnM_meas, var_remaining, obsnames_remaining[0], obsnames_remaining)
                else:
                    break
        else:
            lnlike_obs = None
        return xi_lnweights, zeta_lnweights, mass_lnweights, lnlike_obs

    def get_lndN_dallobs(self, obsnames, covmat, dlnM_dlnobs, dataID):
        """Returns ln(dN/d(obs,xi,z))"""
        # Mass draws uniform in log
        lnM = self.lnM_arr[0] + (self.lnM_arr[-1]-self.lnM_arr[0])*self.rng.random(Ndraw_ini)
        # Likelihood
        xi_lnweights, zeta_lnweights, mass_lnweights, lnlike_obs = self.weights_for_mass_samples(lnM, obsnames, covmat, dlnM_dlnobs, dataID)
        # Mass|obs,xi
        lnweights_Pobsxi = xi_lnweights + zeta_lnweights + mass_lnweights + np.sum(lnlike_obs, axis=0)
        max_lnweights_Pobsxi = np.amax(lnweights_Pobsxi)
        # Check if we have enough "good" weights
        if (max_lnweights_Pobsxi-lnweights_Pobsxi < 2).sum() < 10:
            # New mass draws uniform in log
            idx = np.argsort(lnweights_Pobsxi)[-10:]
            lo = np.amin(lnM[idx])
            lnM = lo + (np.amax(lnM[idx])-lo)*self.rng.random(Ndraw_ini)
            # Likelihood
            xi_lnweights, zeta_lnweights, mass_lnweights, lnlike_obs = self.weights_for_mass_samples(lnM, obsnames, covmat, dlnM_dlnobs, dataID)
            # Mass|obs,xi
            lnweights_Pobsxi = xi_lnweights + zeta_lnweights + mass_lnweights + np.sum(lnlike_obs, axis=0)
            max_lnweights_Pobsxi = np.amax(lnweights_Pobsxi)
        # Mean and std of lnM
        w = np.exp(lnweights_Pobsxi-max_lnweights_Pobsxi)
        sum_w = np.sum(w)
        if np.sum(w**2)/sum_w == sum_w:
            print('bad mass post', self.catalog['SPT_ID'][dataID])
            # np.save(self.catalog['SPT_ID'][dataID], [lnM, xi_lnweights, zeta_lnweights, mass_lnweights, lnlike_obs[0], lnlike_obs[1]])
            return -np.inf
        mean_lnM_obsxi = np.sum(w*lnM)/sum_w
        std_lnM_obsxi = np.sqrt(np.sum(w*(lnM-mean_lnM_obsxi)**2) / (sum_w - np.sum(w**2)/sum_w))
        # Refined estimate P(obs,xi)
        r_min = ndtr((self.lnM_arr[0]-mean_lnM_obsxi)/std_lnM_obsxi)
        r_max = ndtr((self.lnM_arr[-1]-mean_lnM_obsxi)/std_lnM_obsxi)
        r = r_min + (r_max-r_min)*self.rng.random(Ndraw)
        lnM = ndtri(r)*std_lnM_obsxi + mean_lnM_obsxi
        draw_lnweights = -(-.5*((lnM-mean_lnM_obsxi)/std_lnM_obsxi)**2 - .5*ln2pi - np.log(std_lnM_obsxi))
        # Likelihood
        xi_lnweights, zeta_lnweights, mass_lnweights, lnlike_obs = self.weights_for_mass_samples(lnM, obsnames, covmat, dlnM_dlnobs, dataID)
        lnweights_Pobsxi = draw_lnweights + xi_lnweights + zeta_lnweights + mass_lnweights + np.sum(lnlike_obs, axis=0)
        shift_lnweights = np.amax(lnweights_Pobsxi)
        lndNdobsxi = np.log(np.mean(np.exp(lnweights_Pobsxi-shift_lnweights))) + shift_lnweights
        # Stacked DES profile
        if self.get_stacked_DES & ('WLDES' in obsnames):
            lnweights = draw_lnweights + xi_lnweights + zeta_lnweights + mass_lnweights
            lnweights -= np.amax(lnweights)
            weights = np.exp(lnweights)
            sum_weights = np.sum(weights)
            # Shear profile
            profile_interp = interp1d(self.catalog['WLdata'][dataID]['r_arcmin'], self.DES_shear_profile_MC, fill_value='extrapolate', assume_sorted=True)
            profile_interpolated = profile_interp(self.catalog['WLdata'][dataID]['r_arcmin_stack'])
            self.catalog['DES_shear_profile_mean'][dataID] = np.sum(profile_interpolated*weights[:, None], axis=0)/sum_weights
        return lndNdobsxi
