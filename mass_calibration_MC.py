from __future__ import division, print_function
import numpy as np
import os
import imp
import warnings
from math import sqrt as msqrt
from multiprocessing import Pool
from astropy.table import Table
from scipy.special import erfinv, ndtr
from scipy import integrate, signal
from scipy.interpolate import interp1d, RectBivariateSpline
import cosmo, Mconversion_concentration, scaling_relations

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
GETPULL = False
Ndraw = 8192
ndtr_m5 = ndtr(-5)
ndtr_p4 = ndtr(4)

# Limits for stack
z_mid = .55
xi_mid = 5.5
lnr_r200c_stack = np.linspace(np.log(.3), np.log(5), 16)

scatter_dict = {'zeta': 'Dsz', 'richness': 'Drichness',
                'Mgas': 'Dx', 'Yx': 'Dx',
                'WLMegacam': 'DWL_Megacam', 'WLDES': 'one', 'WLHST': 'one'}
rho_dict = {'zeta': 'SZ', 'richness': 'richness', 'Mgas': 'X', 'Yx': 'X',
            'WLDES': 'WL', 'WLHST': 'WL', 'WLMegacam': 'WL'}

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlike(*arg)

################################################################################
class MassCalibration:

    def __init__(self, todo, mcType,
                 surveyCutRedshift, surveyCutRichness, richness_scatter_model,
                 SPT_survey_fields, SPTcatalogfile,
                 HSTcalibfile,
                 NPROC,
                 get_stacked_DES=False):

        self.NPROC = NPROC
        self.get_stacked_DES = get_stacked_DES
        self.todo = todo
        self.mcType = mcType
        self.surveyCutRedshift = surveyCutRedshift
        self.surveyCutRichness = surveyCutRichness
        self.richness_scatter_model = richness_scatter_model

        # Read input files
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        self.catalog = Table.read(SPTcatalogfile)
        if self.get_stacked_DES:
            self.catalog['DES_shear_profile_mean'] = [None for i in range(len(self.catalog))]
            self.catalog['DES_DeltaSigma_mean'] = [None for i in range(len(self.catalog))]
            self.catalog['DES_DeltaSigma_data_mean'] = [None for i in range(len(self.catalog))]
        self.HSTcalib = Table.read(HSTcalibfile, format='ascii.commented_header')



    ############################################################################
    def lnlike(self, HMF, cosmology, scaling):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        self.cosmology = cosmology
        self.scaling = scaling
        self.scaling['one'] = 1.
        self.xi_min = scaling_relations.zeta2xi(self.scaling['zeta_min'])

        ##### Set up interpolation for HMF
        HMF_in = HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.lnM_arr = np.log(HMF['M_arr'])
        self.HMF_interp = RectBivariateSpline(np.log(HMF['z_arr'][1:]), self.lnM_arr, np.log(HMF_in), kx=1, ky=1)

        ##### Initialize mass-concentration relation class (for WL and dispersions)
        if self.todo['veldisp']:
            self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology,
                                                                           setup_interp=True, interp_massdef=500)

        ##### Evaluate the individual likelihoods
        len_data = len(self.catalog['SPT_ID'])
        if self.NPROC==0:
            # Iterate through cluster list
            likelihoods = np.array([self.clusterlike(i) for i in range(len_data)])
        else:
            # Launch a multiprocessing pool and get the likelihoods
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*len_data, range(len_data))
                likelihoods = pool.map(unwrap_self_f, argin)

        with np.errstate(all='ignore'):
            lnlike = np.sum(np.log(likelihoods))

        if np.isinf(lnlike)|np.isnan(lnlike):
            return -np.inf, None

        ##### DES stacked shear profile
        if self.get_stacked_DES:
            stack = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                notNone = [this is not None for this in self.catalog['DES_shear_profile_mean']]
                idx = (notNone&(self.catalog['REDSHIFT']<z_mid)&(self.catalog['XI']<xi_mid)).nonzero()[0]
                stack['shear_zloxilo'] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
                stack['DeltaSigma_zloxilo'] = np.mean(self.catalog['DES_DeltaSigma_mean'][idx], axis=0)
                tmp = np.array([x for x in self.catalog['DES_DeltaSigma_data_mean'][idx]])
                stack['DeltaSigma_data_zloxilo'] = np.nanmean(tmp, axis=0)
                idx = (notNone&(self.catalog['REDSHIFT']<z_mid)&(self.catalog['XI']>=xi_mid)).nonzero()[0]
                stack['shear_zloxihi'] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
                stack['DeltaSigma_zloxihi'] = np.mean(self.catalog['DES_DeltaSigma_mean'][idx], axis=0)
                tmp = np.array([x for x in self.catalog['DES_DeltaSigma_data_mean'][idx]])
                stack['DeltaSigma_data_zloxihi'] = np.nanmean(tmp, axis=0)
                idx = (notNone&(self.catalog['REDSHIFT']>=z_mid)&(self.catalog['XI']<xi_mid)).nonzero()[0]
                stack['shear_zhixilo'] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
                stack['DeltaSigma_zhixilo'] = np.mean(self.catalog['DES_DeltaSigma_mean'][idx], axis=0)
                tmp = np.array([x for x in self.catalog['DES_DeltaSigma_data_mean'][idx]])
                stack['DeltaSigma_data_zhixilo'] = np.nanmean(tmp, axis=0)
                idx = (notNone&(self.catalog['REDSHIFT']>=z_mid)&(self.catalog['XI']>=xi_mid)).nonzero()[0]
                stack['shear_zhixihi'] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
                stack['DeltaSigma_zhixihi'] = np.mean(self.catalog['DES_DeltaSigma_mean'][idx], axis=0)
                tmp = np.array([x for x in self.catalog['DES_DeltaSigma_data_mean'][idx]])
                stack['DeltaSigma_data_zhixihi'] = np.nanmean(tmp, axis=0)
            return lnlike, stack
        else:
            return lnlike, None



    ############################################################################
    def clusterlike(self, i):
        """Return multi-wavelength mass-calibration likelihood for a
        given cluster (index) by calling get_P_1obs_xi or get_P_2obs_xi or
        returning 1 if no follow-up data is available."""
        # t0 = time.time()
        name = self.catalog['SPT_ID'][i]

        ##### Do we actually want this guy?
        if self.catalog['COSMO_SAMPLE'][i]==0:
            return 1.
        if (self.catalog['XI'][i]<self.SPT_survey['XI_MIN'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]]):
            return 1.
        if not (self.surveyCutRedshift[0]<self.catalog['REDSHIFT'][i]<self.surveyCutRedshift[1]):
            return 1.

        ##### Check if follow-up is available
        nobs = 0
        obsnames = ['zeta',]
        if self.todo['WL'] and self.catalog['WLdata'][i] is not None:
            nobs+= 1
            if self.catalog['WLdata'][i]['datatype']=='Megacam':
                obsnames.append('WLMegacam')
            elif self.catalog['WLdata'][i]['datatype']=='DES':
                obsnames.append('WLDES')
            elif self.catalog['WLdata'][i]['datatype']=='HST':
                obsnames.append('WLHST')
        if self.todo['veldisp'] and self.catalog['veldisp'][i]!=0.:
            nobs+= 1
            obsnames.append('disp')
        if self.todo['Yx'] and self.catalog['Mg_fid'][i]!=0:
            nobs+= 1
            obsnames.append('Yx')
        if self.todo['Mgas'] and self.catalog['Mg_fid'][i]!=0:
            nobs+= 1
            obsnames.append('Mgas')
        if self.todo['richness'] and self.catalog['richness'][i]>0.:
            nobs+= 1
            obsnames.append('richness')
        if nobs==0:
            return 1.

        ##### Set SPT field scaling factor
        self.thisSPTfield_gamma = float(self.SPT_survey['GAMMA'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]])
        if self.SPT_survey['SURVEY'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]]=='SPECS':
            self.thisSPTfield_gamma*= self.scaling['SPECS_calib']

        ##### Set up random number generator and get likelihood
        seed = np.abs(int(123456.*i*np.prod([self.scaling[key] for key in ['Asz', 'Bsz', 'Csz', 'Dsz']])))
        self.rng = np.random.default_rng(seed)
        probability = self.get_P_obs_xi_zetadraw(obsnames, i)

        if (probability<0) | np.isnan(probability) | np.isinf(probability):
            return 0.
            # raise ValueError("P(obs|xi) =", probability, name)

        # print(name, obsnames, probability,)# time.time()-t0)
        return probability


    ############################################################################
    def conversion_factor_Xray_obs_r500ref(self, dataID):
        """Account for the cosmological dependence of the X-ray observable and
        convert to the model expectation at r500ref using the slope of the
        radial profile. This is done for the mass array self.HMF_convos['M_arr']."""
        # Angular diameter distances in current and reference cosmology [Mpc]
        dA = cosmo.dA(self.catalog['REDSHIFT'][dataID], self.cosmology)/self.cosmology['h']
        dAref = cosmo.dA(self.catalog['REDSHIFT'][dataID], cosmologyRef)/cosmologyRef['h']
        # R500 [kpc]
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['REDSHIFT'][dataID], self.cosmology)**2
        r500 = 1000 * (3*self.HMF_convos['M_arr']/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
        # r500 in reference cosmology [kpc]
        r500ref = r500 * dAref/dA
        # Xray observable at fiducial r500...
        correction = (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
        # ... corrected to reference cosmology
        correction*= (dAref/dA)**2.5
        return correction


    def get_zeta_draws(self, xi):
        """Draw zetas from `xi`. In practice, draw from N(xi-1, 1) so that
        there are more low-mass samples which will later be up-weighted by the
        mass function. Return zeta and weights."""
        xi_offset = -3/xi**2
        # xi_draw > xi_min, xi_draw > xi-5, xi_draw < xi+4
        r_min = ndtr(self.xi_min-(xi+xi_offset))
        if r_min<ndtr_m5:
            r_min = ndtr_m5
        r = r_min + (ndtr_p4-r_min)*self.rng.random(Ndraw)
        # Percent point function (scipy stats is too slow)
        xi0 = erfinv(2*r-1)*msqrt(2) + xi+xi_offset
        zeta = scaling_relations.xi2zeta(xi0)
        # Ratio of normals
        zeta_lnweights = -.5 * ((xi0-xi)**2 - (xi0-(xi+xi_offset))**2)
        return zeta, zeta_lnweights


    def get_mass_function_lnweights(self, z, lnM):
        """Return log-probability of halo mass function
        ln(P(M)) = ln(dN/dlnM /M) at given `z` and array `lnM`."""
        # Scipy.RectBivariateSpline only accepts sorted inputs
        idx = np.argsort(lnM)
        mass_lnweights = np.zeros(len(lnM))
        mass_lnweights[idx] = self.HMF_interp(np.log(z), lnM[idx])-lnM[idx]
        mass_lnweights-= np.amax(mass_lnweights)
        return mass_lnweights


    def draw_lnm_given_lnzeta(self, lnM_zeta, SZscatter_lnM):
        """Return draws of ln(mass) given ln(zeta)."""
        offset = -3*SZscatter_lnM**2
        r_min = ndtr((self.lnM_arr[0]-(lnM_zeta+offset))/SZscatter_lnM)
        r_max = ndtr((self.lnM_arr[-1]-lnM_zeta)/SZscatter_lnM)
        r_min[r_min<ndtr_m5] = ndtr_m5
        r_max[r_max>ndtr_p4] = ndtr_p4
        r = r_min + (r_max-r_min)*self.rng.random(len(lnM_zeta))
        lnM = erfinv(2*r-1)*SZscatter_lnM*msqrt(2) + lnM_zeta+offset
        lnweights = -.5/SZscatter_lnM**2 * ((lnM-lnM_zeta)**2 - (lnM-(lnM_zeta+offset))**2)
        return lnM, lnweights


    def get_covmat_obs(self, obsnames):
        """Returns covariance matrix for the requested `obsnames`."""
        N_obs = len(obsnames)
        scatter = np.array([self.scaling[scatter_dict[obs]] for obs in obsnames])
        covmat = scatter[:,None]*scatter[None,:]
        for i in range(N_obs):
            for j in range(i+1,N_obs):
                covmat[i,j]*= self.scaling['rho%s%s'%(rho_dict[obsnames[i]], rho_dict[obsnames[j]])]
                covmat[j,i]*= self.scaling['rho%s%s'%(rho_dict[obsnames[i]], rho_dict[obsnames[j]])]
        return covmat


    def get_lnM_zeta_given_xi(self, dataID):
        """Returns mass draws given xi."""
        zeta, lnweights = self.get_zeta_draws(self.catalog['XI'][dataID])
        zeta/= self.thisSPTfield_gamma
        lnM = np.log(scaling_relations.obs2mass('zeta', zeta, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
        return lnM, lnweights


    def get_conditional(self, lnM, lnM_meas, covmat_lnM, obsname_meas, all_obsnames):
        """Return mean and (co-)variance conditioned on
        `obsname_meas`=`lnM_meas` along with list of remaining observable names
        `obsnames_cond`."""
        # index of obsname_meas
        idx_meas = all_obsnames.index(obsname_meas)
        # Mean of conditional distribution
        if lnM.ndim==1:
            lnM_cond = lnM[:,None] + np.delete(covmat_lnM[:,idx_meas,:], idx_meas, axis=1) / covmat_lnM[:,idx_meas,idx_meas][:,None] * (lnM_meas-lnM)[:,None]
        else:
            lnM_cond = np.delete(lnM, idx_meas, axis=1) + np.delete(covmat_lnM[:,idx_meas,:], idx_meas, axis=1) / covmat_lnM[:,idx_meas,idx_meas][:,None] * (lnM_meas-lnM[:,idx_meas])[:,None]
        # Variance (depends on dimensionality)
        obsnames_cond = all_obsnames.copy()
        obsnames_cond.remove(obsname_meas)
        if len(obsnames_cond)==1:
            idx_cond = all_obsnames.index(obsnames_cond[0])
            var = (covmat_lnM[:,idx_cond,idx_cond] - covmat_lnM[:,0,1]**2/covmat_lnM[:,idx_meas,idx_meas])[:,None,None]
        elif len(obsnames_cond)==2:
            idx_cond_0 = all_obsnames.index(obsnames_cond[0])
            idx_cond_1 = all_obsnames.index(obsnames_cond[1])
            var = np.delete(np.delete(covmat_lnM.copy(), idx_meas, axis=1), idx_meas, axis=2)
            var[:,0,0]-= covmat_lnM[:,idx_meas,idx_cond_0]**2/covmat_lnM[:,idx_meas,idx_meas]
            var[:,1,1]-= covmat_lnM[:,idx_meas,idx_cond_1]**2/covmat_lnM[:,idx_meas,idx_meas]
            tmp = covmat_lnM[:,idx_meas,idx_cond_0]*covmat_lnM[:,idx_meas,idx_cond_1]/covmat_lnM[:,idx_meas,idx_meas]
            var[:,0,1]-= tmp
            var[:,1,0]-= tmp
        return lnM_cond, var, obsnames_cond


    def get_lnlike_richness(self, lnM, richness_std_lnM, dataID):
        """Return ln-likelihood of richness and the draws of intrinsic
        ln(M_richness)."""
        # Compute lambda_min
        if self.todo['lambda_min']:
            if self.catalog['FIELD'][dataID]=='SPTPOL_500d':
                lambda_min = self.surveyCutRichness['deep'](self.catalog['REDSHIFT'][dataID])
            else:
                lambda_min = self.surveyCutRichness['shallow'](self.catalog['REDSHIFT'][dataID])
        # Lognormal scatter
        if self.richness_scatter_model=='lognormal':
            lnM_richness = np.log(scaling_relations.obs2mass('richness', self.catalog['richness'][dataID], self.catalog['REDSHIFT'][dataID], self.scaling))
            lnrichness_std = richness_std_lnM/scaling_relations.dlnM_dlnobs('richness', self.scaling)
            richness = scaling_relations.mass2obs('richness', np.exp(lnM), self.catalog['REDSHIFT'][dataID], self.scaling)
            lnlike = -.5*np.log(self.catalog['richness'][dataID]/richness)**2/lnrichness_std**2 - np.log(self.catalog['richness'][dataID]*lnrichness_std) - .5*np.log(2*np.pi)
            if self.todo['lambda_min']:
                with np.errstate(all='ignore'):
                    lnP_lambda_gtr_lambda_min = np.log(ndtr(np.log(richness/lambda_min)/lnrichness_std))
                lnlike-= lnP_lambda_gtr_lambda_min
        # In all other cases we need to draw richness
        elif self.richness_scatter_model in ['lognormalrelPoisson', 'lognormalGaussPoisson']:
            lnM_richness = self.rng.normal(lnM, richness_std_lnM)
            richness = scaling_relations.mass2obs('richness', np.exp(lnM_richness), self.catalog['REDSHIFT'][dataID], self.scaling)
            # Lognormal scatter in richness gets additional 1/lambda for relative shot noise
            if self.richness_scatter_model=='lognormalrelPoisson':
                lnlike = -.5*np.log(self.catalog['richness'][dataID]/richness)**2*richness - np.log(self.catalog['richness'][dataID]*np.sqrt(2*np.pi/richness))
                if self.todo['lambda_min']:
                    lnP_lambda_gtr_lambda_min = np.log(ndtr(np.log(richness/lambda_min)*np.sqrt(richness)))
                    lnlike-= lnP_lambda_gtr_lambda_min
            # Convolve lognormal scatter with Gaussian of width sqrt(richness)
            elif self.richness_scatter_model=='lognormalGaussPoisson':
                lnlike = -.5*(self.catalog['richness'][dataID]-richness)**2/richness - .5*np.log(2*np.pi*richness)
                if self.todo['lambda_min']:
                    with np.errstate(all='ignore'):
                        lnP_lambda_gtr_lambda_min = np.log(ndtr((richness-lambda_min)/np.sqrt(richness)))
                    lnlike-= lnP_lambda_gtr_lambda_min
        # No valid option
        else:
            raise RuntimeError("richness_scatter_model %s not found"%self.richness_scatter_model)
        return lnlike, lnM_richness


    def get_lnlike_WL(self, lnM, WL_std_lnM, dataID, obsname):
        """Return ln-likelihood of lensing shear profile and the draws of
        intrinsic ln(M_WL)."""
        lnM_lensing = self.rng.normal(lnM, WL_std_lnM)
        obs = scaling_relations.mass2obs(obsname, np.exp(lnM_lensing), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, self.catalog['SPT_ID'][dataID])
        # Draw from HST large-scale structure scatter
        if obsname=='WLHST':
            idx = (self.HSTcalib['SPT_ID']==self.catalog['SPT_ID'][dataID]).nonzero()[0]
            std = msqrt(self.HSTcalib['LSS'][idx]**2 + self.HSTcalib['LOS'][idx]**2)
            r_min = ndtr((1-obs)/std)
            r_min[ndtr_m5>r_min] = ndtr_m5
            r = r_min + (ndtr_p4-r_min)*self.rng.random(len(obs))
            obs+= erfinv(2*r-1)*std*msqrt(2)
        # Lensing likelihood
        lnobs = np.log(obs)
        lnM_ = np.linspace(np.amin(lnobs), np.amax(lnobs), 32)
        tmp = self.WL.one_cluster(self.catalog[dataID], np.exp(lnM_))
        lnlike = interp1d(lnM_, tmp[0])(lnobs)
        # Shear profile for DES stacks
        if self.get_stacked_DES & (obsname=='WLDES'):
            self.DES_shear_profile_MC = interp1d(lnM_, tmp[1], axis=0)(lnobs)
            self.DES_DeltaSigma_MC = interp1d(lnM_, tmp[2], axis=0)(lnobs)
            self.DES_r_r200c_MC = interp1d(lnM_, tmp[3], axis=0)(lnobs)
            self.DES_DeltaSigma_data_MC = interp1d(lnM_, tmp[4], axis=0)(lnobs)
        return lnlike, lnM_lensing


    ############################################################################
    def get_P_obs_xi_zetadraw(self, obsnames, dataID):
        """Returns P(obs|xi,z,p)"""
        # Basic setup
        N_obs = len(obsnames)
        z_cluster = self.catalog['REDSHIFT'][dataID]
        dlnM_dlnobs = np.array([scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames])
        # Mass given zeta
        lnM_zeta, xi_lnweights = self.get_lnM_zeta_given_xi(dataID)
        SZscatter_lnM = self.scaling['Dsz'] * dlnM_dlnobs[obsnames.index('zeta')]
        lnM, zeta_lnweights = self.draw_lnm_given_lnzeta(lnM_zeta, SZscatter_lnM)
        # Sometimes there are failures when mean obs is very unlikely
        if np.all(np.isinf(lnM)):
            return 0.
        idx = np.isfinite(lnM)
        if not np.all(idx):
            lnM = lnM[idx]
            lnM_zeta = lnM_zeta[idx]
            xi_lnweights = xi_lnweights[idx]
            zeta_lnweights = zeta_lnweights[idx]
        # Weight with mass function
        mass_lnweights = self.get_mass_function_lnweights(z_cluster, lnM)
        # Normalization P(xi)
        Pxi = np.mean(np.exp(xi_lnweights + zeta_lnweights + mass_lnweights))
        # Covariance matrix in ln-mass [draw, N_obs, N_obs]
        covmat = self.get_covmat_obs(obsnames)
        Jacobian = dlnM_dlnobs[:,None]*dlnM_dlnobs[None,:]
        covmat_lnM = covmat * Jacobian * np.ones((len(lnM),N_obs,N_obs))
        if 'WLDES' in obsnames:
            DES_scatter = scaling_relations.WLscatter('main', np.exp(lnM), z_cluster, self.scaling)
            covmat_lnM[:,obsnames.index('WLDES'),:]*= DES_scatter[:,None]
            covmat_lnM[:,:,obsnames.index('WLDES')]*= DES_scatter[:,None]
        elif 'WLHST' in obsnames:
            scatter = self.scaling['DWL_HST'][self.catalog['SPT_ID'][dataID]]
            covmat_lnM[:,obsnames.index('WLHST'),:]*= scatter
            covmat_lnM[:,:,obsnames.index('WLHST')]*= scatter
        # Get mean and (co-)variance of follow-up observables conditioned on lnM_zeta
        lnM_remaining, var_remaining, obsnames_remaining = self.get_conditional(lnM, lnM_zeta, covmat_lnM, 'zeta', obsnames)
        # Likelihood of follow-up observables
        lnlike_obs = []
        # Always pick the first element and then remove it from list
        while True:
            if obsnames_remaining[0]=='richness':
                tmp, lnM_meas = self.get_lnlike_richness(lnM_remaining[:,0], np.sqrt(var_remaining[:,0,0]), dataID)
            elif obsnames_remaining[0] in ['WLDES', 'WLHST', 'WLMegacam']:
                tmp, lnM_meas = self.get_lnlike_WL(lnM_remaining[:,0], np.sqrt(var_remaining[:,0,0]), dataID, obsnames_remaining[0])
            lnlike_obs.append(tmp)
            # Condition on this follow-up observable or finish
            if len(obsnames_remaining)>1:
                lnM_remaining, var_remaining, obsnames_remaining = self.get_conditional(lnM_remaining, lnM_meas, var_remaining, obsnames_remaining[0], obsnames_remaining)
            else:
                break
        # Final likelihood
        lnweights = xi_lnweights + zeta_lnweights + mass_lnweights + np.sum(lnlike_obs, axis=0)
        # Let's accept a few invalid draws (and discard them)
        lnweights = lnweights[np.isfinite(lnweights)]
        if len(lnweights)<Ndraw-32:
            return 0.
        # Lots of potential warnings ahead...
        with np.errstate(all='ignore'):
            # If we cannot even compute the largest weight we're doomed
            if np.exp(np.amax(lnweights))==0:
                return 0.
            # Let's be optimistic
            Pobsxi = np.mean(np.exp(lnweights))
            like = Pobsxi/Pxi
            # Maybe it works by shifting to the mean ln-weight
            if (like<=0.) | np.isinf(like) | np.isnan(like):
                shift_lnweights = np.mean(lnweights)
                diff_lnweights = lnweights - shift_lnweights
                Pobsxi = np.exp(shift_lnweights) * np.mean(np.exp(diff_lnweights))
                like = Pobsxi/Pxi
                # Now we're desperate - maybe shift such that max(weight) = 1
                if (like<=0.) | np.isinf(like) | np.isnan(like):
                    shift_lnweights = np.amax(lnweights)
                    diff_lnweights = lnweights - shift_lnweights
                    Pobsxi = np.exp(shift_lnweights) * np.mean(np.exp(diff_lnweights))
                    like = Pobsxi/Pxi
                    # OK we're giving up
                    if (like<=0.) | np.isinf(like) | np.isnan(like):
                        print(self.catalog['SPT_ID'][dataID], N_obs, 'like', like)
        # Stacked DES profile
        if self.get_stacked_DES & ('WLDES' in obsnames):
            weights = np.exp(xi_lnweights + zeta_lnweights + mass_lnweights)
            # Shear profile
            profile_interp = interp1d(self.catalog['WLdata'][dataID]['r_arcmin'], self.DES_shear_profile_MC, fill_value='extrapolate')
            profile_interpolated = profile_interp(self.catalog['WLdata'][dataID]['r_arcmin_stack'])
            self.catalog['DES_shear_profile_mean'][dataID] = np.sum(profile_interpolated*weights[:,None], axis=0)/np.sum(weights)
            # DeltaSigma model
            DeltaSigma_mean = np.sum(self.DES_DeltaSigma_MC*weights[:,None], axis=0)/np.sum(weights)
            r_r200c_mean = np.sum(self.DES_r_r200c_MC*weights[:,None], axis=0)/np.sum(weights)
            DeltaSigma_interp = interp1d(np.log(r_r200c_mean), np.log(DeltaSigma_mean), fill_value='extrapolate')
            self.catalog['DES_DeltaSigma_mean'][dataID] = np.exp(DeltaSigma_interp(lnr_r200c_stack))
            # DeltaSigma data
            DeltaSigma_data_mean = np.sum(self.DES_DeltaSigma_data_MC*weights[:,None], axis=0)/np.sum(weights)
            DeltaSigma_interp = interp1d(np.log(r_r200c_mean), DeltaSigma_data_mean, bounds_error=False)
            self.catalog['DES_DeltaSigma_data_mean'][dataID] = DeltaSigma_interp(lnr_r200c_stack)

        return like
