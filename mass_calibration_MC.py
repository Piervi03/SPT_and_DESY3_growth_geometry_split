from __future__ import division, print_function
import numpy as np
import os
import imp
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
                 SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
                 HSTcalibfile,
                 NPROC):

        self.NPROC = NPROC
        self.todo = todo
        self.mcType = mcType
        self.surveyCutRedshift = surveyCutRedshift
        self.surveyCutRichness = surveyCutRichness
        self.richness_scatter_model = richness_scatter_model

        # Read input files
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        SPTdata = imp.load_source('SPTdata', SPT_doublecounts)
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        self.catalog = Table.read(SPTcatalogfile)
        self.HSTcalib = Table.read(HSTcalibfile, format='ascii.commented_header')



    ############################################################################
    def lnlike(self, HMF, cosmology, scaling, rng_seed=1328):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        self.rng = np.random.default_rng(rng_seed)
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
            return -np.inf

        return lnlike



    ############################################################################
    def clusterlike(self, i):
        """Return multi-wavelength mass-calibration likelihood for a
        given cluster (index) by calling get_P_1obs_xi or get_P_2obs_xi or
        returning 1 if no follow-up data is available."""
        # t0 = time.time()
        name = self.catalog['SPT_ID'][i]

        ##### Do we actually want this guy? (some clusters in SPT-SZ are at field boundaries)
        if (name,self.catalog['FIELD'][i]) in self.SPTdoubleCount:
            return 1.
        if not self.SPT_survey['XI_MIN'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]]<self.catalog['XI'][i] or not self.surveyCutRedshift[0]<self.catalog['REDSHIFT'][i]<self.surveyCutRedshift[1]:
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
        if self.todo['richness'] and self.catalog['richness'][i]!=0.:
            nobs+= 1
            obsnames.append('richness')
        if nobs==0:
            return 1.

        ##### Set SPT field scaling factor
        self.thisSPTfield_gamma = float(self.SPT_survey['GAMMA'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]])
        if self.SPT_survey['SURVEY'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]]=='SPECS':
            self.thisSPTfield_gamma*= self.scaling['SPECS_calib']

        #####
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
        r_min = ndtr(self.xi_min-(xi-1))
        r = r_min + (1-r_min)*self.rng.random(Ndraw)
        # Percent point function (scipy stats is too slow)
        xi0 = erfinv(2*r-1)*msqrt(2) + xi-1
        zeta = scaling_relations.xi2zeta(xi0)
        # Ratio of normals
        zeta_lnweights = -.5 * ((xi0-xi)**2 - (xi0-(xi-1))**2)
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
        r_min = ndtr((self.lnM_arr[0]-lnM_zeta)/SZscatter_lnM)
        r_max = ndtr((self.lnM_arr[-1]-lnM_zeta)/SZscatter_lnM)
        r = r_min + (r_max-r_min)*self.rng.random(len(lnM_zeta))
        lnM = erfinv(2*r-1)*SZscatter_lnM*msqrt(2) + lnM_zeta
        return lnM


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


    def get_conditional(self, lnM, lnM_meas, covmat_lnM, pos, obsname_meas, all_obsnames):
        """Return mean and (co-)variance conditioned on
        `obsname_meas`=`lnM_meas`."""
        # Mean of conditional distribution
        lnM_cond = lnM[:,None] + np.delete(covmat_lnM[:,pos[obsname_meas],:], pos[obsname_meas], axis=1) / covmat_lnM[:,pos[obsname_meas],pos[obsname_meas]][:,None] * (lnM_meas-lnM)[:,None]
        # Variance (depends on dimensionality
        obsnames_cond = all_obsnames.copy()
        obsnames_cond.remove(obsname_meas)
        if len(obsnames_cond)==1:
            var = covmat_lnM[:,pos[obsnames_cond[0]],pos[obsnames_cond[0]]] - covmat_lnM[:,0,1]**2/covmat_lnM[:,pos[obsname_meas],pos[obsname_meas]]
        elif len(obsnames_cond)==2:
            var = np.delete(np.delete(covmat_lnM.copy(), pos[obsname_meas], axis=1), pos[obsname_meas], axis=2)
            var[:,0,0]-= covmat_lnM[:,pos[obsname_meas],pos[obsnames_cond[0]]]**2/covmat_lnM[:,pos[obsname_meas],pos[obsname_meas]]
            var[:,1,1]-= covmat_lnM[:,pos[obsname_meas],pos[obsnames_cond[1]]]**2/covmat_lnM[:,pos[obsname_meas],pos[obsname_meas]]
            tmp = covmat_lnM[:,pos[obsname_meas],pos[obsnames_cond[0]]]*covmat_lnM[:,pos[obsname_meas],pos[obsnames_cond[1]]]/covmat_lnM[:,pos[obsname_meas],pos[obsname_meas]]
            var[:,0,1]-= tmp 
            var[:,1,0]-= tmp 
        return lnM_cond, var, obsnames_cond


    def get_lnlike_obs(self, dataID, obsnames, lnM_obs_given_lnzeta_mean, covmat_lnM):
        """Return likelihood of observable(s) given their expected mean and
        covariance. For multiple observables, repeatedly apply conditional
        probability."""
        pos = {}
        for o,obs in enumerate(obsnames):
            pos[obs] = o
        lnlike = len(obsnames)*[None]

        if 'richness' in obsnames:
            obsnames_norichness = obsnames.copy()
            obsnames_norichness.remove('richness')
            # Variance in richness
            if len(obsnames)==1:
                richness_var_lnM = covmat_lnM
            elif len(obsnames)==2:
                richness_var_lnM = covmat_lnM[:,pos['richness'], pos['richness']]
            # Compute lambda_min
            if self.todo['lambda_min']:
                if self.catalog['FIELD'][dataID]=='SPTPOL_500d':
                    lambda_min = self.surveyCutRichness['deep'](self.catalog['REDSHIFT'][dataID])
                else:
                    lambda_min = self.surveyCutRichness['shallow'](self.catalog['REDSHIFT'][dataID])
            # Lognormal scatter
            if self.richness_scatter_model=='lognormal':
                lnM_richness = np.log(scaling_relations.obs2mass('richness', self.catalog['richness'][dataID], self.catalog['REDSHIFT'][dataID], self.scaling))
                lnrichness_std = np.sqrt(richness_var_lnM)/scaling_relations.dlnM_dlnobs(obs, self.scaling)
                richness = scaling_relations.mass2obs('richness', np.exp(lnM_obs_given_lnzeta_mean[:,pos['richness']]), self.catalog['REDSHIFT'][dataID], self.scaling)
                lnlike[pos['richness']] = -.5*(np.log(self.catalog['richness'][dataID]/richness))**2/lnrichness_std**2 - np.log(self.catalog['richness'][dataID]*lnrichness_std) - .5*np.log(2*np.pi)
                if self.todo['lambda_min']:
                    with np.errstate(all='ignore'):
                        lnP_lambda_gtr_lambda_min = np.log(ndtr(np.log(self.catalog['richness'][dataID]/lambda_min)/lnrichness_std))
                    lnlike[pos['richness']]-= lnP_lambda_gtr_lambda_min
            # Lognormal scatter in richness gets additional 1/lambda for relative shot noise
            elif self.richness_scatter_model=='lognormalrelPoisson':
                lnM_richness = self.rng.normal(lnM_obs_given_lnzeta_mean[:,pos['richness']], np.sqrt(richness_var_lnM))
                richness = scaling_relations.mass2obs('richness', np.exp(lnM_richness), self.catalog['REDSHIFT'][dataID], self.scaling)
                lnlike[pos['richness']] = -.5*np.log(self.catalog['richness'][dataID]/richness)**2*richness - np.log(self.catalog['richness'][dataID]*np.sqrt(2*np.pi/richness))
                if self.todo['lambda_min']:
                    lnP_lambda_gtr_lambda_min = np.log(ndtr(np.log(richness/lambda_min)*np.sqrt(richness)))
                    lnlike[pos['richness']]-= lnP_lambda_gtr_lambda_min
            # Convolve lognormal scatter with Gaussian of width sqrt(richness)
            elif self.richness_scatter_model=='lognormalGaussPoisson':
                lnM_richness = self.rng.normal(lnM_obs_given_lnzeta_mean[:,pos['richness']], np.sqrt(richness_var_lnM))
                richness = scaling_relations.mass2obs('richness', np.exp(lnM_richness), self.catalog['REDSHIFT'][dataID], self.scaling)
                lnlike[pos['richness']] = -.5*(self.catalog['richness'][dataID]-richness)**2/richness - .5*np.log(2*np.pi*richness)
                if self.todo['lambda_min']:
                    with np.errstate(all='ignore'):
                        lnP_lambda_gtr_lambda_min = np.log(ndtr((richness-lambda_min)/np.sqrt(richness)))
                    lnlike[pos['richness']]-= lnP_lambda_gtr_lambda_min
            # No valid option
            else:
                raise RuntimeError("richness_scatter_model %s not found"%self.richness_scatter_model)
            # Condition remaining observable(s) on measured richness
            if len(obsnames)>1:
                # Conditional mean: mu + Sigma_12 / var_lnrichness (lnrichness - mu_lnrichness)
                lnobs_given_lnrichness_mean = np.delete(lnM_obs_given_lnzeta_mean, pos['richness'], axis=1) + np.delete(covmat_lnM[:,pos['richness'],:], pos['richness'], axis=1)/richness_var_lnM[:,None] * (lnM_richness - lnM_obs_given_lnzeta_mean[:,pos['richness']])[:,None]
                # Conditional (co-)variance = Sigma_11 - Sigma_12 / var_lnrichness * Sigma_21
                if len(obsnames)==2:
                    lnobs_given_lnrichness_mean = lnobs_given_lnrichness_mean[:,0]
                    var = covmat_lnM[:,pos[obsnames_norichness[0]],pos[obsnames_norichness[0]]] - covmat_lnM[:,0,1]**2/richness_var_lnM

        if ('WLDES' in obsnames)|('WLHST' in obsnames)|('WLMegacam' in obsnames):
            lnM_lensing = self.rng.normal(lnobs_given_lnrichness_mean, np.sqrt(var))
            obs = scaling_relations.mass2obs(obsnames_norichness[0], np.exp(lnM_lensing), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, self.catalog['SPT_ID'][dataID])
            # Draw from HST large-scale structure scatter
            if 'WLHST' in obsnames:
                idx = (self.HSTcalib['SPT_ID']==self.catalog['SPT_ID'][dataID]).nonzero()[0]
                std = msqrt(self.HSTcalib['LSS'][idx]**2 + self.HSTcalib['LOS'][idx]**2)
                r_min = ndtr((self.WL.M_arr[0]-obs)/std)
                r_max = ndtr((self.WL.M_arr[-1]-obs)/std)
                r = r_min + (r_max-r_min)*self.rng.random(len(obs))
                obs+= erfinv(2*r-1)*std*msqrt(2)
            # Lensing likelihood
            lnlike[pos[obsnames_norichness[0]]] = interp1d(self.WL.lnM_arr, self.catalog['lnp_Mwl'][dataID], fill_value='extrapolate')(np.log(obs))

        return lnlike


    ############################################################################
    def get_P_obs_xi_zetadraw(self, obsnames, dataID):
        """Returns P(obs|xi,z,p)"""
        # Basic setup
        N_obs = len(obsnames)
        z_cluster = self.catalog['REDSHIFT'][dataID]
        dlnM_dlnobs = np.array([scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames])
        pos = {}
        for o,obs in enumerate(obsnames):
            pos[obs] = o
        # Mass given zeta
        lnM_zeta, zeta_lnweights = self.get_lnM_zeta_given_xi(dataID)
        SZscatter_lnM = self.scaling['Dsz'] * dlnM_dlnobs[pos['zeta']]
        lnM = self.draw_lnm_given_lnzeta(lnM_zeta, SZscatter_lnM)
        # Sometimes there are failures when mean obs is very unlikely
        if np.all(np.isinf(lnM)):
            return 0.
        idx = np.isfinite(lnM)
        if not np.all(idx):
            lnM = lnM[idx]
            lnM_zeta = lnM_zeta[idx]
            zeta_lnweights = zeta_lnweights[idx]
        # Weight with mass function
        mass_lnweights = self.get_mass_function_lnweights(z_cluster, lnM)
        # Normalization P(xi)
        Pxi = np.mean(np.exp(zeta_lnweights + mass_lnweights))
        # Covariance matrix in ln-mass [draw, N_obs, N_obs]
        covmat = self.get_covmat_obs(obsnames)
        Jacobian = dlnM_dlnobs[:,None]*dlnM_dlnobs[None,:]
        covmat_lnM = covmat * Jacobian * np.ones((len(lnM),N_obs,N_obs))
        if 'WLDES' in obsnames:
            DES_scatter = scaling_relations.WLscatter('main', np.exp(lnM), z_cluster, self.scaling)
            covmat_lnM[:,pos['WLDES'],:]*= DES_scatter[:,None]
            covmat_lnM[:,:,pos['WLDES']]*= DES_scatter[:,None]
        elif 'WLHST' in obsnames:
            scatter = self.scaling['DWL_HST'][self.catalog['SPT_ID'][dataID]]
            covmat_lnM[:,pos['WLHST'],:]*= scatter
            covmat_lnM[:,:,pos['WLHST']]*= scatter
        # Get mean and (co-)variance of follow-up observables conditioned on lnM_zeta
        lnM_obs_given_lnzeta_mean, var, obsnames_nozeta = self.get_conditional(lnM, lnM_zeta, covmat_lnM, pos, 'zeta', obsnames)
        # Likelihood of follow-up observables
        lnlike_obs = self.get_lnlike_obs(dataID, obsnames_nozeta, lnM_obs_given_lnzeta_mean, var)
        # Final likelihood
        lnweights = zeta_lnweights + mass_lnweights + np.sum(lnlike_obs, axis=0)
        # Let's accept two invalid draws (and discard them)
        lnweights = lnweights[np.isfinite(lnweights)]
        if len(lnweights)<Ndraw-32:
            return 0.
        # If we cannot even compute the largest weight we're doomed
        if np.exp(np.amax(lnweights))==0:
            return 0.
        # Lots of potential warnings ahead...
        with np.errstate(all='ignore'):
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
        return like
