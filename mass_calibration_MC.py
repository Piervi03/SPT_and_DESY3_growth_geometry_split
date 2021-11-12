from __future__ import division, print_function
import numpy as np
import os
import imp
from multiprocessing import Pool
from astropy.table import Table
import time
import scipy.special as ss
from scipy import integrate, signal
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.stats import norm
from scipy.special import erfinv
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
                 surveyCutRedshift, surveyCutRichness,
                 SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
                 HSTcalibfile,
                 NPROC):

        self.NPROC = NPROC
        self.todo = todo
        self.mcType = mcType
        self.surveyCutRedshift = surveyCutRedshift
        self.surveyCutRichness = surveyCutRichness

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
        r_min = norm.cdf(self.xi_min, loc=xi-1, scale=1)
        r = r_min + (1-r_min)*self.rng.random(Ndraw)
        # Percent point function (scipy stats is too slow)
        xi0 = erfinv(2*r-1)*np.sqrt(2) + xi-1
        zeta = scaling_relations.xi2zeta(xi0)
        # Ratio of normals
        zeta_lnweights = -.5 * ((xi0-xi)**2 - (xi0-(xi-1))**2)
        return zeta, zeta_lnweights


    def get_Mwl_draws(self, dataID):
        """Draw Mwl from lensing likelihood. We draw from P(Mwl)/Mwl to get a
        broader distribution and to extend to lower mass. Correct for this with
        weights."""
        # Draw lnMwl from P(Mwl)/Mwl = exp(ln(PMwl)-ln(Mwl))
        WL_cum = integrate.cumtrapz(np.exp(self.catalog['lnp_Mwl'][dataID]-self.WL.lnM_arr), self.WL.lnM_arr)
        WL_cum = np.insert(WL_cum/WL_cum[-1], 0, 0.)
        r = self.rng.random(size=Ndraw)
        lnMwl = np.interp(r, WL_cum, self.WL.lnM_arr)
        # We drew from P(Mwl)/Mwl but we need to sample P(Mwl)
        WL_lnweights = lnMwl
        # LSS noise for HST
        if self.catalog['WLdata'][dataID]['datatype']=='HST':
            mean = np.exp(lnMwl)
            idx = (self.HSTcalib['SPT_ID']==self.catalog['SPT_ID'][dataID]).nonzero()[0]
            std = np.sqrt(self.HSTcalib['LSS'][idx]**2 + self.HSTcalib['LOS'][idx]**2)
            r_min = norm.cdf(self.WL.M_arr[0], loc=mean, scale=std)
            r_max = norm.cdf(self.WL.M_arr[-1], loc=mean, scale=std)
            r = r_min + (r_max-r_min)*self.rng.random(Ndraw)
            lnMwl = np.log(erfinv(2*r-1)*std*np.sqrt(2) + mean)
        return lnMwl, WL_lnweights


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
        r_min = norm.cdf(self.lnM_arr[0], loc=lnM_zeta, scale=SZscatter_lnM)
        r_max = norm.cdf(self.lnM_arr[-1], loc=lnM_zeta, scale=SZscatter_lnM)
        r = r_min + (r_max-r_min)*self.rng.random(len(lnM_zeta))
        lnM = erfinv(2*r-1)*SZscatter_lnM*np.sqrt(2) + lnM_zeta
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


    def get_lnlike_obs(self, dataID, obsnames, lnM_obs):
        """Return likelihood of observable(s) given mass."""
        lnlike = len(obsnames)*[None]
        for o,obsname in enumerate(obsnames):
            obs = scaling_relations.mass2obs(obsname, np.exp(lnM_obs[o]), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
            if obsname=='richness':
                lnlike[o] = -.5*(self.catalog['richness'][dataID]-obs)**2/obs - .5*np.log(2*np.pi*obs)
                if self.todo['lambda_min']:
                    if self.catalog['FIELD'][dataID]=='SPTPOL_500d':
                        lambda_min = self.surveyCutRichness['deep'](self.catalog['REDSHIFT'][dataID])
                    else:
                        lambda_min = self.surveyCutRichness['shallow'](self.catalog['REDSHIFT'][dataID])
                    with np.errstate(all='ignore'):
                        lnP_lambda_gtr_lambda_min = np.log(norm.cdf(obs, lambda_min, np.sqrt(obs)))
                    lnlike[o]-= lnP_lambda_gtr_lambda_min
            elif obsname in ['WLDES', 'WLHST', 'WLMegacam']:
                if obsname=='WLHST':
                    idx = (self.HSTcalib['SPT_ID']==self.catalog['SPT_ID'][dataID]).nonzero()[0]
                    std = np.sqrt(self.HSTcalib['LSS'][idx]**2 + self.HSTcalib['LOS'][idx]**2)
                    r_min = norm.cdf(self.WL.M_arr[0], loc=obs, scale=std)
                    r_max = norm.cdf(self.WL.M_arr[-1], loc=obs, scale=std)
                    r = r_min + (r_max-r_min)*self.rng.random(len(obs))
                    obs+= erfinv(2*r-1)*std*np.sqrt(2)
                WL_interp = InterpolatedUnivariateSpline(np.log(self.WL.M_arr), self.catalog['lnp_Mwl'][dataID], k=1)
                lnlike[o] = WL_interp(np.log(obs))
        return lnlike


    ############################################################################
    def get_P_obs_xi_zetadraw(self, obsnames, dataID):
        """Returns P(obs|xi,z,p)"""
        # Basic setup
        N_obs = len(obsnames)
        z_cluster = self.catalog['REDSHIFT'][dataID]
        obsnames_nozeta = obsnames.copy()
        obsnames_nozeta.remove('zeta')
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
        lnweights = zeta_lnweights + mass_lnweights
        #with np.errstate(all='ignore'):
        Pxi = np.mean(np.exp(lnweights))
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
        # Mean and covariance of conditional probability of follow-up observables
        covmat_lnM_zeta = covmat_lnM[:,pos['zeta'],pos['zeta']]
        covmat_lnM_mix = np.delete(covmat_lnM[:,pos['zeta'],:], pos['zeta'], axis=1)
        lnobs_given_lnzeta_mean = lnM[:,None] + covmat_lnM_mix / covmat_lnM_zeta[:,None] * (lnM_zeta - lnM)[:,None]
        inv = np.linalg.inv(covmat_lnM)
        lnobs_given_lnzeta_cov = np.linalg.inv(np.delete(np.delete(inv, pos['zeta'], axis=1), pos['zeta'], axis=2))
        # Draw follow-up observables
        if N_obs==2:
            lnM_obs = [self.rng.normal(lnobs_given_lnzeta_mean[:,0], np.sqrt(lnobs_given_lnzeta_cov[:,0,0])),]
        else:
            Cho = np.linalg.cholesky(lnobs_given_lnzeta_cov)
            u = self.rng.normal(0, 1, size=lnobs_given_lnzeta_mean.shape)
            r = np.sum(Cho[:,:,:]*u[:,None,:], axis=2)
            lnM_obs = (lnobs_given_lnzeta_mean + r).T
        # Likelihood of follow-up observables
        lnlike_obs = self.get_lnlike_obs(dataID, obsnames_nozeta, lnM_obs)
        # Final likelihood
        lnweights = zeta_lnweights + mass_lnweights + np.sum(lnlike_obs, axis=0)
        # Let's accept two invalid draws (and discard them)
        lnweights = lnweights[np.isfinite(lnweights)]
        if len(lnweights)<Ndraw-2:
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
