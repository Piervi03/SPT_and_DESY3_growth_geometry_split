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
from scipy.stats import norm, lognorm, multivariate_normal
from scipy.special import erfinv
import cosmo, Mconversion_concentration, scaling_relations
import multivariate_normal

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
GETPULL = False
Ndraw = 8192

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlike(*arg)

################################################################################
class MassCalibration:

    def __init__(self, todo, mcType, surveyCutSZ, surveyCutRedshift,
                 SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
                 WLsimcalibfile,
                 NPROC):

        self.NPROC = NPROC
        self.todo = todo
        self.mcType = mcType
        self.surveyCutSZ = surveyCutSZ
        self.surveyCutRedshift = surveyCutRedshift

        # Read input files
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        SPTdata = imp.load_source('SPTdata', SPT_doublecounts)
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        self.catalog = Table.read(SPTcatalogfile)
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration



    ############################################################################
    def lnlike(self, HMF, cosmology, scaling, rng_seed=1328):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        self.rng = np.random.default_rng(rng_seed)
        self.cosmology = cosmology
        self.scaling = scaling
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

        # If likelihood computation failed it returned 0
        if np.count_nonzero(likelihoods)<len_data:
            return -np.inf

        lnlike = np.sum(np.log(likelihoods))

        return lnlike



    ############################################################################
    def clusterlike(self, i):
        """Return multi-wavelength mass-calibration likelihood (no log!) for a
        given cluster (index) by calling get_P_1obs_xi or get_P_2obs_xi or
        returning 1 if no follow-up data is available."""
        # t0 = time.time()
        name = self.catalog['SPT_ID'][i]

        ##### Do we actually want this guy? (some clusters in SPT-SZ are at field boundaries)
        if (name,self.catalog['FIELD'][i]) in self.SPTdoubleCount:
            return 1.
        if not self.surveyCutSZ[0]<self.catalog['XI'][i]<self.surveyCutSZ[1] or not self.surveyCutRedshift[0]<self.catalog['REDSHIFT'][i]<self.surveyCutRedshift[1]:
            return 1

        ##### Check if follow-up is available
        nobs = 0
        obsnames = []
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
        if nobs==1:
            probability = self.get_P_1obs_xi(obsnames[0], i)

        elif nobs==2:
            probability = self.get_P_2obs_xi(obsnames, i)

        else:
            raise ValueError(name,"has",nobs,"follow-up observables. I don't know what to do!")

        if (probability<0) | (np.isnan(probability)):
            return 0
            # raise ValueError("P(obs|xi) =", probability, name)

        # print(name, obsnames, probability, time.time()-t0)
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
        # Percent point function (scipy stats is too slow
        xi0 = erfinv(2*r-1)*np.sqrt(2) + xi-1
        zeta = scaling_relations.xi2zeta(xi0)
        # Ratio of normals (scipy.stats.norm too slow)
        zeta_weights = np.exp(-.5 * ((xi0-xi)**2 - (xi0-(xi-1))**2))
        return zeta, zeta_weights


    def get_Mwl_draws(self, dataID):
        """Draw Mwl from lensing likelihood. We draw from ln(P(Mwl)) to get a
        broader distribution. Correct for this with weights."""
        # Draw lnMwl from ln(P(Mwl))
        WL_interp = InterpolatedUnivariateSpline(self.WL.lnM_arr, self.catalog['lnp_Mwl'][dataID], k=1)
        WL_cum = integrate.cumtrapz(self.catalog['lnp_Mwl'][dataID], self.WL.lnM_arr)
        WL_cum = np.insert(WL_cum/WL_cum[-1], 0, 0.)
        r = self.rng.random(size=Ndraw)
        lnMwl = np.interp(r, WL_cum, self.WL.lnM_arr)
        lnP = WL_interp(lnMwl)
        # We drew from ln(P(Mwl)) but we want to sample P(Mwl)
        WL_weights = np.exp(lnP)/np.abs(lnP)
        return lnMwl, WL_weights


    def draw_lnm_given_lnobs(self, ln_obs, cov_inv):
        """Return draws of ln(mass) given ln(vec(obs))."""
        D = cov_inv.shape[1]
        if D==2:
            T1 = ln_obs[0]*np.sum(cov_inv[0,:]) + ln_obs[1]*np.sum(cov_inv[1,:])
            T2 = cov_inv[0,0] + cov_inv[1,1] + 2*cov_inv[0,1]
        elif D==3:
            T1 = ln_obs[0]*np.sum(cov_inv[0,:]) + ln_obs[1]*np.sum(cov_inv[1,:]) +  ln_obs[2]*np.sum(cov_inv[2,:])
            T2 = cov_inv[0,0] + cov_inv[1,1] + cov_inv[2,2] + 2*cov_inv[0,1]  + 2*cov_inv[0,2] + 2*cov_inv[1,2]
        mean = T1/T2
        std = 1/np.sqrt(T2)
        r_min = norm.cdf(self.lnM_arr[0], loc=mean, scale=std)
        r_max = norm.cdf(self.lnM_arr[-1], loc=mean, scale=std)
        r = r_min + (r_max-r_min)*self.rng.random(Ndraw)
        ln_m = erfinv(2*r-1)*std*np.sqrt(2) + mean
        return ln_m


    ############################################################################
    def get_P_1obs_xi(self, obsname, dataID):
        """Returns P(obs|xi,z,p) for a single type of follow-up data."""
        # Draw zeta w/ weights
        zeta, zeta_weights = self.get_zeta_draws(self.catalog['XI'][dataID])
        lnM_zeta = np.log(scaling_relations.obs2mass('zeta', zeta, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)/self.thisSPTfield_gamma)

        # Mass(observable) and covariance matrix
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = scaling_relations.dlnM_dlnobs(obsname, self.scaling)
        Jacobian = np.array([[dlnM_dlnobs**2, dlnM_dlnobs*dlnM_dlnzeta],
                             [dlnM_dlnobs*dlnM_dlnzeta, dlnM_dlnzeta**2]])
        if obsname=='richness':
            lnM_obs = np.log(scaling_relations.obs2mass(obsname, self.catalog['richness'][dataID], self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
            obs_weights = 1.
            covmat = self.scaling['cov_richness_SZ']
        elif obsname in ['WLDES', 'WLHST', 'WLMegacam']:
            lnMwl, obs_weights = self.get_Mwl_draws(dataID)
            lnM_obs = np.log(scaling_relations.obs2mass('WLDES', np.exp(lnMwl), self.catalog['REDSHIFT'][dataID], self.scaling))
            if obsname=='WLDES':
                cov_base = np.array([[1, self.scaling['rhoSZWL']*self.scaling['Dsz']],
                                     [self.scaling['rhoSZWL']*self.scaling['Dsz'], self.scaling['Dsz']**2]])
                # Covariance matrix based on central SZ mass
                m_fid = scaling_relations.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID]), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
                DES_scatter = scaling_relations.WLscatter('main', m_fid, self.catalog['REDSHIFT'][dataID], self.scaling)
                covmat = cov_base * np.array([[DES_scatter**2, DES_scatter], [DES_scatter, 1.]])
            elif obsname=='WLHST':
                covmat = self.scaling['cov_HST_SZ_%s'%self.catalog['SPT_ID'][dataID]]

        # Convert ln-observable covmat into covmat in ln-mass
        covmat_lnM = covmat * Jacobian

        # Draw mass given multi-obs covariance     
        cov_inv = np.linalg.inv(covmat_lnM)
        lnM = self.draw_lnm_given_lnobs([lnM_obs, lnM_zeta], cov_inv)

        # Correct for the fact that we drew from inexact covariance matrix for DES
        if obsname=='WLDES':
            # chi2 using SZ-based covariance matrix
            chi2_obs = multivariate_normal.bivariate_chi2_multivec(lnM-lnM_obs,
                                                                   lnM-lnM_zeta,
                                                                   np.tile(cov_inv, (Ndraw,1,1)))
            # Covariance matrix based on mass
            DES_scatter = scaling_relations.WLscatter('main', np.exp(lnM), self.catalog['REDSHIFT'][dataID], self.scaling)
            covmat = cov_base * np.array([[DES_scatter**2, DES_scatter], [DES_scatter, 1.]])
            covmat_lnM = covmat * Jacobian
            cov_inv = np.linalg.inv(covmat_lnM)
            # chi2 using mass-based covariance matrix
            chi2_m = multivariate_normal.bivariate_chi2_multivec(lnM-lnM_obs,
                                                                 lnM-lnM_zeta,
                                                                 cov_inv)
            chi2_weights = np.exp(-.5 * (chi2_m-chi2_obs))
        else:
            chi2_weights = 1.

        # Normalized mass function
        lndN_dlnM = self.HMF_interp(np.log(self.catalog['REDSHIFT'][dataID]), self.lnM_arr)[0]
        P_m_norm = np.sum(np.diff(self.lnM_arr) * np.exp(.5*(lndN_dlnM[1:]+lndN_dlnM[:-1])))
        # Mass function weights
        idx = np.argsort(lnM)
        mass_weights = np.zeros(Ndraw)
        mass_weights[idx] = np.exp(self.HMF_interp(np.log(self.catalog['REDSHIFT'][dataID]), lnM[idx]) - lnM[idx]) / P_m_norm

        # Final likelihood
        likelihood = np.average(zeta_weights*obs_weights*chi2_weights*mass_weights)
        return likelihood


    ############################################################################
    def get_P_2obs_xi(self, obsnames, dataID):
        """Returns P(obs1, obs2|xi,z,p) for two types of follow-up data (e.g.,
        WL and X-ray)."""
        # Draw zeta w/ weights
        zeta, zeta_weights = self.get_zeta_draws(self.catalog['XI'][dataID])
        lnM_zeta = np.log(scaling_relations.obs2mass('zeta', zeta, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)/self.thisSPTfield_gamma)

        # Mass(observable) 
        lnM_obs = 2*[None]
        for i in range(2):
            if obsnames[i]=='Yx':
                raise NotImplementedError("2d mass calibration for %s not ready"%obsnames[i])
                obsmeas[i], obserr[i] = self.catalog['Yx_fid'][dataID], self.catalog['Yx_err'][dataID]
            elif obsnames[i]=='Mgas':
                raise NotImplementedError("2d mass calibration for %s not ready"%obsnames[i])
                obsmeas[i], obserr[i] = self.catalog['Mg_fid'][dataID], self.catalog['Mg_err'][dataID]
            elif obsnames[i]=='disp':
                raise NotImplementedError("2d mass calibration for %s not ready"%obsnames[i])
                obsmeas[i], obserr[i] = self.catalog['veldisp'][dataID], self.scaling['DdispN']/self.catalog['Ngal'][dataID]
            elif obsnames[i]=='richness':
                lnM_obs[i] = np.log(scaling_relations.obs2mass('richness', self.catalog['richness'][dataID], self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
                obs_weights = 1.
            elif obsnames[i] in ['WLDES', 'WLHST', 'WLMegacam']:
                lnMwl, obs_weights = self.get_Mwl_draws(dataID)
                lnM_obs[i] = np.log(scaling_relations.obs2mass('WLDES', np.exp(lnMwl), self.catalog['REDSHIFT'][dataID], self.scaling))

        # Covariance matrix
        dlnM_dlnzeta = scaling_relations.dlnM_dlnobs('zeta', self.scaling)
        dlnM_dlnobs = [scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames]
        Jacobian = np.array([[dlnM_dlnobs[0]**2,             dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[0]*dlnM_dlnzeta],
                                 [dlnM_dlnobs[0]*dlnM_dlnobs[1], dlnM_dlnobs[1]**2,             dlnM_dlnobs[1]*dlnM_dlnzeta],
                                 [dlnM_dlnobs[0]*dlnM_dlnzeta,   dlnM_dlnobs[1]*dlnM_dlnzeta,   dlnM_dlnzeta**2]])
        if obsnames[1]=='richness':
            if obsnames[0]=='WLDES':
                # Prerequisits for covariance matrix in mass space
                cov_base = np.array([[1, self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['rhoSZWL']*self.scaling['Dsz']],
                                     [self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['Drichness']**2, self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness']],
                                     [self.scaling['rhoSZWL']*self.scaling['Dsz'], self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness'], self.scaling['Dsz']**2]])
                # Covariance matrix based on central SZ mass
                m_fid = scaling_relations.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID]), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
                DES_scatter = scaling_relations.WLscatter('main', m_fid, self.catalog['REDSHIFT'][dataID], self.scaling)
                covmat = cov_base * np.array([[DES_scatter**2, DES_scatter, DES_scatter],
                                              [DES_scatter, 1, 1],
                                              [DES_scatter, 1, 1]])
            elif obsnames[0]=='WLHST':
                covmat = self.scaling['cov_HST_richness_SZ_%s'%self.catalog['SPT_ID'][dataID]]

            # Convert ln-observable covmat into covmat in ln-mass
            covmat_lnM = covmat * Jacobian

            # Draw masses
            cov_inv = np.linalg.inv(covmat_lnM)
            lnM = self.draw_lnm_given_lnobs([lnM_obs[0], lnM_obs[1], lnM_zeta], cov_inv)

        # Correct for the fact that we drew from inexact covariance matrix for DES
        if obsnames[0]=='WLDES':
            # chi2 using SZ-based covariance matrix
            chi2_obs = multivariate_normal.trivariate_chi2_multivec(lnM-lnM_obs[0],
                                                                    lnM-lnM_obs[1],
                                                                    lnM-lnM_zeta,
                                                                    np.tile(cov_inv, (Ndraw,1,1)))
            # Covariance matrix based on mass
            DES_scatter = scaling_relations.WLscatter('main', np.exp(lnM), self.catalog['REDSHIFT'][dataID], self.scaling)
            covmat = cov_base * np.array([DES_scatter**2, DES_scatter, DES_scatter,
                                          DES_scatter, np.ones(len(DES_scatter)), np.ones(len(DES_scatter)),
                                          DES_scatter, np.ones(len(DES_scatter)), np.ones(len(DES_scatter))]).T.reshape(len(DES_scatter),3,3)
            covmat_lnM = covmat * Jacobian
            cov_inv = np.linalg.inv(covmat_lnM)
            # chi2 using mass-based covariance matrix
            chi2_m = multivariate_normal.trivariate_chi2_multivec(lnM-lnM_obs[0],
                                                                  lnM-lnM_obs[1],
                                                                  lnM-lnM_zeta,
                                                                  cov_inv)
            chi2_weights = np.exp(-.5 * (chi2_m-chi2_obs))
        else:
            chi2_weights = 1.

        # Normalized mass function
        lndN_dlnM = self.HMF_interp(np.log(self.catalog['REDSHIFT'][dataID]), self.lnM_arr)[0]
        P_m_norm = np.sum(np.diff(self.lnM_arr) * np.exp(.5*(lndN_dlnM[1:]+lndN_dlnM[:-1])))
        # Mass function weights
        idx = np.argsort(lnM)
        mass_weights = np.zeros(Ndraw)
        mass_weights[idx] = np.exp(self.HMF_interp(np.log(self.catalog['REDSHIFT'][dataID]), lnM[idx]) - lnM[idx]) / P_m_norm

        # Final likelihood
        likeli = np.average(zeta_weights*obs_weights*chi2_weights*mass_weights)
        return likeli
