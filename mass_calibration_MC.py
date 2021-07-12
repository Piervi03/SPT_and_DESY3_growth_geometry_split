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

    def __init__(self, todo, method, mcType,
                 surveyCutRedshift, surveyCutRichness,
                 SPT_survey_fields, SPT_doublecounts, SPTcatalogfile,
                 WLsimcalibfile,
                 NPROC):

        self.NPROC = NPROC
        self.todo = todo
        self.method = method
        self.mcType = mcType
        self.surveyCutRedshift = surveyCutRedshift
        self.surveyCutRichness = surveyCutRichness

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
        probability = np.inf
        n = 0
        while np.isinf(probability) | np.isnan(probability):
            if self.method=='multiobsdraw':
                probability = self.get_P_obs_xi_multiobsdraw(obsnames, i)
            elif self.method=='zetadraw':
                probability = self.get_P_obs_xi_zetadraw(obsnames, i)
            n+= 1
            if n==10:
                break

        if (probability<0) | np.isnan(probability):
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
        broader distribution. Correct for this with weights."""
        # Draw lnMwl from ln(P(Mwl))
        WL_cum = integrate.cumtrapz(np.exp(self.catalog['lnp_Mwl'][dataID]-self.WL.lnM_arr), self.WL.lnM_arr)
        WL_cum = np.insert(WL_cum/WL_cum[-1], 0, 0.)
        r = self.rng.random(size=Ndraw)
        lnMwl = np.interp(r, WL_cum, self.WL.lnM_arr)
        # We drew from ln(P(Mwl)) but we want to sample P(Mwl)
        WL_lnweights = lnMwl
        return lnMwl, WL_lnweights


    def draw_lnm_given_lnobs(self, ln_obs, cov_inv):
        """Return draws of ln(mass) given ln(vec(obs)) along with ln(weights)."""
        ln_obs = np.array(ln_obs)
        D = cov_inv.shape[1]
        tmp = ln_obs[None,:,:]*ln_obs[:,None,:]*cov_inv[:,:,None]
        T0 = np.sum(tmp, axis=(0,1))
        T1 = np.sum(ln_obs*np.sum(cov_inv, axis=1)[:,None], axis=0)
        T2 = np.sum(cov_inv)
        mean = T1/T2
        std = 1/np.sqrt(T2)
        r_min = norm.cdf(self.lnM_arr[0], loc=mean, scale=std)
        r_max = norm.cdf(self.lnM_arr[-1], loc=mean, scale=std)
        r = r_min + (r_max-r_min)*self.rng.random(Ndraw)
        ln_m = erfinv(2*r-1)*std*np.sqrt(2) + mean
        # We drew from N(mean, std) so let's undo the normalization
        lnweights = np.log(np.sqrt(2*np.pi)*std)
        # Norm of multi-var Gaussian is sqrt((2*pi)**D * det(cov))
        lnweights-= .5*np.log((2*np.pi)**D /np.linalg.det(cov_inv))
        # Account for T factors
        lnweights-= .5 * (T0 - T1**2/T2)
        return ln_m, lnweights


    def get_mass_function_lnweights(self, z, lnM):
        """Return log-probability of halo mass function
        ln(P(lnM)) = ln(dN/dlnM) at given `z` and array `lnM`."""
        # Scipy.RectBivariateSpline only accepts sorted inputs
        idx = np.argsort(lnM)
        mass_lnweights = np.zeros(len(lnM))
        mass_lnweights[idx] = self.HMF_interp(np.log(z), lnM[idx])
        return mass_lnweights


    def get_P_xi(self, z, lnM_zeta, zeta_lnweights, covmat_lnM, pos, dataID=None):
        """Return P(xi) = \int dM P(xi|M) P(M). `covmat_lnM` must be ordered
        richness-SZ."""
        SZscatter_lnM = np.sqrt(covmat_lnM[pos['zeta'],pos['zeta']])
        r_min = norm.cdf(self.lnM_arr[0], loc=lnM_zeta, scale=SZscatter_lnM)
        r_max = norm.cdf(self.lnM_arr[-1], loc=lnM_zeta, scale=SZscatter_lnM)
        r = r_min + (r_max-r_min)*self.rng.random(Ndraw)
        lnM = erfinv(2*r-1)*SZscatter_lnM*np.sqrt(2) + lnM_zeta
        idx = np.isfinite(lnM)
        if not np.any(idx):
            return 0.
        if self.todo['lambda_min']:
            lnM_lambda_mean = lnM[idx] + covmat_lnM[pos['zeta'],pos['richness']]/covmat_lnM[pos['zeta'],pos['zeta']] * (lnM_zeta-lnM)[idx]
            lnM_lambda_std = np.sqrt(covmat_lnM[pos['richness'],pos['richness']] - covmat_lnM[pos['zeta'],pos['richness']]**2/covmat_lnM[pos['zeta'],pos['zeta']])
            if self.catalog['FIELD'][dataID]=='SPTPOL_500d':
                lambda_min = self.surveyCutRichness['deep'](z)
            else:
                lambda_min = self.surveyCutRichness['shallow'](z)
            lnM_lambda_min = np.log(scaling_relations.obs2mass('richness', lambda_min, z, self.scaling))
            xi_lambdacut_lnweights = np.log(norm.cdf(lnM_lambda_mean, lnM_lambda_min, lnM_lambda_std))
        else:
            xi_lambdacut_lnweights = 0.
        mass_lnweights = self.get_mass_function_lnweights(z, lnM[idx])
        lnweights = zeta_lnweights[idx]+mass_lnweights+xi_lambdacut_lnweights
        mean_lnweights = np.mean(lnweights)
        diff_lnweights = lnweights - mean_lnweights
        with np.errstate(all='ignore'):
            Pxi = np.exp(mean_lnweights) * np.mean(np.exp(diff_lnweights))
        return Pxi


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


    ############################################################################
    def get_P_obs_xi_multiobsdraw(self, obsnames, dataID):
        """Returns P(obs|xi,z,p)"""
        # Basic setup
        N_obs = len(obsnames)
        z_cluster = self.catalog['REDSHIFT'][dataID]
        covmat = self.get_covmat_obs(obsnames)
        dlnM_dlnobs = np.array([scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames])

        # Mass given follow-up observables
        lnM_obs, obs_lnweights = N_obs*[None], N_obs*[None]
        pos = {}
        for o,obs in enumerate(obsnames):
            pos[obs] = o
            if obs in ['WLDES', 'WLHST', 'WLMegacam']:
                lnMwl, obs_lnweights[o] = self.get_Mwl_draws(dataID)
                lnM_obs[o] = np.log(scaling_relations.obs2mass('WLDES', np.exp(lnMwl), z_cluster, self.scaling, self.cosmology, self.catalog['SPT_ID'][dataID]))
                if obs=='WLDES':
                    # Covariance matrix with scatter based on central SZ mass
                    m_fid = scaling_relations.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID]), z_cluster, self.scaling, self.cosmology)
                    DES_scatter = scaling_relations.WLscatter('main', m_fid, z_cluster, self.scaling)
                    covmat[o,:]*= DES_scatter
                    covmat[:,o]*= DES_scatter
                elif obs=='WLHST':
                    scatter = self.scaling['DWL_HST'][self.catalog['SPT_ID'][dataID]]
                    covmat[o,:]*= scatter
                    covmat[:,o]*= scatter
            elif obs=='richness':
                lnM_obs[o] = np.log(scaling_relations.obs2mass('richness', self.catalog['richness'][dataID], z_cluster, self.scaling, self.cosmology))*np.ones(Ndraw)
                obs_lnweights[o] = np.zeros(Ndraw)
            elif obs=='zeta':
                zeta, obs_lnweights[o] = self.get_zeta_draws(self.catalog['XI'][dataID])
                obs_lnweights[o]+= np.log(scaling_relations.zeta2xi(zeta)/zeta**2)
                zeta/= self.thisSPTfield_gamma
                lnM_obs[o] = np.log(scaling_relations.obs2mass('zeta', zeta, z_cluster, self.scaling, self.cosmology))
            else:
                print('to do')
                return 0
            obs_lnweights[o]+= np.log(dlnM_dlnobs[pos[obs]])

        # Covariance matrix in ln-mass
        Jacobian = dlnM_dlnobs[:,None]*dlnM_dlnobs[None,:]
        covmat_lnM = covmat * Jacobian

        # Normalization P(xi)
        Pxi = self.get_P_xi(z_cluster, lnM_obs[pos['zeta']], obs_lnweights[pos['zeta']],
                            covmat_lnM, pos,
                            dataID)
        if Pxi==0:
            return 0.

        # Draw mass given multi-obs covariance
        cov_inv = np.linalg.inv(covmat_lnM)
        lnM, mass_draw_lnweights = self.draw_lnm_given_lnobs(lnM_obs, cov_inv)

        # Sometimes there are failures when mean obs is very unlikely
        if np.all(np.isinf(lnM)):
            return 0.
        idx = np.isfinite(lnM)
        if not np.all(idx):
            lnM = lnM[idx]
            mass_draw_lnweights = mass_draw_lnweights[idx]
            for i in range(N_obs):
                lnM_obs[i] = lnM_obs[i][idx]
                obs_lnweights[i] = obs_lnweights[i][idx]

        # Correct for the fact that we drew from inexact covariance matrix for DES
        if 'WLDES' in obsnames:
            det_obs = np.linalg.det(covmat_lnM)
            # Exact covariance matrix based on mass
            covmat = self.get_covmat_obs(obsnames)*np.ones((len(lnM),N_obs,N_obs))
            DES_scatter = scaling_relations.WLscatter('main', np.exp(lnM), self.catalog['REDSHIFT'][dataID], self.scaling)
            covmat[:,pos['WLDES'],:]*= DES_scatter[:,None]
            covmat[:,:,pos['WLDES']]*= DES_scatter[:,None]
            covmat_lnM = covmat * Jacobian
            cov_inv_m = np.linalg.inv(covmat_lnM)
            # Compute chi2s
            if N_obs==2:
                chi2_obs = multivariate_normal.bivariate_chi2_multivec(lnM-lnM_obs[0],
                                                                       lnM-lnM_obs[1],
                                                                       np.tile(cov_inv, (len(lnM),1,1)))
                chi2_m = multivariate_normal.bivariate_chi2_multivec(lnM-lnM_obs[0],
                                                                     lnM-lnM_obs[1],
                                                                     cov_inv_m)
            elif N_obs==3:
                chi2_obs = multivariate_normal.trivariate_chi2_multivec(lnM-lnM_obs[0],
                                                                        lnM-lnM_obs[1],
                                                                        lnM-lnM_obs[2],
                                                                        np.tile(cov_inv, (len(lnM),1,1)))
                chi2_m = multivariate_normal.trivariate_chi2_multivec(lnM-lnM_obs[0],
                                                                      lnM-lnM_obs[1],
                                                                      lnM-lnM_obs[2],
                                                                      cov_inv_m)
            chi2_lnweights = -.5 * np.log((2*np.pi)**N_obs * np.linalg.det(covmat_lnM)) -.5 * chi2_m
            chi2_lnweights-= -.5 * np.log((2*np.pi)**N_obs * det_obs) - .5 * chi2_obs
        else:
            chi2_lnweights = 0.

        # Mass function
        mass_lnweights = self.get_mass_function_lnweights(z_cluster, lnM)

        # Final likelihood
        lnweights = np.sum(obs_lnweights, axis=0)+mass_draw_lnweights+chi2_lnweights+mass_lnweights
        mean_lnweights = np.mean(lnweights)
        diff_lnweights = lnweights - mean_lnweights
        with np.errstate(all='ignore'):
            Pobsxi = np.exp(mean_lnweights) * np.mean(np.exp(diff_lnweights))
        like = Pobsxi/Pxi
        return like


    ############################################################################
    def get_P_obs_xi_zetadraw(self, obsnames, dataID):
        """Returns P(obs|xi,z,p)"""
        # Basic setup
        N_obs = len(obsnames)
        z_cluster = self.catalog['REDSHIFT'][dataID]
        obsnames_nozeta = obsnames.copy()
        obsnames_nozeta.remove('zeta')
        dlnM_dlnobs = np.array([scaling_relations.dlnM_dlnobs(obs, self.scaling) for obs in obsnames])

        # Mass given follow-up observables
        lnM_obs, obs_lnweights = N_obs*[None], N_obs*[None]
        pos = {}
        for o,obs in enumerate(obsnames):
            pos[obs] = o
            if obs in ['WLDES', 'WLHST', 'WLMegacam']:
                lnMwl, obs_lnweights[o] = self.get_Mwl_draws(dataID)
                lnM_obs[o] = np.log(scaling_relations.obs2mass('WLDES', np.exp(lnMwl), z_cluster, self.scaling, self.cosmology, self.catalog['SPT_ID'][dataID]))
            elif obs=='richness':
                lnM_obs[o] = np.log(scaling_relations.obs2mass('richness', self.catalog['richness'][dataID], z_cluster, self.scaling, self.cosmology))*np.ones(Ndraw)
                obs_lnweights[o] = np.zeros(Ndraw)
            elif obs=='zeta':
                zeta, obs_lnweights[o] = self.get_zeta_draws(self.catalog['XI'][dataID])
                obs_lnweights[o]+= np.log(scaling_relations.zeta2xi(zeta)/zeta**2)
                zeta/= self.thisSPTfield_gamma
                lnM_obs[o] = np.log(scaling_relations.obs2mass('zeta', zeta, z_cluster, self.scaling, self.cosmology))
            else:
                print('to do')
                return 0
            obs_lnweights[o]+= np.log(dlnM_dlnobs[pos[obs]])

        # Draw mass given zeta
        SZscatter_lnM = self.scaling['Dsz'] * dlnM_dlnobs[pos['zeta']]
        lnM = self.draw_lnm_given_lnzeta(lnM_obs[pos['zeta']], SZscatter_lnM)

        # Sometimes there are failures when mean obs is very unlikely
        if np.all(np.isinf(lnM)):
            return 0.
        idx = np.isfinite(lnM)
        if not np.all(idx):
            lnM = lnM[idx]
            for i in range(N_obs):
                lnM_obs[i] = lnM_obs[i][idx]
                obs_lnweights[i] = obs_lnweights[i][idx]

        # Mass function
        mass_lnweights = self.get_mass_function_lnweights(z_cluster, lnM)

        # Weights of optical cleaning
        if self.todo['lambda_min']:
            covmat = self.get_covmat_obs(['zeta', 'richness'])
            covmat_lnM = covmat * dlnM_dlnobs[[pos['zeta'], pos['richness']]][:,None]*dlnM_dlnobs[[pos['zeta'], pos['richness']]][None,:]
            lnM_lambda_mean = lnM + covmat_lnM[0,1]/covmat_lnM[0,0]*(lnM_obs[pos['zeta']]-lnM)
            lnM_lambda_std = np.sqrt(covmat_lnM[1,1] - covmat_lnM[0,1]**2/covmat_lnM[0,0])
            if self.catalog['FIELD'][dataID]=='SPTPOL_500d':
                lambda_min = self.surveyCutRichness['deep'](z_cluster)
            else:
                lambda_min = self.surveyCutRichness['shallow'](z_cluster)
            lnM_lambda_min = np.log(scaling_relations.obs2mass('richness', lambda_min, z_cluster, self.scaling))
            xi_lambdacut_lnweights = np.log(norm.cdf(lnM_lambda_mean, lnM_lambda_min, lnM_lambda_std))
        else:
            xi_lambdacut_lnweights = 0.

        # Normalization P(xi)
        lnweights = obs_lnweights[pos['zeta']] + mass_lnweights + xi_lambdacut_lnweights
        mean_lnweights = np.mean(lnweights)
        diff_lnweights = lnweights - mean_lnweights
        with np.errstate(all='ignore'):
            Pxi = np.exp(mean_lnweights) * np.mean(np.exp(diff_lnweights))
        if Pxi==0:
            return 0.

        # Covariance matrix in ln-mass [draw, N_obs, N_obs]
        covmat = self.get_covmat_obs(obsnames)
        Jacobian = dlnM_dlnobs[:,None]*dlnM_dlnobs[None,:]
        covmat_lnM = covmat * Jacobian * np.ones((len(lnM),N_obs,N_obs))
        if 'WLDES' in obsnames:
            DES_scatter = scaling_relations.WLscatter('main', np.exp(lnM), self.catalog['REDSHIFT'][dataID], self.scaling)
            covmat_lnM[:,pos['WLDES'],:]*= DES_scatter[:,None]
            covmat_lnM[:,:,pos['WLDES']]*= DES_scatter[:,None]
        elif 'WLHST' in obsnames:
            scatter = self.scaling['DWL_HST'][self.catalog['SPT_ID'][dataID]]
            covmat_lnM[:,pos['WLHST'],:]*= scatter
            covmat_lnM[:,:,pos['WLHST']]*= scatter

        # Conditional probability of follow-up observables
        covmat_lnM_zeta = covmat_lnM[:,pos['zeta'],pos['zeta']]
        covmat_lnM_mix = np.delete(covmat_lnM[:,pos['zeta'],:], pos['zeta'], axis=1)

        lnobs_given_lnzeta_mean = (lnM[:,None] + covmat_lnM_mix / covmat_lnM_zeta[:,None] * (lnM_obs[pos['zeta']] - lnM)[:,None]).T
        inv = np.linalg.inv(covmat_lnM)
        lnobs_given_lnzeta_cov_inv = np.delete(np.delete(inv, pos['zeta'], axis=1), pos['zeta'], axis=2)

        if N_obs==2:
            chi2 = (lnM_obs[pos[obsnames_nozeta[0]]]-lnobs_given_lnzeta_mean)[0]**2 * lnobs_given_lnzeta_cov_inv[:,0,0]
        elif N_obs==3:
            chi2 = multivariate_normal.bivariate_chi2_multivec(lnM_obs[pos[obsnames_nozeta[0]]]-lnobs_given_lnzeta_mean[0],
                                                               lnM_obs[pos[obsnames_nozeta[1]]]-lnobs_given_lnzeta_mean[1],
                                                               lnobs_given_lnzeta_cov_inv)
        elif N_obs==4:
            chi2 = multivariate_normal.trivariate_chi2_multivec(lnM_obs[pos[obsnames_nozeta[0]]]-lnobs_given_lnzeta_mean[0],
                                                                lnM_obs[pos[obsnames_nozeta[1]]]-lnobs_given_lnzeta_mean[1],
                                                                lnM_obs[pos[obsnames_nozeta[2]]]-lnobs_given_lnzeta_mean[2],
                                                                lnobs_given_lnzeta_cov_inv)
        P_obs_lnweights = -.5 * chi2 - .5 * np.log((2*np.pi)**(N_obs-1) / np.linalg.det(lnobs_given_lnzeta_cov_inv))

        # Final likelihood
        lnweights = np.sum(obs_lnweights, axis=0) + mass_lnweights + P_obs_lnweights
        mean_lnweights = np.mean(lnweights)
        diff_lnweights = lnweights - mean_lnweights
        with np.errstate(all='ignore'):
            Pobsxi = np.exp(mean_lnweights) * np.mean(np.exp(diff_lnweights))
        like = Pobsxi/Pxi
        if (like<=0.) | np.isinf(like) | np.isnan(like):
            print(self.catalog['SPT_ID'][dataID], 'like', like)
        return like
