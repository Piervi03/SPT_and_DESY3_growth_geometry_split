import numpy as np
import os
import imp
import warnings
from math import sqrt as msqrt
from multiprocessing import Pool
from astropy.table import Table
from scipy.special import erfinv, ndtr, log_ndtr
from scipy import integrate, signal
from scipy.interpolate import RectBivariateSpline
import cosmo, Mconversion_concentration, scaling_relations

# Integration method
xi_err = 1.
zeta_draw_method = 'truncated_lognorm_norm'


cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
GETPULL = False
Ndraw = 2**13
ndtr_m5 = ndtr(-5)
ndtr_p4 = ndtr(4)
ln2pi = np.log(2.*np.pi)

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
                 z_cl_min_max, lambda_min, richness_scatter_model,
                 SPT_survey_fields, SPTcatalogfile,
                 HSTcalibfile,
                 NPROC,
                 get_stacked_DES=False):

        self.NPROC = NPROC
        self.get_stacked_DES = get_stacked_DES
        self.todo = todo
        self.mcType = mcType
        self.z_cl_min_max = z_cl_min_max
        lambda_min = lambda_min
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
        HMF_in = HMF['dNdlnM']
        if np.any(HMF_in==0):
            HMF_in[HMF_in==0] = np.nextafter(0, 1)
        self.lnM_arr = HMF['lnM_arr']
        self.HMF_interp = RectBivariateSpline(HMF['z_arr'], self.lnM_arr, np.log(HMF_in), kx=1, ky=1)

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
                idx = (notNone&(self.catalog['REDSHIFT']<z_mid)&(self.catalog['XI']<xi_mid))
                stack['shear_zloxilo'] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
                stack['DeltaSigma_zloxilo'] = np.mean(self.catalog['DES_DeltaSigma_mean'][idx], axis=0)
                tmp = np.array([x for x in self.catalog['DES_DeltaSigma_data_mean'][idx]])
                stack['DeltaSigma_data_zloxilo'] = np.nanmean(tmp, axis=0)
                idx = (notNone&(self.catalog['REDSHIFT']<z_mid)&(self.catalog['XI']>=xi_mid))
                stack['shear_zloxihi'] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
                stack['DeltaSigma_zloxihi'] = np.mean(self.catalog['DES_DeltaSigma_mean'][idx], axis=0)
                tmp = np.array([x for x in self.catalog['DES_DeltaSigma_data_mean'][idx]])
                stack['DeltaSigma_data_zloxihi'] = np.nanmean(tmp, axis=0)
                idx = (notNone&(self.catalog['REDSHIFT']>=z_mid)&(self.catalog['XI']<xi_mid))
                stack['shear_zhixilo'] = np.mean(self.catalog['DES_shear_profile_mean'][idx], axis=0)
                stack['DeltaSigma_zhixilo'] = np.mean(self.catalog['DES_DeltaSigma_mean'][idx], axis=0)
                tmp = np.array([x for x in self.catalog['DES_DeltaSigma_data_mean'][idx]])
                stack['DeltaSigma_data_zhixilo'] = np.nanmean(tmp, axis=0)
                idx = (notNone&(self.catalog['REDSHIFT']>=z_mid)&(self.catalog['XI']>=xi_mid))
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
        if not (self.z_cl_min_max[0]<self.catalog['REDSHIFT'][i]<self.z_cl_min_max[1]):
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


    def get_mass_function_lnweights(self, z, lnM):
        """Return log-probability of halo mass function
        ln(P(lnM)) = ln(dN/dlnM) at given `z` and array `lnM`."""
        # Scipy.RectBivariateSpline only accepts sorted inputs
        idx = np.argsort(lnM)
        mass_lnweights = np.zeros(len(lnM))
        mass_lnweights[idx] = self.HMF_interp(z, lnM[idx])
        # mass_lnweights-= np.amax(mass_lnweights)
        return mass_lnweights



    ############################################################################
    def get_P_obs_xi_zetadraw(self, obsnames, dataID):
        """Returns P(obs|xi,z,p)"""
        # Mass
        lnM_mean = scaling_relations.obs2lnmass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID]), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, SPTfield=self.SPTsurvey[self.SPT_survey['FIELD']==self.catalog['FIELD'][dataID]])
        scatter_lnM = np.sqrt(self.scaling['Dsz']**2 + xi_err**2/self.catalog['XI'][dataID]**2)/self.scaling['Bsz']
        lnM_min = lnM_mean-4*scatter_lnM
        lnM_max = lnM_mean+3*scatter_lnM
        lnM = lnM_min + (lnM_max-lnM_min)*self.rng.random(Ndraw)
        # zeta
        lnzeta0 = scaling_relations.lnmass2lnobs('zeta', lnM, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology, SPTfield=self.SPTsurvey[self.SPT_survey['FIELD']==self.catalog['FIELD'][dataID]])
        if zeta_draw_method in ['lognorm', 'lognorm_cut']:
            lnzeta = self.rng.normal(lnzeta0, self.scaling['Dsz'])
            if zeta_draw_method=='lognorm':
                zeta_lnweights = 0.
            elif zeta_draw_method=='lognorm_cut':
                zeta_lnweights = np.zeros(Ndraw)
                zeta_lnweights[np.exp(lnzeta)<self.scaling['zeta_min']] = -np.inf
        elif zeta_draw_method in ['truncated_lognorm_nonorm', 'truncated_lognorm_norm']:
            r_min = ndtr((np.log(self.scaling['zeta_min'])-lnzeta0)/self.scaling['Dsz'])
            r_min[r_min<ndtr_m5] = ndtr_m5
            r = r_min + (ndtr_p4-r_min)*self.rng.random(Ndraw)
            lnzeta = erfinv(2*r-1)*msqrt(2)*self.scaling['Dsz'] + lnzeta0
            if zeta_draw_method=='truncated_lognorm_nonorm':
                zeta_lnweights = 0.
            elif zeta_draw_method=='truncated_lognorm_norm':
                zeta_lnweights = log_ndtr((lnzeta0-np.log(self.scaling['zeta_min']))/self.scaling['Dsz'])
        # xi
        xi0 = scaling_relations.zeta2xi(np.exp(lnzeta))
        xi_lnweights = -.5*(self.catalog['XI'][dataID]-xi0)**2/xi_err**2 - .5*ln2pi - np.log(xi_err)
        # Mass function
        mass_lnweights = self.get_mass_function_lnweights(self.catalog['REDSHIFT'][dataID], lnM)
        # Richness
        lnrichness = scaling_relations.lnmass2lnobs('richness', lnM, self.catalog['REDSHIFT'][dataID], self.scaling)
        richness_lnweights = -.5*((np.log(self.catalog['richness'][dataID])-lnrichness)/self.scaling['Drichness'])**2 - np.log(self.catalog['richness'][dataID]*self.scaling['Drichness']) - .5*ln2pi
        # Likelihood
        Pxi = np.mean(np.exp(zeta_lnweights+xi_lnweights+mass_lnweights))
        Pxirichness = np.mean(np.exp(zeta_lnweights+xi_lnweights+mass_lnweights+richness_lnweights))
        like = Pxirichness/Pxi
        return like
