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

import cosmo, Mconversion_concentration, scaling_relations
import multivariate_normal

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
GETPULL = False
Ndraw = 1000

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
    def lnlike(self, HMF, cosmology, scaling):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        self.rng = np.random.default_rng(1328)
        self.cosmology = cosmology
        self.scaling = scaling
        self.xi_min = scaling_relations.zeta2xi(self.scaling['zeta_min'])

        ##### Set up interpolation for HMF
        HMF_in = HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(HMF['z_arr'][1:]), np.log(HMF['M_arr']), np.log(HMF_in), kx=1, ky=1)

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


    def get_zeta_M_draws(self, xi, z):
        """Draw zetas from `xi` and masses from mass function at redshift `z`.
        Return ln(zeta), Marr, and weights."""
        # Draw zeta from xi
        xi0 = self.rng.normal(loc=xi-1, scale=1, size=Ndraw)
        bad_idx = (xi0<self.xi_min).nonzero()[0]
        for i in bad_idx:
            while xi0[i]<self.xi_min:
                xi0[i] = self.rng.normal(loc=xi-1, scale=1, size=1)
        zeta = scaling_relations.xi2zeta(xi0)
        weights = norm.pdf(scaling_relations.zeta2xi(zeta), xi)/norm.pdf(scaling_relations.zeta2xi(zeta), xi-1)
        # Draw mass
        lnM0 = np.log(scaling_relations.zeta2mass(scaling_relations.xi2zeta(xi), z, self.scaling, self.cosmology))
        lnMmin = lnM0 - 4*self.scaling['Dsz']
        lnMmax = lnM0 + 3*self.scaling['Dsz']
        lnMarr = np.linspace(lnMmin, lnMmax, Ndraw)
        Marr = np.exp(lnMarr)
        P_M = np.exp(self.HMF_interp(np.log(z), lnMarr))[0]/Marr
        weights*= P_M/Marr
        return np.log(zeta), Marr, weights



    ############################################################################
    def get_P_1obs_xi(self, obsname, dataID):
        """Returns P(obs|xi,z,p) for a single type of follow-up data."""
        # Draw zeta and mass w/ weights
        lnzeta, Marr, weights = self.get_zeta_M_draws(self.catalog['XI'][dataID], self.catalog['REDSHIFT'][dataID])
        # Observable arrays given Marr
        lnzeta_arr = np.log(self.thisSPTfield_gamma * scaling_relations.mass2obs('zeta', Marr, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
        lnobs_arr = np.log(scaling_relations.mass2obs(obsname, Marr, self.catalog['REDSHIFT'][dataID], self.scaling))
        # Multi-obs covariance matrix and chi2
        if obsname=='richness':
            cov = np.array([[self.scaling['Drichness']**2, self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness']],
                            [self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness'], self.scaling['Dsz']**2]])
            cov_inv = np.linalg.inv(cov)
            lnobsmeas = np.log(self.catalog['richness'][dataID])
        chi2 = multivariate_normal.bivariate_chi2_multivec(lnobs_arr-lnobsmeas,
                                                           lnzeta_arr-lnzeta,
                                                           np.tile(cov_inv, (Ndraw,1,1)))
        weights*= np.exp(-.5 * chi2)
        # Final likelihood
        likelihood = np.average(weights)
        return likelihood


    ############################################################################
    def get_P_2obs_xi(self, obsnames, dataID):
        """Returns P(obs1, obs2|xi,z,p) for two types of follow-up data (e.g.,
        WL and X-ray)."""
        # Draw zeta and mass w/ weights
        lnzeta, Marr, weights = self.get_zeta_M_draws(self.catalog['XI'][dataID], self.catalog['REDSHIFT'][dataID])
        # Observable arrays given Marr
        obsArr, lnobsArr, obsmeas, obserr = [], [], np.empty(2), np.empty(2)
        for i in range(2):
            if obsnames[i]=='Yx':
                obsmeas[i], obserr[i] = self.catalog['Yx_fid'][dataID], self.catalog['Yx_err'][dataID]
            elif obsnames[i]=='Mgas':
                obsmeas[i], obserr[i] = self.catalog['Mg_fid'][dataID], self.catalog['Mg_err'][dataID]
            elif obsnames[i]=='disp':
                obsmeas[i], obserr[i] = self.catalog['veldisp'][dataID], self.scaling['DdispN']/self.catalog['Ngal'][dataID]
            elif obsnames[i]=='richness':
                obsmeas[i] = self.catalog['richness'][dataID]
            elif obsnames[i]=='WLMegacam':
                LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
            elif obsnames[i]=='WLHST':
                LSSnoise = self.WLcalib['HSTsim'][self.catalog['SPT_ID'][dataID]]['obs_scatter']
            elif obsnames[i]=='WLDES':
                LSSnoise = 0.
            obsArrTemp = scaling_relations.mass2obs(obsnames[i], Marr, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
            # Account for radial dependence for X-ray observables
            if obsnames[i] in ('Mgas', 'Yx'):
                correction = self.conversion_factor_Xray_obs_r500ref(dataID)
                obsArrTemp*= correction
            obsArr.append( obsArrTemp )
            lnobsArr.append( np.log(obsArrTemp) )
        lnzeta_arr = np.log(self.thisSPTfield_gamma * scaling_relations.mass2obs('zeta', Marr, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
        # Draw lnMwl from ln(P(Mwl))
        WL_interp = InterpolatedUnivariateSpline(self.WL.lnM_arr, self.catalog['lnp_Mwl'][dataID], k=1)
        WL_cum = integrate.cumtrapz(self.catalog['lnp_Mwl'][dataID], self.WL.lnM_arr)
        WL_cum = np.insert(WL_cum/WL_cum[-1], 0, 0.)
        r = self.rng.random(size=Ndraw)
        lnMwl = np.interp(r, WL_cum, self.WL.lnM_arr)
        lnP = WL_interp(lnMwl)
        # We drew from ln(P(Mwl)) but we want to sample P(Mwl)
        weights*= np.exp(lnP)/np.abs(lnP)
        # Multi-obs covariance matrix and chi2
        if obsnames[1]=='richness':
            cov_base = np.array([[1, self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['rhoSZWL']*self.scaling['Dsz']],
                                 [self.scaling['rhoWLrichness']*self.scaling['Drichness'], self.scaling['Drichness']**2, self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness']],
                                 [self.scaling['rhoSZWL']*self.scaling['Dsz'], self.scaling['rhoSZrichness']*self.scaling['Dsz']*self.scaling['Drichness'], self.scaling['Dsz']**2]])
            DES_scatter = scaling_relations.WLscatter('main', Marr, self.catalog['REDSHIFT'][dataID], self.scaling)
            covmat = cov_base * np.array([DES_scatter**2, DES_scatter, DES_scatter,
                                          DES_scatter, np.ones(len(DES_scatter)), np.ones(len(DES_scatter)),
                                          DES_scatter, np.ones(len(DES_scatter)), np.ones(len(DES_scatter))]).T.reshape(len(DES_scatter),3,3)
            cov_inv = np.linalg.inv(covmat)
            chi2 = multivariate_normal.trivariate_chi2_multivec(lnobsArr[0]-lnMwl,
                                                                lnobsArr[1]-np.log(self.catalog['richness'][dataID]),
                                                                lnzeta_arr-lnzeta,
                                                                cov_inv)
            weights*= np.exp(-.5 * chi2)
        else:
            Px = norm.pdf(obsmeas[1], obsArr[1], obserr[1])
            Pobs = Pwl[:,None] * Px[None,:]
            likeli = np.trapz(np.trapz(dP_dobs01*Pobs, obsArr[1], axis=1), obsArr[0])

        likeli = np.average(weights)
        return likeli
