from __future__ import division
import numpy as np
import os
import imp
from multiprocessing import Pool
from astropy.table import Table

import scipy.ndimage
import scipy.special as ss
from scipy import integrate, signal
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm, multivariate_normal

from cosmosis.datablock import option_section
import cosmo, lensing, Mconversion_concentration, scaling_relations

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
getpull = False

THRESHOLD = 1e-8

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlike(*arg)

################################################################################
class MassCalibration:

    def __init__(self, options):
        ##### Config parameters
        self.todo = {}
        for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
            self.todo[opt[2:]] = options.get_bool(option_section, opt)
        self.scaling = {}
        for opt in ['SZmPivot', 'XraymPivot', 'richmPivot', 'YXPARAM']:
            self.scaling[opt] = options.get_double(option_section, opt)
        self.mcType = options.get_string(option_section, 'mcType')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        self.NPROC = options.get_int(option_section, 'NPROC')
        # SPT survey
        SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
        assert os.path.isfile(SPT_survey_fields), "SPT survey table does not exist"
        self.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
        # Double counted clusters
        SPT_doublecounts = options.get_string(option_section, 'SPT_doublecounts')
        SPTdata = imp.load_source('SPTdata', SPT_doublecounts)
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        # Cluster catalog
        SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        ##### WL simulation calibration
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

        # Weak lensing
        if self.todo['WL']:
            self.WL = lensing.SPTlensing(options, self.catalog)


    ############################################################################
    def lnlike(self, block):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        ##### Extract from datablock
        self.cosmology = {
            'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
            'Omega_b': block.get_double('cosmological_parameters', 'Omega_b'),
            'h': block.get_double('cosmological_parameters', 'hubble')/100,
            'ns': block.get_double('cosmological_parameters', 'n_s'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa'),
            'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
        # SZ
        for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Bsz2', 'Csz2', 'Esz', 'DszM']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # X-ray
        for p in ['Ax', 'Bx', 'Cx', 'Dx', 'Ex', 'dlnMg_dlnr']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # WL
        for p in ['WLbias', 'WLscatter', 'HSTbias', 'HSTscatterLSS', 'MegacamScatterLSS', 'DESscatterLSS']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # Richness
        for p in ['Arichness', 'Brichness', 'Crichness', 'Drichness']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # dispersion
        for p in ['Adisp', 'Bdisp', 'Cdisp', 'Ddisp0', 'DdispN']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # Correlation coefficients
        for p in ['rhoSZWL', 'rhoWLX', 'rhoXdisp']:
            self.scaling[p] = block.get_double('mor_parameters', p)
        # Covariance matrices
        self.covmat = {}
        for c in ['cov_X_SZ', 'cov_Megacam_SZ', 'cov_DES_SZ', 'cov_rich_SZ', 'cov_Megacam_X_SZ', 'cov_DES_X_SZ']:
            self.covmat[c] = block.get_double_array_nd('scaling', c)

        # Halo mass function
        self.HMF = {
            'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
            'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
            'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
        self.HMF['len_z'] = len(self.HMF['z_arr'])

        ##### WL: Precompute array of angular diameter distances
        if self.todo['WL']:
            self.WL.get_dAs(self.cosmology)

        ##### Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in))

        ##### Initialize mass-concentration relation class (for WL and dispersions)
        if self.todo['WL'] or self.todo['veldisp']:
            self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology)

        ##### Compute interpolation table for M500-M200
        if self.todo['WL'] or self.todo['veldisp']:
            z_arr = np.linspace(.1, 2, 20)
            M500 = np.logspace(np.log10(self.HMF['M_arr'][0]), np.log10(self.HMF['M_arr'][-1]), 20)
            M200 = np.array([np.array([self.MCrel.MDelta_to_M200(m, 500., z) for m in M500]) for z in z_arr])
            self.lnM500_to_lnM200 = RectBivariateSpline(z_arr, np.log(M500), np.log(M200))
        if self.todo['veldisp']:
            z_arr = np.linspace(.1, 2, 20)
            M200 = np.logspace(np.log10(np.amin(M200)), np.log10(np.amax(M200)), 20)
            M500 = np.array([np.array([self.MCrel.M200_to_MDelta(m, 500., z) for m in M200]) for z in z_arr])
            self.lnM200_to_lnM500 = RectBivariateSpline(z_arr, np.log(M200), np.log(M500))

        ##### Evaluate the individual likelihoods
        len_data = len(self.catalog['SPT_ID'])

        if self.NPROC==0:
            # Iterate through cluster list
            likelihoods = np.array([self.clusterlike(i) for i in range(len_data)])
        else:
            # Launch a multiprocessing pool and get the likelihoods
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len_data, range(len_data))
            likelihoods = pool.map(unwrap_self_f, argin)
            pool.close()

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
        name = self.catalog['SPT_ID'][i]

        ##### Do we actually want this guy? (some clusters in SPT-SZ are at field boundaries)
        if (name,self.catalog['FIELD'][i]) in self.SPTdoubleCount: return 1.
        if not self.surveyCutSZ[0]<self.catalog['XI'][i]<self.surveyCutSZ[1] or not self.surveyCutRedshift[0]<self.catalog['REDSHIFT'][i]<self.surveyCutRedshift[1]: return 1

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
                # bias = bSim + bMassModel + (bN(z)+bShearCal)
                self.scaling['bWL_HST'] = self.WLcalib['HSTsim'][name][0] + self.scaling['WLbias']*self.catalog['WLdata'][i]['massModelErr'] + self.scaling['HSTbias']*self.catalog['WLdata'][i]['zDistShearErr']
                # lognormal scatter
                self.scaling['DWL_HST'] = self.WLcalib['HSTsim'][name][2] + self.scaling['WLscatter']*self.WLcalib['HSTsim'][name][3]
                cov = [[self.scaling['DWL_HST']**2, self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST']],
                    [self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST'], self.scaling['Dsz']**2]]
                if np.linalg.det(cov)<THRESHOLD:
                    return 0.
                self.covmat['cov_HST_SZ'] = cov

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
        self.thisSPTfield_gamma = self.SPT_survey['GAMMA'][self.SPT_survey['FIELD']==self.catalog['FIELD'][i]]

        #####
        if nobs==1:
            if obsnames[0] in ('Yx', 'Mgas'):
                covmat = self.covmat['cov_X_SZ']
            elif obsnames[0]=='WLMegacam':
                covmat = self.covmat['cov_Megacam_SZ']
            elif obsnames[0]=='WLDES':
                covmat = self.covmat['cov_DES_SZ']
            elif obsnames[0]=='richness':
                covmat = self.covmat['cov_rich_SZ']
            elif obsnames[0]=='WLHST':
                covmat = self.covmat['cov_HST_SZ']
            probability = self.get_P_1obs_xi(obsnames[0], i, covmat)

        elif nobs==2:
            if 'disp' in obsnames:
                if self.scaling['rhoXdisp']==0:
                    probability = self.get_P_1obs_xi(obsnames[0], i) * self.get_P_1obs_xi(obsnames[1], i)
                else:
                    probability = self.get_P_2obs_xi(obsnames[:2], i, 'Yxdisp')
            else:
                if 'WLMegacam' in obsnames:
                    covmat = self.covmat['cov_Megacam_X_SZ']
                elif 'WLDES' in obsnames:
                    covmat = self.covmat['cov_DES_X_SZ']
                elif 'WLHST' in obsnames:
                    covname = 'cov_HST_X_SZ'
                    cov = [[self.scaling['DWL_HST']**2, self.scaling['rhoWLX']*self.scaling['DWL_HST']*self.scaling['Dx'], self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST']],
                    [self.scaling['rhoWLX']*self.scaling['DWL_HST']*self.scaling['Dx'], self.scaling['Dx']**2, self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx']],
                    [self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST'], self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx'], self.scaling['Dsz']**2]]
                    if np.linalg.det(cov)<THRESHOLD:
                        return 0.
                    self.covmat[cov_HST_X_SZ] = cov
                if self.scaling['rhoWLX']==0:
                    probability = self.get_P_1obs_xi(obsnames[0], i) * self.get_P_1obs_xi(obsnames[1], i)
                else:
                    probability = self.get_P_2obs_xi(obsnames[:2], i, covmat)

        else:
            raise ValueError(name,"has",nobs,"follow-up observables. I don't know what to do!")

        if (probability<0) | (np.isnan(probability)):
            return 0
            # raise ValueError("P(obs|xi) =", probability, name)

        # print name, obsnames, probability
        return probability




    ############################################################################
    def get_P_1obs_xi(self, obsname, dataID, covmat):
        """Returns P(obs|xi,z,p) for a single type of follow-up data."""

        ##### Get the follow-up observable, obsintr is used for setting up mass range
        if obsname=='Yx':
            obsmeas, obsintr, obserr = self.catalog['Yx_fid'][dataID], self.scaling['Dx'], self.catalog['Yx_err'][dataID]
        elif obsname=='Mgas':
            obsmeas, obsintr, obserr = self.catalog['Mg_fid'][dataID], self.scaling['Dx'], self.catalog['Mg_err'][dataID]
        elif obsname=='disp':
            Dsigma = self.scaling['Ddisp0'] + self.scaling['DdispN']/self.catalog['Ngal'][dataID]
            cov = [[Dsigma**2, self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma],
                [self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma, self.scaling['Dsz']**2]]
            covmat = cov, np.linalg.det(cov)
            obsmeas, obserr, obsintr = self.catalog['veldisp'][dataID], Dsigma, Dsigma
        elif obsname=='richness':
            obsmeas, obserr, obsintr = self.catalog['richness'][dataID], self.catalog['richness_err'][dataID], (self.scaling['Drichness']**2 + 1/((1-self.scaling['Drichness'])*self.catalog['richness'][dataID]))**.5
        elif obsname=='WLMegacam':
            LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_Megacam']*self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), .3, self.scaling['DWL_Megacam']
        elif obsname=='WLHST':
            LSSnoise = self.WLcalib['HST_LSS'][0] + self.scaling['HSTscatterLSS'] * self.WLcalib['HST_LSS'][1]
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_HST']*self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), .3, self.scaling['DWL_HST']
        elif obsname=='WLDES':
            LSSnoise = self.WLcalib['DES_LSS'][0] + self.scaling['DESscatterLSS'] * self.WLcalib['DES_LSS'][1]
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_DES']*self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), .3, self.scaling['DWL_DES']

        ##### Define reasonable mass range
        # xi -> M(xi)
        xi_minmax = np.array([max(2.6,self.catalog['XI'][dataID]-5), self.catalog['XI'][dataID]+3])
        M_xi_minmax = self.obs2mass('zeta', scaling_relations.xi2zeta(xi_minmax)/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
        if M_xi_minmax[0]>self.HMF['M_arr'][-1]:
            print "cluster mass exceeds HMF mass range", self.catalog['SPT_ID'][dataID],\
                M_xi_minmax[0], self.HMF['M_arr'][-1]
            return 0

        # obs: prediction
        lnobs0 = np.log(self.mass2obs(obsname, self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
        SZscatterobs = scaling_relations.dlnM_dlnobs('zeta', self.scaling) / scaling_relations.dlnM_dlnobs(obsname, self.scaling, self.cosmology, self.scaling['SZmPivot'], self.catalog['REDSHIFT'][dataID]) * self.scaling['Dsz']
        intrscatter = (SZscatterobs**2 + obsintr**2)**.5
        obsthminmax = np.exp(np.array([lnobs0-5.*intrscatter, lnobs0+3.5*intrscatter]))
        M_obsth_minmax = self.obs2mass(obsname, obsthminmax, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
        # obs: measurement
        if obsname=='richness':
            obsmeasminmax = np.amax((.1,obsmeas-3*obserr)), obsmeas+3*obserr
        elif obsname in ('Mgas', 'Yx'):
            obsmeasminmax = np.amax((.1, obsmeas-3*obserr)), obsmeas+3*obserr
        else:
            obsmeasminmax = np.exp(np.log(obsmeas)-4*obserr), np.exp(np.log(obsmeas)+3*obserr)
        M_obsmeas_minmax = self.obs2mass(obsname, np.array(obsmeasminmax), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)

        ##### Define grid in mass
        Mmin, Mmax = min(M_xi_minmax[0], M_obsth_minmax[0], M_obsmeas_minmax[0]), max(M_xi_minmax[1], M_obsth_minmax[1], M_obsmeas_minmax[1])
        Mmin, Mmax = max(.5*Mmin, self.HMF['M_arr'][0]), min(Mmax, self.HMF['M_arr'][-1])
        lenObs = 54
        M_obsArr = np.logspace(np.log10(Mmin), np.log10(Mmax), lenObs)

        ##### Observable arrays
        lnzeta_arr = np.log(self.thisSPTfield_gamma * self.mass2obs('zeta', M_obsArr, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
        xi_arr = scaling_relations.zeta2xi(np.exp(lnzeta_arr))
        obsArr = self.mass2obs(obsname, M_obsArr, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)

        ##### Add radial dependence for X-ray observables
        if obsname in ('Mgas','Yx'):
            # Angular diameter distances in current and reference cosmology [Mpc]
            dA = cosmo.dA(self.catalog['REDSHIFT'][dataID], self.cosmology)/self.cosmology['h']
            dAref = cosmo.dA(self.catalog['REDSHIFT'][dataID], cosmologyRef)/cosmologyRef['h']
            # R500 [kpc]
            rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['REDSHIFT'][dataID], self.cosmology)**2
            r500 = 1000 * (3*M_obsArr/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
            # r500 in reference cosmology [kpc]
            r500ref = r500 * dAref/dA
            # Xray observable at fiducial r500...
            obsArr*= (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
            # ... corrected to reference cosmology
            obsArr*= (dAref/dA)**2.5

        lnobsArr = np.log(obsArr)

        ##### HMF array for convolution
        M_HMF_arr = M_obsArr

        ##### Convert self.HMF to dN/(dlnzeta dlnobs) = dN/dlnM * dlnM/dlnzeta * dlnM/dlnobs
        # This only matter if dlnM/dlnobs is mass-dependent, as for dispersions
        dN_dlnzeta_dlnobs = np.exp(self.HMF_interp(np.log(self.catalog['REDSHIFT'][dataID]), np.log(M_HMF_arr)))[0]
        if obsname=='disp':
            dN_dlnzeta_dlnobs*= scaling_relations.dlnM_dlnobs(obsname, self.scaling, self.cosmology, M_HMF_arr, self.catalog['REDSHIFT'][dataID])

        ##### HMF on 2D observable grid
        HMF_2d_in = np.zeros((lenObs, lenObs))
        np.fill_diagonal(HMF_2d_in, dN_dlnzeta_dlnobs)

        ##### 2D convolution with correlated scatter [lnobs,lnzeta]
        pos = np.empty((lenObs,lenObs,2))
        pos[:,:,0], pos[:,:,1] = np.meshgrid(lnobsArr, lnzeta_arr, indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(lnobsArr[27], lnzeta_arr[27]), cov=covmat)
        HMF_2d = signal.fftconvolve(HMF_2d_in, kernel, mode='same')

        # set to 0 if zeta<2
        HMF_2d[:,np.where(lnzeta_arr<np.log(2.))] = 0.

        # Set small negative values to zero (FFT noise)
        if np.any(HMF_2d<-1e-7):
            if np.abs(np.amin(HMF_2d))/np.amax(HMF_2d)>1e-6:
                print "HMF_2d has negative entries:",np.amin(HMF_2d), np.amax(HMF_2d)
        HMF_2d[np.where(HMF_2d<0)] = 0.

        # Safety check
        if np.all(HMF_2d==0.):
            print self.catalog['SPT_ID'][dataID],'HMF_2d is zero, det',np.linalg.det(covmat),self.scaling['Dsz'],obsintr,self.scaling['rhoSZX']
            return 0.

        ##### dN/(dxi dlnobs) = dN/(dlnzeta dlnobs) * dlnzeta/dxi [lnobs,xi]
        HMF_2d*= scaling_relations.dlnzeta_dxi(xi_arr)[None,:]

        #### Convolve with xi measurement error [lnobs]
        dP_dlnobs = np.trapz(HMF_2d * norm.pdf(self.catalog['XI'][dataID], xi_arr[None,:], 1.), xi_arr, axis=1)


        ##### Evaluate likelihood
        if obsname=='richness':
            # Convolve with uncorrelated part of intrinsic scatter
            integrand = dP_dlnobs[None,:] * norm.pdf(lnobsArr[:,None], lnobsArr[None,:], 1/obsArr[None,:]**.5)
            dP_drichness_Poisson = np.trapz(integrand, lnobsArr, axis=1)
            # dP/dobs = dP/dlnobs * dlnobs/dobs = dP/dlnobs /obs
            dP_dobs = dP_drichness_Poisson/obsArr
            # normalize
            dP_dobs/= np.trapz(dP_dobs, obsArr)
            # Likelihood is integral over P(obs)*P(predicted)
            likeli = np.trapz(dP_dobs*norm.pdf(obsArr, obsmeas, obserr), obsArr)

        else:
            ##### dP/dobs = dP/dlnobs * dlnobs/dobs = dP/dlnobs /obs
            dP_dobs = dP_dlnobs/obsArr
            # normalize
            dP_dobs/= np.trapz(dP_dobs, obsArr)

            ##### WL
            if obsname in ('WLHST', 'WLMegacam', 'WLDES'):
                # Concolve with Gaussian LSS scatter
                if LSSnoise>0.:
                    integrand = dP_dobs[None,:] * norm.pdf(obsArr[:,None], obsArr[None,:], LSSnoise)
                    dP_dobs = np.trapz(integrand, obsArr, axis=1)
                    dP_dobs/= np.trapz(dP_dobs, obsArr)
                # P(Mwl) from data
                Pwl = self.WL.like(self.catalog, dataID, obsArr, self.cosmology, self.MCrel, self.lnM500_to_lnM200)
                # Get likelihood
                likeli = np.trapz(Pwl*dP_dobs, obsArr)

            ##### dispersion
            elif obsname=='disp':
                likeli = np.interp(obsmeas, obsArr, dP_dobs)

            ##### X-ray
            else:
                # Get likelihood
                likeli = np.trapz(dP_dobs*norm.pdf(obsmeas, obsArr, obserr), obsArr)

                if getpull:
                    integrand = dP_dobs[None,:] * norm.pdf(obsArr[:,None], obsArr[None,:], obserr)
                    dP_dobs_obs = np.trapz(integrand, obsArr, axis=1)
                    dP_dobs_obs/= np.trapz(dP_dobs_obs,obsArr)
                    cumtrapz = integrate.cumtrapz(dP_dobs_obs,obsArr)
                    perc = np.interp(obsmeas, obsArr[1:], cumtrapz)
                    print self.catalog['SPT_ID'][dataID], '%.4f %.4f %.4f %.4e'%(self.catalog['XI'][dataID], self.catalog['REDSHIFT'][dataID], obsmeas, 2**.5 * ss.erfinv(2*perc-1))

        if ((likeli<0)|(np.isnan(likeli))):
            print self.catalog['SPT_ID'][dataID], obsname, likeli
            #np.savetxt(self.catalog['SPT_ID'][dataID],np.transpose((obsArr, dP_dobs)))
            return 0.

        #np.savetxt(self.catalog['SPT_ID'][dataID]+obsname,np.transpose((obsArr, dP_dobs)))

        return likeli



    ############################################################################
    def get_P_2obs_xi(self, obsnames, dataID, covmat):
        """Returns P(obs1, obs2|xi,z,p) for two types of follow-up data (e.g.,
        WL and X-ray)."""
        ##### Get observables, obsintr is used for setting up mass range
        obsmeas, obserr, obsintr = np.empty(2), np.empty(2), np.empty(2)
        for i in range(2):
            if obsnames[i]=='Yx':
                obsmeas[i], obsintr[i], obserr[i] = self.catalog['Yx_fid'][dataID], self.scaling['Dx'], self.catalog['Yx_err'][dataID]
            elif obsnames[i]=='Mgas':
                obsmeas[i], obsintr[i], obserr[i] = self.catalog['Mg_fid'][dataID], self.scaling['Dx'], self.catalog['Mg_err'][dataID]
            elif obsnames[i]=='disp':
                Dsigma = self.scaling['Ddisp0'] + self.scaling['DdispN']/self.catalog['Ngal'][dataID]
                obsmeas[i], obserr[i], obsintr[i] = self.catalog['veldisp'][dataID], Dsigma, Dsigma
            elif obsnames[i]=='richness':
                obsmeas[i], obserr[i], obsintr[i] = self.catalog['richness'][dataID], self.catalog['richness_err'][dataID], (self.scaling['Drichness']**2 + 1/((1-self.scaling['Drichness'])*self.catalog['richness'][dataID]))**.5
            elif obsnames[i]=='WLMegacam':
                LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_Megacam']*self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), .3, self.scaling['DWL_Megacam']
            elif obsnames[i]=='WLHST':
                LSSnoise = self.WLcalib['HST_LSS'][0] + self.scaling['HSTscatterLSS'] * self.WLcalib['HST_LSS'][1]
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_HST']*self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), .3, self.scaling['DWL_HST']
            elif obsnames[i]=='WLDES':
                LSSnoise = self.WLcalib['DES_LSS'][0] + self.scaling['DESscatterLSS'] * self.WLcalib['DES_LSS'][1]
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_DES']*self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), .3, self.scaling['DWL_DES']

        ##### Special case for dispersions
        if ('Yx' in obsnames) and ('disp' in obsnames):
            cov = [[Dsigma**2, self.scaling['rhoXdisp']*Dsigma*self.scaling['Dx'], self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma],
                [self.scaling['rhoXdisp']*Dsigma*self.scaling['Dx'], self.scaling['Dx']**2, self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx']],
                [self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma, self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx'], self.scaling['Dsz']**2]]
            covmat = cov


        ##### Define reasonable mass range
        # xi -> M(xi)
        xi_minmax = np.array((np.amax((2.6,self.catalog['XI'][dataID]-5)), self.catalog['XI'][dataID]+3))
        M_xi_minmax = self.obs2mass('zeta', scaling_relations.xi2zeta(xi_minmax)/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
        if M_xi_minmax[0]>self.HMF['M_arr'][-1]:
            print "cluster mass exceeds HMF mass range", self.catalog['SPT_ID'][dataID],\
                M_xi_minmax[0], self.HMF['M_arr'][-1]
            return 0

        M_obsminmax = []
        for i in range(2):
            # obs: prediction
            lnobs0 = np.log(self.mass2obs(obsnames[i], self.obs2mass('zeta', scaling_relations.xi2zeta(self.catalog['XI'][dataID])/self.thisSPTfield_gamma, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology), self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
            if obsnames[i]=='disp': SZscatterobs = scaling_relations.dlnM_dlnobs('zeta', self.scaling)/3.*self.scaling['Dsz']
            else: SZscatterobs = scaling_relations.dlnM_dlnobs('zeta', self.scaling)/scaling_relations.dlnM_dlnobs(obsnames[i], self.scaling)*self.scaling['Dsz']
            intrscatter = (SZscatterobs**2 + obsintr[i]**2)**.5
            obsthminmax = np.exp(np.array((lnobs0-5*intrscatter, lnobs0+3.5*intrscatter)))
            # obs: measurement
            if obsnames[i]=='richness':
                obsmeasminmax = np.amax((.1, obsmeas[i]-3*obserr[i])), obsmeas[i]+3*obserr[i]
            elif obsnames[i] in ('Mgas', 'Yx'):
                obsmeasminmax = np.amax((.1, obsmeas[i]-3*obserr[i])), obsmeas[i]+3*obserr[i]
            else:
                obsmeasminmax = np.exp(np.log(obsmeas[i])-4*obserr[i]), np.exp(np.log(obsmeas[i])+3*obserr[i])
            # put together
            obsminmax = np.array((min(obsthminmax[0],obsmeasminmax[0]), max(obsthminmax[1],obsmeasminmax[1])))
            M_obsminmax.append(self.obs2mass(obsnames[i], obsminmax, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))

        ##### Define grid in mass
        Mmin, Mmax = min(M_xi_minmax[0],M_obsminmax[0][0],M_obsminmax[1][0]), max(M_xi_minmax[1],M_obsminmax[0][1],M_obsminmax[1][1])
        Mmin, Mmax = max(.5*Mmin, self.HMF['M_arr'][0]), min(Mmax, self.HMF['M_arr'][-1])
        lenObs = 54
        M_obsArr = np.logspace(np.log10(Mmin), np.log10(Mmax), lenObs)
        M_HMF_arr = M_obsArr


        ##### Observable arrays
        lnzeta_arr = np.log(self.thisSPTfield_gamma * self.mass2obs('zeta', M_obsArr, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology))
        xi_arr = scaling_relations.zeta2xi(np.exp(lnzeta_arr))
        obsArr, lnobsArr = [], []
        for i in range(2):
            obsArrTemp = self.mass2obs(obsnames[i], M_obsArr, self.catalog['REDSHIFT'][dataID], self.scaling, self.cosmology)
            ##### Add radial dependence for X-ray observables
            if obsnames[i] in ('Mgas','Yx'):
                # Angular diameter distances in current and reference cosmology [Mpc]
                dA = cosmo.dA(self.catalog['REDSHIFT'][dataID], self.cosmology)/self.cosmology['h']
                dAref = cosmo.dA(self.catalog['REDSHIFT'][dataID], cosmologyRef)/cosmologyRef['h']
                # R500 [kpc]
                rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['REDSHIFT'][dataID], self.cosmology)**2
                r500 = 1000 * (3*M_obsArr/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
                # r500 in reference cosmology [kpc]
                r500ref = r500 * dAref/dA
                # Xray observable at rFid...
                obsArrTemp*= (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
                # ... corrected to reference cosmology
                obsArrTemp*= (dAref/dA)**2.5
            obsArr.append( obsArrTemp )
            lnobsArr.append( np.log(obsArrTemp) )


        ##### HMF to dN/(dlnzeta dlnobs0 dlnobs1) = dN/dlnM * dlnM/dlnzeta * dlnM/dlnobs0 * dlnM/dlnobs1
        # This only matter if dlnM/dlnobs is mass-dependent, as for dispersions
        dN_dlnzeta_dlnobs = np.exp(self.HMF_interp(np.log(self.catalog['REDSHIFT'][dataID]), np.log(M_HMF_arr)))[0]
        if 'disp' in obsnames:
            dN_dlnzeta_dlnobs*= scaling_relations.dlnM_dlnobs('disp', self.scaling, self.cosmology, M_HMF_arr, self.catalog['REDSHIFT'][dataID])

        ##### HMF on 3D observable grid [lnobs0,lnobs1,lnzeta]
        HMF_3d_in = np.zeros((lenObs, lenObs, lenObs))
        np.fill_diagonal(HMF_3d_in, dN_dlnzeta_dlnobs)

        ##### 3D convolution with correlated scatter
        # kernel is min(lenObs, max(20bins, +/-5sigma))
        Nbin_obs0 = int(np.amin((lenObs, 10 * np.amax((2, covmat[0][0]**.5/(lnobsArr[0][1] - lnobsArr[0][0]))))))
        Nbin_obs1 = int(np.amin((lenObs, 10 * np.amax((2, covmat[1][1]**.5/(lnobsArr[1][1] - lnobsArr[1][0]))))))
        Nbin_zeta = int(np.amin((lenObs, 10 * np.amax((2, covmat[-1][-1]**.5/(lnzeta_arr[1] - lnzeta_arr[0]))))))
        pos = np.empty((Nbin_obs0, Nbin_obs1, Nbin_zeta, 3))
        pos[:,:,:,0], pos[:,:,:,1], pos[:,:,:,2] = np.meshgrid(lnobsArr[0][:Nbin_obs0], lnobsArr[1][:Nbin_obs1], lnzeta_arr[:Nbin_zeta], indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(lnobsArr[0][int(Nbin_obs0/2)], lnobsArr[1][int(Nbin_obs1/2)], lnzeta_arr[int(Nbin_zeta/2)]), cov=covmat)
        HMF_3d = signal.fftconvolve(HMF_3d_in, kernel, mode='same')

        # set to 0 if zeta<2
        HMF_3d[:,:,np.where(lnzeta_arr<np.log(2.))] = 0.
        # Safety check
        if np.any(HMF_3d<-1e-6):
            print np.amin(HMF_3d), np.amax(HMF_3d)
            print self.catalog['SPT_ID'][dataID],'HMF_3d<0, det',np.linalg.det(covmat),self.scaling['Dsz'],obsintr
        HMF_3d[np.where(HMF_3d<0)] = 0
        if np.all(HMF_3d==0.):
            print self.catalog['SPT_ID'][dataID],'HMF_3d<=0, det',np.linalg.det(covmat),self.scaling['Dsz'],obsintr
            return 0

        ##### dN/(dxi dlnobs) = dN/(dlnzeta dlnobs) * dlnzeta/dxi [lnobs0][lnobs1][xi]
        HMF_3d*= scaling_relations.dlnzeta_dxi(xi_arr)[None,None,:]

        #### Convolve with xi measurement error [lnobs0][lnobs1]
        dP_dlnobs = np.trapz(HMF_3d * norm.pdf(self.catalog['XI'][dataID], xi_arr[None,None,:], 1.), xi_arr, axis=2)

        ##### Go to linear space [obs0][obs1]
        dP_dobs01 = dP_dlnobs/obsArr[0][:,None]/obsArr[1][None,:]

        ##### P0
        dP_dobs0 = np.trapz(dP_dobs01, obsArr[1], axis=1)
        dP_dobs0/= np.trapz(dP_dobs0, obsArr[0])

        if obsnames[0] in ('WLHST', 'WLMegacam', 'WLDES'):
            # Concolve with Gaussian LSS scatter
            if LSSnoise>0.:
                integrand = dP_dobs0[None,:] * norm.pdf(obsArr[0][:,None], obsArr[0][None,:], LSSnoise)
                dP_dobs0 = np.trapz(integrand, obsArr[0], axis=1)
                dP_dobs0/= np.trapz(dP_dobs0, obsArr[0])
            # P(Mwl) from data
            Pobs = self.WL.like(self.catalog, dataID, obsArr[0], self.cosmology, self.MCrel, self.lnM500_to_lnM200)
        else: print "not ready!"

        likeli0 = np.trapz(dP_dobs0*Pobs, obsArr[0])

        ##### P1 (Yx)
        dP_dobs1 = np.trapz(dP_dobs01, obsArr[0], axis=0)

        # Normalize (in principe, multiply with dlnX/dlnXfid, but this is mass-independent)
        dP_dobs1/= np.trapz(dP_dobs1, obsArr[1])
        likeli1 = np.trapz(dP_dobs1*norm.pdf(obsmeas[1], obsArr[1], obserr[1]), obsArr[1])

        #np.savetxt(self.catalog['SPT_ID'][dataID]+'_3d'+obsnames[0],np.transpose((obsArr[0], dP_dobs0)))
        #np.savetxt(self.catalog['SPT_ID'][dataID]+'_3d'+obsnames[1],np.transpose((obsArr[1], dP_dobs1)))


        ##### Probability
        likeli = likeli0*likeli1

        return likeli
