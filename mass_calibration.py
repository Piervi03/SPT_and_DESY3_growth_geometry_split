from __future__ import division
import numpy as np
import os
import imp
from multiprocessing import Pool
import scipy.ndimage
import scipy.special as ss
from scipy import integrate
from scipy import interpolate
from scipy import signal
from scipy.stats import norm, lognorm
from scipy.stats import multivariate_normal
from astropy.table import Table

from cosmosis.datablock import option_section
import cosmo, Mconversion_concentration, lensing, Xrayprofile, observablecovmat

DEBUG = False
if DEBUG:
    import time

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1}
getpull = False

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg, **kwarg):
    return MassCalibration.clusterlike(*arg, **kwarg)

################################################################################
class MassCalibration:

    def __init__(self, options):
        ##### Config parameters
        self.todo = {
            'WL': options.get_bool(option_section, 'doWL'),
            'Yx': options.get_bool(option_section, 'doYx'),
            'Mgas': options.get_bool(option_section, 'doMgas'),
            'veldisp': options.get_bool(option_section, 'doveldisp'),
            'richness': options.get_bool(option_section, 'dorichness'),
            }
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        self.XraymPivot = options.get_double(option_section, 'XraymPivot')
        self.richmPivot = options.get_double(option_section, 'richmPivot')
        self.YXPARAM = options.get_string(option_section, 'YXPARAM')
        self.Xdata = options.get_string(option_section, 'Xdata')
        self.mcType = options.get_string(option_section, 'mcType')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        self.NPROC = options.get_int(option_section, 'NPROC')
        self.XrayProfileHandling = options.get_string(option_section, 'XrayProfileHandling')
        assert self.XrayProfileHandling in ('fixed', 'old', 'modelMgasPL'), "invalid XrayProfileHandling"
        ##### SPT survey
        # Data
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        ###### X-ray data analysis mess
        if self.todo['Mgas'] or self.todo['Yx']:
            Xray_profile = Xrayprofile.XrayProfile(options)
            # Using real data or simulated profiles
            if self.XrayProfileHandling!='fixed':
                if self.Xdata=='SPT_XVP':
                    self.catalog['Mg'] = self.catalog['Mg_MM']
                    self.catalog['lnMg_err'] = self.catalog['lnMg_err_MM']
                    if self.todo['Yx']:
                        self.catalog['Tx'] = self.catalog['Tx_MM']
                        self.catalog['lnYx_err'] = self.catalog['lnYx_err_MM']
                elif self.Xdata=='WtG':
                    if self.todo['Mgas']:
                        self.catalog['Mg'] = self.catalog['Mg_AM']
                        self.catalog['lnMg_err'] = self.catalog['lnMg_err_AM']
            # Used for mock tests
            else:
                if self.todo['Yx']:
                    self.catalog['lnYx_err'] = self.catalog['lnYx_err_MM']
                elif todo['Mgas']:
                    self.catalog['lnMg_err'] = self.catalog['lnMg_err_MM']
            if self.XrayProfileHandling!='old':
                Xray_profile.setRef(self.catalog)

        # Survey specs
        self.SPTfieldNames = SPTdata.SPTfieldNames
        self.SPTfieldCorrection = SPTdata.SPTfieldCorrection
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        self.XraySample = SPTdata.XraySample
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
        self.cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
            'Omega_b': block.get_double('cosmological_parameters', 'Omega_b'),
            'h': block.get_double('cosmological_parameters', 'hubble'),
            'ns': block.get_double('cosmological_parameters', 'n_s'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
        self.scaling = {
            # SZ
            'Asz': block.get_double('mor_parameters', 'Asz'),
            'Bsz': block.get_double('mor_parameters', 'Bsz'),
            'Csz': block.get_double('mor_parameters', 'Csz'),
            'Dsz': block.get_double('mor_parameters', 'Dsz'),
            'Bsz2': block.get_double('mor_parameters', 'Bsz2'),
            'Csz2': block.get_double('mor_parameters', 'Csz2'),
            'Esz': block.get_double('mor_parameters', 'Esz'),
            'DszM': block.get_double('mor_parameters', 'DszM'),
            # X-ray
            'Ax': block.get_double('mor_parameters', 'Ax'),
            'Bx': block.get_double('mor_parameters', 'Bx'),
            'Cx': block.get_double('mor_parameters', 'Cx'),
            'Dx': block.get_double('mor_parameters', 'Dx'),
            'Ex': block.get_double('mor_parameters', 'Ex'),
            'slope_MgR': block.get_double('mor_parameters', 'slope_MgR'),
            # WL
            'WLbias': block.get_double('mor_parameters', 'WLbias'),
            'WLscatter': block.get_double('mor_parameters', 'WLscatter'),
            'HSTbias': block.get_double('mor_parameters', 'HSTbias'),
            'HSTscatterLSS': block.get_double('mor_parameters', 'HSTscatterLSS'),
            'MegacamBias': block.get_double('mor_parameters', 'MegacamBias'),
            'MegacamScatterLSS': block.get_double('mor_parameters', 'MegacamScatterLSS'),
            'DESbias': block.get_double('mor_parameters', 'DESbias'),
            'DESscatterLSS': block.get_double('mor_parameters', 'DESscatterLSS'),
            # dispersion
            'Adisp': block.get_double('mor_parameters', 'Adisp'),
            'Bdisp': block.get_double('mor_parameters', 'Bdisp'),
            'Cdisp': block.get_double('mor_parameters', 'Cdisp'),
            'Ddisp0': block.get_double('mor_parameters', 'Ddisp0'),
            'DdispN': block.get_double('mor_parameters', 'DdispN'),
            # Correlation coefficients
            'rhoSZWL': block.get_double('mor_parameters', 'rhoSZWL'),
            'rhoWLX': block.get_double('mor_parameters', 'rhoWLX'),
            'rhoSZX': block.get_double('mor_parameters', 'rhoSZX'),
            'rhoXdisp': block.get_double('mor_parameters', 'rhoXdisp'),
            'rhoSZdisp': block.get_double('mor_parameters', 'rhoSZdisp'),
            }
        # Halo mass function
        self.HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
            'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
            'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
        self.HMF['len_z'] = len(self.HMF['z_arr'])

        ##### Setup stuff for WL
        if self.todo['WL']:
            # Set bias and scatter for Megacam and DES from sim calibration
            # and nuissance parameters
            self.WL.set_scaling(self.scaling)
            # Precompute array of angular diameter distances
            self.WL.get_dAs(self.cosmology)

        ##### Populate and check observable covariance matrices
        self.covmat = {}
        if not observablecovmat.set_covmats(self.todo, self.scaling, self.covmat):
            return -np.inf

        ##### Set up spline interpolation for HMF
        self.HMF_interp = interpolate.interp2d(np.log(self.HMF['M_arr']), np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['dNdlnM'][1:,:]), kind='cubic')

        ##### Get X-ray (old method only)
        if self.XrayProfileHandling=='old':
            Xray_profile.getXray(self.catalog, self.todo, self.cosmology, self.scaling)

        ##### Initialize mass-concentration relation class
        self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology)

        ##### Compute interpolation table for M500-M200
        if self.todo['WL'] or self.todo['veldisp']:
            z_arr = np.linspace(.1, 2, 20)
            M500 = np.logspace(np.log10(self.HMF['M_arr'][0]), np.log10(self.HMF['M_arr'][-1]), 20)
            M200 = np.array([np.array([self.MCrel.MDelta_to_M200(m, 500., z) for m in M500]) for z in z_arr])
            self.lnM500_to_lnM200 = interpolate.interp2d(np.log(M500), z_arr, np.log(M200), kind='cubic')
        if self.todo['veldisp']:
            z_arr = np.linspace(.1, 2, 20)
            M200 = np.logspace(np.log10(np.amin(M200)), np.log10(np.amax(M200)), 20)
            M500 = np.array([np.array([self.MCrel.M200_to_MDelta(m, 500., z) for m in M200]) for z in z_arr])
            self.lnM200_to_lnM500 = interpolate.interp2d(np.log(M200), z_arr, np.log(M500), kind='cubic')

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

        lnlike = np.sum(np.log(likelihoods))

        return lnlike



    ############################################################################
    def clusterlike(self, i):
        """Return multi-wavelength mass-calibration likelihood (no log!) for a
        given cluster (index) by calling get_P_1obs_xi or get_P_2obs_xi or
        returning 1 if no follow-up data is available."""
        if DEBUG:
            t0 = time.time()

        name = self.catalog['SPT_ID'][i]

        ##### Do we actually want this guy? (some clusters in SPT-SZ are at field boundaries)
        if (name,self.catalog['field'][i]) in self.SPTdoubleCount: return 1.
        if not self.surveyCutSZ[0]<self.catalog['xi'][i]<self.surveyCutSZ[1] or not self.surveyCutRedshift[0]<self.catalog['redshift'][i]<self.surveyCutRedshift[1]: return 1

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
                self.scaling['DWL_HST'] = self.WLcalib['HSTsim'][name][2]+self.scaling['WLscatter']*self.WLcalib['HSTsim'][name][3]
                cov = [[self.scaling['DWL_HST']**2, self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST']],
                    [self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST'], self.scaling['Dsz']**2]]
                if np.linalg.det(cov)<observablecovmat.THRESHOLD:
                    return 0.
                self.covmat['WLHST'] = cov

        if self.todo['veldisp'] and self.catalog['veldisp'][i]!=0.:
            nobs+= 1
            obsnames.append('disp')
        if self.todo['Yx'] and name in self.XraySample:
                nobs+= 1
                obsnames.append('Yx')
        if self.todo['Mgas'] and name in self.XraySample:
                nobs+= 1
                obsnames.append('Mgas')
        if self.todo['richness'] and self.catalog['richness'][i]!=0.:
            nobs+= 1
            obsnames.append('richness')
        if nobs==0:
            return 1.

        ##### Set SPT field scaling factor
        self.thisSPTfieldCorrection = self.SPTfieldCorrection[self.SPTfieldNames.index(self.catalog['field'][i])]

        #####
        if nobs==1:
            probability = self.get_P_1obs_xi(obsnames[0], i)

        elif nobs==2:
            # probability = self.get_P_1obs_xi(obsnames[0], i)
            # probability*= self.get_P_1obs_xi(obsnames[1], i)

            if 'disp' in obsnames:
                if self.scaling['rhoXdisp']==0:
                    probability = self.get_P_1obs_xi(obsnames[0], i)*self.get_P_1obs_xi(obsnames[1], i)
                else:
                    probability = self.get_P_2obs_xi(obsnames[:2], i, 'Yxdisp')
            else:
                if 'WLMegacam' in obsnames: covname = 'XrayMegacam'
                elif 'WLDES' in obsnames: covname = 'XrayDES'
                elif 'WLHST' in obsnames:
                    covname = 'XrayHST'
                    cov = [[self.scaling['DWL_HST']**2, self.scaling['rhoWLX']*self.scaling['DWL_HST']*self.scaling['Dx'], self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST']],
                    [self.scaling['rhoWLX']*self.scaling['DWL_HST']*self.scaling['Dx'], self.scaling['Dx']**2, self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx']],
                    [self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST'], self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx'], self.scaling['Dsz']**2]]
                    if np.linalg.det(cov)<observablecovmat.THRESHOLD:
                        return 0.
                    self.covmat[covname] = cov
                if self.scaling['rhoWLX']==0:
                    probability = self.get_P_1obs_xi(obsnames[0], i) * self.get_P_1obs_xi(obsnames[1], i)
                else:
                    probability = self.get_P_2obs_xi(obsnames[:2], i, covname)

        else:
            raise ValueError(name,"has",nobs,"follow-up observables. I don't know what to do!")

        if (probability<0) | (np.isnan(probability)):
            raise ValueError("P(obs|xi) =", probability, name)

        if DEBUG:
            print name, obsnames, probability, time.time()-t0
        return probability




    ############################################################################
    def get_P_1obs_xi(self, obsname, dataID):
        """Returns P(obs|xi,z,p) for a single type of follow-up data."""
        covmat = self.covmat[obsname]

        ##### Get the follow-up observable, obsintr is used for setting up mass range
        if obsname=='Yx':
            obsmeas, obsintr, obserr = self.catalog['XrayRef'][dataID][1], self.scaling['Dx'], self.catalog['lnYx_err'][dataID]
        elif obsname=='Mgas':
            obsmeas, obsintr, obserr = self.catalog['XrayRef'][dataID][1], self.scaling['Dx'], self.catalog['lnMg_err'][dataID]
        elif obsname=='disp':
            Dsigma = self.scaling['Ddisp0'] + self.scaling['DdispN']/self.catalog['Ngal'][dataID]
            cov = [[Dsigma**2, self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma],
                [self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma, self.scaling['Dsz']**2]]
            covmat = cov, np.linalg.det(cov)
            obsmeas, obserr, obsintr = self.catalog['veldisp'][dataID], Dsigma, Dsigma
        elif obsname=='richness':
            obsmeas, obserr, obsintr = self.catalog['richness'][dataID], self.catalog['richness_err'][dataID], (self.scaling['Drichness']**2 + 1/((1-self.scaling['Drichness'])*self.catalog['richness'][dataID]))**.5
        elif obsname=='WLMegacam':
            LSSnoise = self.scaling['MegacamScatterLSS']
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_Megacam']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_Megacam']
        elif obsname=='WLHST':
            LSSnoise = self.scaling['HSTscatterLSS']
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_HST']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_HST']
        elif obsname=='WLDES':
            LSSnoise = self.scaling['DESscatterLSS']
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_DES']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_DES']

        ##### Define reasonable mass range
        # xi -> M(xi)
        xi_minmax = np.array([max(2.6,self.catalog['xi'][dataID]-5), self.catalog['xi'][dataID]+3])
        M_xi_minmax = self.obs2mass('zeta', self.xi2zeta(xi_minmax), self.catalog['redshift'][dataID])
        if M_xi_minmax[0]>self.HMF['M_arr'][-1]:
            print "cluster mass exceeds HMF mass range", self.catalog['SPT_ID'][dataID],\
                M_xi_minmax[0], self.HMF['M_arr'][-1]
            return 0

        # obs: prediction
        lnobs0 = np.log(self.mass2obs(obsname, self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), self.catalog['redshift'][dataID]))
        SZscatterobs = self.dlnM_dlnobs('zeta') / self.dlnM_dlnobs(obsname, self.SZmPivot, self.catalog['redshift'][dataID]) * self.scaling['Dsz']
        intrscatter = (SZscatterobs**2 + obsintr**2)**.5
        obsthminmax = np.exp(np.array([lnobs0-5.*intrscatter, lnobs0+3.5*intrscatter]))
        M_obsth_minmax = self.obs2mass(obsname, obsthminmax, self.catalog['redshift'][dataID])
        # obs: measurement
        if obsname=='richness': lnobsmeasminmax = np.log(np.amin((.1,obsmeas-3*obserr))), np.log(obsmeas+3*obserr)
        else: lnobsmeasminmax = np.log(obsmeas)-4*obserr, np.log(obsmeas)+3*obserr
        M_obsmeas_minmax = self.obs2mass(obsname, np.exp(np.array(lnobsmeasminmax)), self.catalog['redshift'][dataID])

        ##### Define grid in mass
        Mmin, Mmax = min(M_xi_minmax[0], M_obsth_minmax[0], M_obsmeas_minmax[0]), max(M_xi_minmax[1], M_obsth_minmax[1], M_obsmeas_minmax[1])
        Mmin, Mmax = max(.5*Mmin, self.HMF['M_arr'][0]), min(Mmax, self.HMF['M_arr'][-1])
        lenObs = 54
        M_obsArr = np.logspace(np.log10(Mmin), np.log10(Mmax), lenObs)

        ##### Observable arrays
        lnzeta_arr = np.log(self.mass2obs('zeta', M_obsArr, self.catalog['redshift'][dataID]))
        xi_arr = self.zeta2xi(np.exp(lnzeta_arr))
        obsArr = self.mass2obs(obsname, M_obsArr, self.catalog['redshift'][dataID])

        ##### Add radial dependence for X-ray observables
        if obsname in ('Mgas','Yx') and self.XrayProfileHandling=='modelMgasPL':
            # Angular diameter distances in current and reference cosmology [Mpc]
            dA = cosmo.dA(self.catalog['redshift'][dataID], self.cosmology)/self.cosmology['h']
            dAref = cosmo.dA(self.catalog['redshift'][dataID], cosmologyRef)/cosmologyRef['h']
            # R500 [kpc]
            rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['redshift'][dataID], self.cosmology)**2
            r500 = 1000 * (3*M_obsArr/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
            # r500 in reference cosmology [kpc]
            r500ref = r500 * dAref/dA
            # Xray observable at rFid
            obsFid = obsArr * (self.catalog['XrayRef'][dataID][0]/r500ref)**self.scaling['slope_MgR']
            # X-ray observable at rFid, corrected to reference cosmology
            obsArr = obsFid * (dAref/dA)**2.5

        lnobsArr = np.log(obsArr)

        ##### HMF array for convolution
        M_HMF_arr = M_obsArr

        ##### Convert self.HMF to dN/(dlnzeta dlnobs) = dN/dlnM * dlnM/dlnzeta * dlnM/dlnobs
        # This only matter if dlnM/dlnobs is mass-dependent, as for dispersions
        dN_dlnzeta_dlnobs = np.exp(self.HMF_interp(np.log(M_HMF_arr), np.log(self.catalog['redshift'][dataID])))
        if obsname=='disp':
            dN_dlnzeta_dlnobs*= self.dlnM_dlnobs(obsname, M_HMF_arr, self.catalog['redshift'][dataID])

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
        HMF_2d*= self.dlnzeta_dxi(xi_arr)[None,:]

        #### Convolve with xi measurement error [lnobs]
        dP_dlnobs = np.trapz(HMF_2d * norm.pdf(self.catalog['xi'][dataID], xi_arr[None,:], 1.), xi_arr, axis=1)


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
                likeli = np.trapz(dP_dobs*lognorm.pdf(obsArr, scale=obsmeas, s=obserr), obsArr)

                # In old analysis, account for changing measurement
                if self.XrayProfileHandling=='old':
                    likeli*= obsmeas

                if getpull:
                    integrand = dP_dobs[None,:] * lognorm.pdf(obsArr[:,None], scale=obsArr[None,:], s=obserr)
                    dP_dobs_obs = np.trapz(integrand, obsArr, axis=1)
                    dP_dobs_obs/= np.trapz(dP_dobs_obs,obsArr)
                    cumtrapz = integrate.cumtrapz(dP_dobs_obs,obsArr)
                    perc = np.interp(obsmeas, obsArr[1:], cumtrapz)
                    print self.catalog['SPT_ID'][dataID], '%.4f %.4f %.4f %.4e'%(self.catalog['xi'][dataID], self.catalog['redshift'][dataID], obsmeas, 2**.5 * ss.erfinv(2*perc-1))

        if ((likeli<0)|(np.isnan(likeli))):
            print self.catalog['SPT_ID'][dataID], obsname, likeli
            #np.savetxt(self.catalog['SPT_ID'][dataID],np.transpose((obsArr, dP_dobs)))
            return 0.

        #np.savetxt(self.catalog['SPT_ID'][dataID]+obsname,np.transpose((obsArr, dP_dobs)))

        return likeli



    ############################################################################
    def get_P_2obs_xi(self, obsnames, dataID, covname):
        """Returns P(obs1, obs2|xi,z,p) for two types of follow-up data (e.g.,
        WL and X-ray)."""
        ##### Get observables, obsintr is used for setting up mass range
        obsmeas, obserr, obsintr = np.empty(2), np.empty(2), np.empty(2)
        for i in range(2):
            if obsnames[i]=='Yx':
                obsmeas[i], obsintr[i], obserr[i] = self.catalog['XrayRef'][dataID][1], self.scaling['Dx'], self.catalog['lnYx_err'][dataID]
            elif obsnames[i]=='Mgas':
                obsmeas[i], obsintr[i], obserr[i] = self.catalog['XrayRef'][dataID][1], self.scaling['Dx'], self.catalog['lnMg_err'][dataID]
            elif obsnames[i]=='disp':
                Dsigma = self.scaling['Ddisp0'] + self.scaling['DdispN']/self.catalog['Ngal'][dataID]
                obsmeas[i], obserr[i], obsintr[i] = self.catalog['veldisp'][dataID], Dsigma, Dsigma
            elif obsnames[i]=='richness':
                obsmeas[i], obserr[i], obsintr[i] = self.catalog['richness'][dataID], self.catalog['richness_err'][dataID], (self.scaling['Drichness']**2 + 1/((1-self.scaling['Drichness'])*self.catalog['richness'][dataID]))**.5
            elif obsnames[i]=='WLMegacam':
                LSSnoise = self.scaling['MegacamScatterLSS']
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_Megacam']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_Megacam']
            elif obsnames[i]=='WLHST':
                LSSnoise = self.scaling['HSTscatterLSS']
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_HST']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_HST']
            elif obsnames[i]=='WLDES':
                LSSnoise = self.scaling['DESscatterLSS']
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_DES']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_DES']

        ##### Special case for dispersions
        if ('Yx' in obsnames) and ('disp' in obsnames):
            cov = [[Dsigma**2, self.scaling['rhoXdisp']*Dsigma*self.scaling['Dx'], self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma],
                [self.scaling['rhoXdisp']*Dsigma*self.scaling['Dx'], self.scaling['Dx']**2, self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx']],
                [self.scaling['rhoSZdisp']*self.scaling['Dsz']*Dsigma, self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx'], self.scaling['Dsz']**2]]
            covmat = cov
        else:
            covmat = self.covmat[covname]


        ##### Define reasonable mass range
        # xi -> M(xi)
        xi_minmax = np.array((np.amax((2.6,self.catalog['xi'][dataID]-5)), self.catalog['xi'][dataID]+3))
        M_xi_minmax = self.obs2mass('zeta', self.xi2zeta(xi_minmax), self.catalog['redshift'][dataID])
        if M_xi_minmax[0]>self.HMF['M_arr'][-1]:
            print "cluster mass exceeds HMF mass range", self.catalog['SPT_ID'][dataID],\
                M_xi_minmax[0], self.HMF['M_arr'][-1]
            return 0

        M_obsminmax = []
        for i in range(2):
            # obs: prediction
            lnobs0 = np.log(self.mass2obs(obsnames[i], self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), self.catalog['redshift'][dataID]))
            if obsnames[i]=='disp': SZscatterobs = self.dlnM_dlnobs('zeta')/3.*self.scaling['Dsz']
            else: SZscatterobs = self.dlnM_dlnobs('zeta')/self.dlnM_dlnobs(obsnames[i])*self.scaling['Dsz']
            intrscatter = (SZscatterobs**2 + obsintr[i]**2)**.5
            obsthminmax = np.exp(np.array((lnobs0-5*intrscatter, lnobs0+3.5*intrscatter)))
            # obs: measurement
            if obsnames[i]=='richness': obsmeasminmax = np.amin((.1,obsmeas[i]-3*obserr[i])), obsmeas[i]+3*obserr[i]
            else: obsmeasminmax = np.exp(np.log(obsmeas[i])-4*obserr[i]), np.exp(np.log(obsmeas[i])+3*obserr[i])
            # put together
            obsminmax = np.array((min(obsthminmax[0],obsmeasminmax[0]), max(obsthminmax[1],obsmeasminmax[1])))
            M_obsminmax.append(self.obs2mass(obsnames[i], obsminmax, self.catalog['redshift'][dataID]))

        ##### Define grid in mass
        Mmin, Mmax = min(M_xi_minmax[0],M_obsminmax[0][0],M_obsminmax[1][0]), max(M_xi_minmax[1],M_obsminmax[0][1],M_obsminmax[1][1])
        Mmin, Mmax = max(.5*Mmin, self.HMF['M_arr'][0]), min(Mmax, self.HMF['M_arr'][-1])
        lenObs = 54
        M_obsArr = np.logspace(np.log10(Mmin), np.log10(Mmax), lenObs)
        M_HMF_arr = M_obsArr


        ##### Observable arrays
        lnzeta_arr = np.log(self.mass2obs('zeta', M_obsArr, self.catalog['redshift'][dataID]))
        xi_arr = self.zeta2xi(np.exp(lnzeta_arr))
        obsArr, lnobsArr = [], []
        for i in range(2):
            obsArrTemp = self.mass2obs(obsnames[i], M_obsArr, self.catalog['redshift'][dataID])
            ##### Add radial dependence for X-ray observables
            if obsnames[i] in ('Mgas','Yx') and self.XrayProfileHandling=='modelMgasPL':
                # Angular diameter distances in current and reference cosmology [Mpc]
                dA = cosmo.dA(self.catalog['redshift'][dataID], self.cosmology)/self.cosmology['h']
                dAref = cosmo.dA(self.catalog['redshift'][dataID], cosmologyRef)/cosmologyRef['h']
                # R500 [kpc]
                rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['redshift'][dataID], self.cosmology)**2
                r500 = 1000 * (3*M_obsArr/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
                # r500 in reference cosmology [kpc]
                r500ref = r500 * dAref/dA
                # Xray observable at rFid
                obsFid = obsArrTemp * (self.catalog['XrayRef'][dataID][0]/r500ref)**self.scaling['slope_MgR']
                # X-ray observable at rFid, corrected to reference cosmology
                obsArrTemp = obsFid * (dAref/dA)**2.5
            obsArr.append( obsArrTemp )
            lnobsArr.append( np.log(obsArrTemp) )


        ##### HMF to dN/(dlnzeta dlnobs0 dlnobs1) = dN/dlnM * dlnM/dlnzeta * dlnM/dlnobs0 * dlnM/dlnobs1
        # This only matter if dlnM/dlnobs is mass-dependent, as for dispersions
        dN_dlnzeta_dlnobs = np.exp(self.HMF_interp(np.log(M_HMF_arr), np.log(self.catalog['redshift'][dataID])))
        if 'disp' in obsnames:
            dN_dlnzeta_dlnobs*= self.dlnM_dlnobs('disp', M_HMF_arr, self.catalog['redshift'][dataID])

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
        HMF_3d*= self.dlnzeta_dxi(xi_arr)[None,None,:]

        #### Convolve with xi measurement error [lnobs0][lnobs1]
        dP_dlnobs = np.trapz(HMF_3d * norm.pdf(self.catalog['xi'][dataID], xi_arr[None,None,:], 1.), xi_arr, axis=2)

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
        likeli1 = np.trapz(dP_dobs1*lognorm.pdf(obsArr[1], scale=obsmeas[1], s=obserr[1]), obsArr[1])

        if self.XrayProfileHandling=='old':
            likeli1*= obsmeas[1]

        #np.savetxt(self.catalog['SPT_ID'][dataID]+'_3d'+obsnames[0],np.transpose((obsArr[0], dP_dobs0)))
        #np.savetxt(self.catalog['SPT_ID'][dataID]+'_3d'+obsnames[1],np.transpose((obsArr[1], dP_dobs1)))


        ##### Probability
        likeli = likeli0*likeli1

        return likeli




    ############################################################################
    ##### Utility functions
    def xi2zeta(self, xi): return (xi**2 - 3)**.5
    def zeta2xi(self, zeta): return (zeta**2 + 3)**.5
    def dlnzeta_dxi(self, xi): return xi / (xi**2 - 3)
    def dxi_dzeta(self, zeta): return zeta / (zeta**2 + 3)


    ####################
    def obs2mass(self, name, obs, z):
        """Returns mass given (observable, z) using scaling relation."""
        if name=='zeta':
            Asz = self.thisSPTfieldCorrection * self.scaling['Asz']
            lnM = np.log(self.SZmPivot) + (np.log(obs) - np.log(Asz)\
                - self.scaling['Csz']*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology)))/(self.scaling['Bsz']\
                + self.scaling['Esz']*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology)))
            return np.exp(lnM)
        elif name=='Yx':
            if self.YXPARAM=='SPT_XVP':
                return 1e14 * self.scaling['Ax'] * self.cosmology['h']**1.5\
                    * (self.cosmology['h']/.72)**(2.5*self.scaling['Bx']-1.5)\
                    * (obs/3.)**self.scaling['Bx'] * cosmo.Ez(z, self.cosmology)**self.scaling['Cx']
            elif self.YXPARAM=='Munich':
                return self.XraymPivot * self.cosmology['h']**1.5 * (obs/(self.scaling['Ax']
                    *(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx']))**(1/self.scaling['Bx'])
        elif name=='Mgas':
            return self.XraymPivot * self.cosmology['h'] * (obs/self.XraymPivot/self.scaling['Ax']
                /(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx'])**(1./self.scaling['Bx'])
        elif name=='disp':
            h70z = self.cosmology['h']/.7*cosmo.Ez(z, self.cosmology)
            M200c = 1e15*self.cosmology['h'] * (obs/self.scaling['Adisp']/h70z**self.scaling['Cdisp'])**self.scaling['Bdisp']
            return np.exp(self.lnM200_to_lnM500(np.log(M200c), z))
        elif name=='richness':
            return self.richmPivot* (obs/self.scaling['Arichness']
                /(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Crichness'])**(1/self.scaling['Brichness'])
        elif name=='WLMegacam':
            return obs/self.scaling['bWL_Megacam']
        elif name=='WLHST':
            return obs/self.scaling['bWL_HST']
        elif name=='WLDES':
            return obs/self.scaling['bWL_DES']
        else:
            raise ValueError("Observable not known:",name)


    ####################
    def mass2obs(self, name, mass, z):
        """Returns observable given (mass, z) using scaling relation."""
        if name=='zeta':
            lnzeta = np.log(self.scaling['Asz']*self.thisSPTfieldCorrection)\
                + self.scaling['Bsz'] * np.log(mass/self.SZmPivot)\
                + self.scaling['Csz'] * np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))\
                + self.scaling['Esz'] * np.log(mass/self.SZmPivot)*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))
            return np.exp(lnzeta)
        elif name=='Yx':
            if self.YXPARAM=='SPT_XVP':
                return 3.*(mass*1e-14/(self.scaling['Ax'] * self.cosmology['h']**1.5
                    * (self.cosmology['h']/.72)**(2.5*self.scaling['Bx']-1.5)
                    * cosmo.Ez(z, self.cosmology)**self.scaling['Cx']))**(1/self.scaling['Bx'])
            elif self.YXPARAM=='Munich':
                return self.scaling['Ax']* (mass/self.cosmology['h']**1.5/self.XraymPivot)**self.scaling['Bx']\
                    * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx']
        elif name=='Mgas':
            lnMgas = np.log(self.XraymPivot * self.scaling['Ax']) + self.scaling['Bx']*np.log(mass/self.XraymPivot/self.cosmology['h'])\
                + self.scaling['Cx']*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))\
                + self.scaling['Ex']*np.log(mass/self.XraymPivot/self.cosmology['h'])*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))
            return np.exp(lnMgas)
        elif name=='disp':
            h70z = self.cosmology['h']/.7*cosmo.Ez(z, self.cosmology)
            M200c = np.exp(self.lnM500_to_lnM200(np.log(mass), z))
            if len(M200c)==1: M200c = M200c[0]
            return self.scaling['Adisp'] * (M200c/1e15/self.cosmology['h'])**(1/self.scaling['Bdisp']) * h70z**self.scaling['Cdisp']
        elif name=='richness':
            return self.scaling['Arichness'] * (mass/self.richmPivot)**self.scaling['Brichness']\
                * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Crichness']
        elif name=='WLMegacam':
            return self.scaling['bWL_Megacam'] * mass
        elif name=='WLHST':
            return self.scaling['bWL_HST'] * mass
        elif name=='WLDES':
            return self.scaling['bWL_DES'] * mass
        else:
            raise ValueError("Observable not known:",name)


    ####################
    def dlnM_dlnobs(self, name, M0_arr=None, z=None):
        """Returns dlnM/dln(obs) for a given observable."""
        if name=='zeta': return 1/self.scaling['Bsz']
        elif name=='richness': return 1/self.scaling['Brichness']
        elif name=='Yx':
            if self.YXPARAM=='SPT_XVP': return self.scaling['Bx'] / (1-self.scaling['slope_MgR']/3)
            elif self.YXPARAM=='Munich': return 1/ (self.scaling['Bx'] - self.scaling['slope_MgR']/3)
        elif name=='Mgas': return 1/ (self.scaling['Bx'] - self.scaling['slope_MgR']/3)
        elif (name=='WLMegacam')|(name=='WLHST')|(name=='WLDES'): return 1.
        elif name=='disp':
            dlnM = np.log(1.01)
            dlnobs = np.log(self.mass2obs('disp', 1.01*M0_arr, z)/self.mass2obs('disp', M0_arr, z))
            if np.any(dlnobs==0.):
                if dlnobs[-1]==0: dlnobs[-1] = dlnobs[-2]
            return dlnM/dlnobs
