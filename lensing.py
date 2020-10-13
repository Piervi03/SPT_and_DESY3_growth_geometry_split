from __future__ import division, print_function
import numpy as np
from numpy.lib import scimath as sm
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.stats import norm
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
# from scipy.linalg import cho_factor, cho_solve
from multiprocessing import Pool

import h5py
import imp
import time

import cosmo, Mconversion_concentration, miscentering, scaling_relations

########################################

# Radius [Mpc/h]
len_r_fine = 64
grid_lgr_fine_min = -4
grid_lgr_fine_max = 1
# Mass [Msun/h]
grid_lgM_min = 12.5
grid_lgM_max = 15.5

def unwrap_self_like_cluster(arg):
    return SPTlensing.like_cluster(*arg)

##### This class reads and stores shear data and calculates P(shear|P(M))
class SPTlensing:

    def __init__(self, catalog, WLsimcalibfile,
                 HSTfile, MegacamFile, DESfile,
                 mcType):
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration
        # self.Delta_crit = self.WLcalib['Delta_crit']
        self.miscenterer = miscentering.MisCentering('SPT')

        self.len_M_arr = 32
        self.M_arr = np.logspace(grid_lgM_min, grid_lgM_max, self.len_M_arr)
        self.r_fine = np.logspace(grid_lgr_fine_min, grid_lgr_fine_max, len_r_fine)

        self.NPROC = 0

        self.mcType = mcType

        readdata(catalog, HSTfile, MegacamFile, DESfile)


    ########################################
    def like_all(self, catalog, cosmology, scaling):
        """Return p(data|M_arr) for all clusters with WL data."""
        # t = []
        # t.append(time.time())
        self.cosmology = cosmology
        self.scaling = scaling
        if self.mcType != 'None':
            self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology,
                                                                           setup_interp=True, interp_massdef=500)

        # Go through all clusters with WL data
        WL_idx = (catalog['WLdata'] != None).nonzero()[0]

        # Redshift limits
        z_cl_min = np.amin(catalog['REDSHIFT'][WL_idx])
        z_cl_max = np.amax(catalog['REDSHIFT'][WL_idx])

        # Pre-compute angular diameter distances
        self.get_dAs(z_cl_min, z_cl_max, 5., cosmology)
        # t.append(time.time())

        if self.NPROC==0:
            p_Mwl = [self.like_cluster(catalog[i]) for i in WL_idx]
        else:
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len(WL_idx), catalog[WL_idx])
            p_Mwl = pool.map(unwrap_self_like_cluster, argin)
            pool.close()

        catalog['p_Mwl'] = [None]*len(catalog)
        catalog['p_Mwl'][WL_idx] = p_Mwl
        # t.append(time.time())
        # print("lensing done", t[-1]-t[0])

        return 0


    ########################################
    # Get P(Mwl) from dP/dMwl and shear data
    def like_cluster(self, data):
        """Return likelihood of shear profile for a given cluster (index) given
        an array of cluster masses."""
        # t = []
        # t.append(time.time())

        self.cat_cl = data

        ##### Cosmology and halo stuff, in 1/h units
        self.get_beta(self.cosmology)
        # t.append(time.time())

        ##### Likelihood
        if self.cat_cl['WLdata']['datatype']=='DES':
            pOfMass = self.like_DES()
        elif self.cat_cl['WLdata']['datatype']=='Megacam':
            rho_c_z, Dl, delta_c, r_s = self.get_cluster_properties()
            pOfMass = self.like_Megacam(rho_c_z, Dl, delta_c, r_s)
        elif self.cat_cl['WLdata']['datatype']=='HST':
            rho_c_z, Dl, delta_c, r_s = self.get_cluster_properties()
            pOfMass = self.like_HST(rho_c_z, Dl, delta_c, r_s)
        # t.append(time.time())
        # t = np.array(t)
        # print('done', t[-1]-t[0], np.diff(t))
        return pOfMass



    ########################################


    def likelihood_DES(self, g_t):
        """Return P(DES data|Mwl)"""
        diff = self.cat_cl['WLdata']['shear'] - g_t
        chi2 = np.dot(diff, cho_solve(self.cat_cl['WLdata']['shear_cho_factor'], diff))
        likeli = np.exp(-.5 * chi2)

        return likeli



    def like_DES(self):
        """Return array P(DES data|Mwl)."""

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Dl = np.exp(self.lndA_interp(np.log(self.cat_cl['REDSHIFT'])))
        Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.cat_cl['REDSHIFT'], self.cosmology)**2 # [h^2 Msun/Mpc^3]
        r200c = (3*self.M_arr/4/np.pi/200/rho_c_z)**(1/3)
        c200c = 3.
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        r_s = r200c/c200c

        # Get miscentering radius
        # zeta = scaling_relations.mass2obs('zeta', self.M_arr, self.cat_cl['REDSHIFT'], self.scaling, self.cosmology)
        # xi = scaling_relations.zeta2xi(zeta)
        # r_core = 1/self.scaling['r_core_inv']
        # R_mis = self.miscenterer.get_mean_Rmis_SPT(r200c, r_core, xi, Dl)
        R_mis_arcmin = self.miscenterer.get_Rmis_extr_opt(self.cat_cl['LAMBDA_MCMF_COMB'], self.cat_cl['REDSHIFT'], self.WLcalib['miscenter_opt'], self.cosmology)
        R_mis = R_mis_arcmin * Dl * np.pi/60/180

        # NFW surface mass densities [mass][radius]
        r_Mpch = self.cat_cl['WLdata']['r_arcmin'] * Dl * np.pi/60/180
        x = r_Mpch[None,:] / r_s[:,None]

        Sigma_NFW = get_Sigma(x, r_s[:,None], rho_c_z, delta_c)
        Delta_Sigma_NFW = get_Delta_Sigma(x, r_s[:,None], rho_c_z, delta_c)
        Sigma_NFW_mean = Sigma_NFW - Delta_Sigma_NFW

        # NFW surface mass densities at Rmis [mass]
        x_Rmis = R_mis/r_s
        Sigma_NFW_at_Rmis = get_Sigma(x_Rmis, r_s, rho_c_z, delta_c)
        Delta_Sigma_NFW_at_Rmis = get_Delta_Sigma(x_Rmis, r_s, rho_c_z, delta_c)
        Sigma_NFW_mean_at_Rmis = Sigma_NFW_at_Rmis - Delta_Sigma_NFW_at_Rmis

        # Miscentered quantities
        # Sigma = Sigma(R_mis) for r<R_mis
        Sigma_mis = Sigma_NFW.copy()
        if not R_mis<r_Mpch[0]:
            Sigma_mis[:,r_Mpch<R_mis] = Sigma_NFW_at_Rmis

        Sigma_mis_mean = np.empty(Sigma_NFW.shape)
        for i in range(self.len_M_arr):
            Sigma_mis_mean[i,r_Mpch<R_mis] = Sigma_NFW_at_Rmis[i]
            Sigma_mis_mean[i,r_Mpch>R_mis] = Sigma_NFW_mean[i,:] + (R_mis/r_Mpch)**2 * (Sigma_NFW_at_Rmis-Sigma_NFW_mean_at_Rmis)[i]

        # Reduced shear profile [mass][radius]
        reduced_shear = (Sigma_mis-Sigma_mis_mean)/Sigma_c / (1 - Sigma_mis/Sigma_c)

        # Cluster member contamination
        r_s_fcl = self.cat_cl['WLdata']['r200_fid'] / self.WLcalib['c_fcl']
        delta_c_fcl = 200/3 * self.WLcalib['c_fcl']**3 / (np.log(1+self.WLcalib['c_fcl']) - self.WLcalib['c_fcl']/(1+self.WLcalib['c_fcl']))
        x_fcl = r_Mpch / r_s_fcl
        Sigma_fcl = get_Sigma(x_fcl, r_s_fcl, rho_c_z, delta_c_fcl)/get_Sigma(self.WLcalib['x0_fcl'], r_s_fcl, rho_c_z, delta_c_fcl)
        idx = (self.cat_cl['REDSHIFT']>self.WLcalib['A_fcl_z']).nonzero()[0][-1]
        f_cl = self.WLcalib['A_fcl'][idx] * (self.cat_cl['LAMBDA_MCMF_COMB']/self.WLcalib['lambda_piv_fcl'])**self.WLcalib['B_fcl'] * Sigma_fcl
        reduced_shear_cont = (1-f_cl) * reduced_shear

        # Likelihood!
        diffs = reduced_shear_cont - self.cat_cl['WLdata']['shear']
        chi2 = (diffs/self.cat_cl['WLdata']['shear_err'])**2
        P_DES_Mwl = np.exp(-.5*np.sum(chi2, axis=1))

        return P_DES_Mwl



    ########################################

    def get_cluster_properties(self):
        """Return rho_c(z_cluster), luminosity distance (z_cluster), delta_c,
        and r_s."""
        Dl = np.exp(self.lndA_interp(np.log(self.cat_cl['REDSHIFT'])))
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.cat_cl['REDSHIFT'], self.cosmology)**2 # [h^2 Msun/Mpc^3]

        M200c = np.exp(self.MCrel.lnM_to_lnM200(self.cat_cl['REDSHIFT'], np.log(self.M_arr)))[0]
        r200c = (3*M200c/4/np.pi/200/rho_c_z)**(1/3)
        c200c = MCrel.calC200(M200c, self.cat_cl['REDSHIFT'])
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        r_s = r200c/c200c
        return rho_c_z, Dl, delta_c, r_s


    ########################################

    def like_Megacam(self, rho_c_z, Dl, delta_c, r_s):
        """Return array P(Megacam data|Mwl)."""
        # Dimensionless radial distance [Radius][Mass]
        x = self.cat_cl['WLdata']['r_deg'][:,None] * Dl * np.pi/180 / r_s[None,:]
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg

        # gamma_t, kappa, g_t [Radius][Mass]
        gamma_2d = get_Delta_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_2d/(1-kappa_2d) * (1 + kappa_2d*(self.beta2_avg/self.beta_avg**2-1))

        likelihood = norm.pdf(g_2d, self.cat_cl['WLdata']['shear'][:,None], self.cat_cl['WLdata']['shearerr'][:,None])

        pOfMass = np.prod(likelihood, axis=0)

        return pOfMass



    ########################################

    def like_HST(self, rho_c_z, Dl, delta_c, r_s):
        """Return array P(HST data|Mwl)."""
        # Dimensionless radial distance [Radius][Mass]
        x = self.cat_cl['WLdata']['r_deg'][:,None] * Dl * np.pi/180 / r_s[None,:]

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2] [Radius]
        rangeR = range(len(self.cat_cl['WLdata']['r_deg']))
        betaR = np.array([self.beta_avg[self.cat_cl['WLdata']['magbinids'][i]] for i in rangeR])
        beta2R = np.array([self.beta2_avg[self.cat_cl['WLdata']['magbinids'][i]] for i in rangeR])
        Sigma_c = 1.6624541593797974e+18/Dl/betaR

        # gamma_t and kappa [Radius][Mass]
        gamma_2d = get_Delta_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:,None]
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:,None]

        # [Radius][Mass]
        mu0_2d = 1/((1-kappa_2d)**2 - gamma_2d**2)
        kappaFake = (mu0_2d-1)/2

        # Magnification correction [Radius][Mass]
        mykappa = kappaFake * 0.3/betaR[:,None]

        magcorr = [np.interp(mykappa[i], self.cat_cl['WLdata']['magcorr'][self.cat_cl['WLdata']['magbinids'][i]][0], self.cat_cl['WLdata']['magcorr'][self.cat_cl['WLdata']['magbinids'][i]][1]) for i in rangeR]

        # Beta correction [Radius][Mass]
        betaratio = beta2R/betaR**2
        betaCorr = (1 + kappa_2d*(betaratio[:,None]-1))

        # Reduced shear g_t [Radius][Mass]
        g_2d = np.array(magcorr) * gamma_2d/(1-kappa_2d) * betaCorr

        # Only consider 500<r/kpc/1500 in reference cosmology
        cosmoRef = {'Omega_m':.3, 'Omega_l':.7, 'h':.7, 'w0':-1., 'wa':0}
        DlRef = cosmo.dA(self.cat_cl['REDSHIFT'], cosmoRef)
        rPhysRef = self.cat_cl['WLdata']['r_deg'] * DlRef * np.pi/180 /cosmoRef['h']
        rInclude = np.where((rPhysRef>.5)&(rPhysRef<1.5))[0]

        likelihood = norm.pdf(g_2d[rInclude,:], self.cat_cl['WLdata']['shear'][rInclude,None], self.cat_cl['WLdata']['shearerr'][rInclude,None])

        pOfMass = np.prod(likelihood, axis=0)

        return pOfMass




    ########################################
    # dA [Mpc/h]
    def get_dAs(self, z_cl_min, z_cl_max, z_s_max, cosmology):
        """Precompute angular diameter distances for an array of redshifts."""
        z = np.logspace(np.log10(z_cl_min), np.log10(z_s_max), 32)
        dA = np.array([cosmo.dA(z_, cosmology) for z_ in z])
        self.lndA_interp = InterpolatedUnivariateSpline(np.log(z), np.log(dA))

        z_cl = np.logspace(np.log10(z_cl_min), np.log10(z_cl_max), 32)
        z_s = np.logspace(np.log10(z_cl_min), np.log10(z_s_max), 32)
        tmp = np.array([cosmo.dA_two_z(z_cl_, z_s_, cosmology) for z_cl_ in z_cl for z_s_ in z_s]).reshape(32,32)
        self.dA_twoz_interp = RectBivariateSpline(np.log(z_cl), np.log(z_s), tmp)

        return 0


    ########################################
    def get_beta(self, cosmology):
        """Compute <beta> and <beta^2> from distribution of redshift galaxies."""
        ##### Only consider redshift bins behind the cluster
        betaArr = np.zeros(len(self.cat_cl['WLdata']['redshifts']))
        bgIdx = np.where(self.cat_cl['WLdata']['redshifts']>self.cat_cl['REDSHIFT'])[0]

        ##### Calculate beta(z_source)
        # beta = dA_ls / dA_l
        betaArr[bgIdx] = self.dA_twoz_interp(np.log(self.cat_cl['REDSHIFT']), np.log(self.cat_cl['WLdata']['redshifts'][bgIdx]))
        betaArr[bgIdx]/= np.exp(self.lndA_interp(np.log(self.cat_cl['WLdata']['redshifts'][bgIdx])))

        ##### Weight beta(z) with N(z) distribution to get <beta> and <beta^2>
        if self.cat_cl['WLdata']['datatype'] in ['DES', 'Megacam']:
            self.beta_avg = np.average(betaArr, weights=self.cat_cl['WLdata']['Nz'])
            self.beta2_avg = np.average(betaArr**2, weights=self.cat_cl['WLdata']['Nz'])

        else:
            self.beta_avg, self.beta2_avg = {}, {}
            for i in self.cat_cl['WLdata']['pzs'].keys():
                self.beta_avg[i] = np.sum(self.cat_cl['WLdata']['pzs'][i]*betaArr)/self.cat_cl['WLdata']['Ntot'][i]
                self.beta2_avg[i] = np.sum(self.cat_cl['WLdata']['pzs'][i]*betaArr**2)/self.cat_cl['WLdata']['Ntot'][i]


################################################################################
# NFW tools by Joerg Dietrich (https://github.com/joergdietrich/NFW)

def arcsec(z):
    """Compute the inverse sec of the complex number z."""
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1/z**2)
    val = 1j * np.log(val2 + val1)
    return .5 * np.pi + val

def get_Delta_Sigma(x, r_s, rho_c_z, delta_c):
    """Return Delta Sigma = Sigma - Sigma_mean"""
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (1 - x**2)
    num = ((3 * x**2) - 2) * arcsec(x)
    div = x**2 * (sm.sqrt(x**2 - 1))**3
    val2 = (num / div).real
    val3 = 2 * np.log(x / 2) / x**2
    return fac * (val1+val2+val3)

def get_Sigma(x, r_s, rho_c_z, delta_c):
    """Return Sigma_NFW"""
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (x**2 - 1)
    val2 = (arcsec(x) / (sm.sqrt(x**2 - 1))**3).real
    return fac * (val1-val2)



################################################################################
def readdata(catalog, HSTfile, MegacamFile, DESfile):
    """Read and load weak-lensing data into `WLdata` field in `catalog` if
    the corresponding path-variables lead to valid files on disk."""
    # Empty WL data field
    catalog['WLdata'] = [None for i in range(len(catalog['SPT_ID']))]

    ##### Check for HST data
    if HSTfile != 'None':
        with h5py.File(HSTfile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'HST', 'center':f[name].attrs['center'],
                        'r_deg':f[name]['shear_profile'][0], 'shear':f[name]['shear_profile'][1], 'shearerr':f[name]['shear_profile'][2],
                        'magbinids':f[name]['magbinid'][:],
                        'redshifts':f[name]['redshifts'][:],
                        'pzs':{}, 'magcorr':{}, 'Ntot':{}}
                    for key in f[name]['magbindata'].keys():
                        dict_key = int(key)
                        catalog['WLdata'][i]['pzs'][dict_key] = f[name]['magbindata'][key]['pzs'][:]
                        catalog['WLdata'][i]['Ntot'][dict_key] = np.sum(catalog['WLdata'][i]['pzs'][dict_key])
                        catalog['WLdata'][i]['magcorr'][dict_key] = f[name]['magbindata'][key]['magnificationcorr'][:]

    ##### Megacam data
    if MegacamFile != 'None':
        with h5py.File(MegacamFile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'Megacam',
                        'r_deg':f[name]['shear_profile'][0], 'shear':f[name]['shear_profile'][1], 'shearerr':f[name]['shear_profile'][2],
                        'redshifts':f[name]['Nz'][0], 'Nz':f[name]['Nz'][1], 'Ntot':np.sum(f[name]['Nz'][1]),}

    ##### Check for DES data
    if DESfile != 'None':
        with h5py.File(DESfile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'DES',
                        'r_arcmin': f[name]['r_arcmin'][:],
                        'shear': f[name]['shear'][:],
                        'shear_err': f[name]['shear_err'][:],
                        'redshifts': f[name]['source_redshifts'][:],
                        'Nz': f[name]['source_Nz'][:],
                        'r200_fid': f[name]['r200_fid'][()],
                        # 'R_mis_arcmin': f[name]['R_mis_arcmin'][()],
                        # 'shear_cho_factor': cho_factor(f[name]['shear_profile_cov'][:]),
                        }
