from __future__ import division
import numpy as np
import scipy.optimize as op
# import sigma
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.integrate
# from colossus.cosmology import cosmology as DKcosmo
# from colossus.halo import concentration

class ConcentrationConversion(object):

    def __init__(self, MCrelation, cosmology=None):
        self.MCrelation = MCrelation
        if isinstance(MCrelation, str):
            if MCrelation=='DK15':
                self.colossuscosmo = DKcosmo.setCosmology('mycosmo', {'flat':True, 'H0':100.*cosmology['h'], 'Om0':cosmology['Omega_m'], 'Ob0':cosmology['Omega_b'], 'sigma8':cosmology['sigma8'], 'ns':cosmology['ns']})

                """Omh2 = cosmology['Omega_m']*cosmology['h']**2
                Obh2 = cosmology['Omega_b']*cosmology['h']**2
                fb = cosmology['Omega_b']/cosmology['Omega_m']
                k_arr = np.logspace(.4, 1, 50)
                self.logk_arr = np.log(k_arr)
                sound_horizon = 44.5 * np.log(9.83/Omh2)/(1+10*Obh2**.75)**.5
                alphaGamma = 1. - .328*np.log(431.*Omh2)*Obh2/Omh2 + .38*np.log(22.3*Omh2)*fb**2
                Gamma = cosmology['Omega_m']*cosmology['h'] * (alphaGamma + (1.-alphaGamma)/(1.+(.43*k_arr*h0*sound_horizon)**4))
                q = k_arr * (2.7255/2.7)**2 / Gamma
                C0 = 14.2 + 731. / (1. + 62.5*q)
                L0 = np.log(2. * np.exp(1.) + 1.8*q)
                TF = L0 / (L0 + C0*q**2)

                E_z = lambda z: (cosmology['Omega_m']*(1+z)**3 + 1-cosmology['Omega_m'])**.5
                integrand = lambda z_int: (1+z_int)/(cosmology['Omega_m']*(1+z_int)**3+1-cosmology['Omega_m'])**1.5

                self.z_arr = np.linspace(0, 2, 50)
                self.D_arr = np.array([E_z(z) * scipy.integrate.quad(integrand, z, 1e3) for z in self.z_arr])

                PK_unnorm = k_arr**cosmology['ns']* TF**2 * self.D_arr[0]**2
                norm = cosmology['sigma_8']/sigma.calc_sigma8(k_arr, PK_unnorm)

                self.PK_EHsmooth = k_arr**cosmology['ns']* TF**2 * norm**2
                self.interp_n = InterpolatedUnivariateSpline(self.logk_arr, np.log(self.PK_EHsmooth))

                r_array = (3.*M_arr/4./math.pi/rho_m)**(1./3.)
                self.M_arr = M_arr
                self.sigma_M_0 = sigma.calc_sigma2_r(r_array, k_arr, self.PK_EHsmooth)"""

            elif MCrelation!='Duffy08':
                raise ValueError('Unknown mass-concentration relation:',
                                 MCrelation)

    def whichMCrel(self):
        print self.MCrelation


    # 200crit from Duffy et al 2008, input [M200c/h]
    def calC200(self, m, z):
        if self.MCrelation=='Duffy08':
            m = np.atleast_1d(m)
            m[np.where(m<1e9)] = 1e9
            #return 6.71*(m/2.e12)**(-0.091)*(1.+z)**(-0.44) # relaxed samples
            return 5.71*(m/2.e12)**(-0.084)*(1.+z)**(-0.47) # full sample
        elif self.MCrelation=='DK15':
            c = np.atleast_1d(concentration.concentration(m, '200c', z, model='diemer15'))
            c[c>30.] = 30.
            return c

            """R_M = (3.*m*h0/4./np.pi/Omega_m/rhocrit)**(1./3.)
            k_r = .69*2.*np.pi/R_M
            n = self.interp_n(np.log(k_r), nu=1)
            nu = 1.686 / np.interp(np.log(k_r), self.logk_arr, np.log(self.sigma_M_0)) * np.interp(z, self.z_arr,  self.D_arr)/self.D_arr[0]
            cfloor = 6.58 + n*1.37
            nu0 = 6.82 + n*1.42
            return .5*cfloor * ((nu0/nu)**1.12 + (nu/nu0)**1.69)"""
        else:
            return float(self.MCrelation)


    ##### Actual input functions
    # Input in [Msun/h]
    def MDelta_to_M200(self,mc,overdensity,z):
        ratio = overdensity/200.
        Mmin = mc * ratio / 4.
        Mmax = mc * ratio * 4.
        return op.brentq(self.mdiff_findM200, Mmin, Mmax, args=(mc,overdensity,z), xtol=1.e-6)

    # Input in [Msun/h]
    def M200_to_MDelta(self, Minput, overdensity, z):
        ratio = 200./overdensity
        Mmin = Minput * ratio / 4.
        Mmax = Minput * ratio * 4.
        return op.brentq(self.mdiff_findMDelta, Mmin, Mmax, args=(Minput,overdensity,z), xtol=1.e-6)


    ##### Functions used for conversion
    # calculate the coefficient for NFW aperture mass given c
    def calcoef(self, c):
        return np.log(1+c)-c/(1+c)

    # root function for concentration
    def diffc(self, c2, c200, ratio):
        return self.calcoef(c200)/self.calcoef(c2) - ratio*(c200/c2)**3

    def findc(self, c200, overdensity):
        ratio = 200./overdensity
        #if self.diffc(.1,c200,ratio)*self.diffc(100, c200, ratio)>0:
        #    print c200
        return op.brentq(self.diffc, .1, 40., args=(c200,ratio), xtol=1.e-6)

    # Root function for mass
    def mdiff_findM200(self, m200, mc, overdensity, z):
        con =  self.calC200(m200,z)
        con2 = self.findc(con,overdensity)
        return m200/mc - self.calcoef(con)/self.calcoef(con2)

    def mdiff_findMDelta(self,mguess,Minput,overdensity,z):
        conin =  self.calC200(Minput,z)
        conguess = self.findc(conin,overdensity)
        return Minput/mguess - self.calcoef(conin)/self.calcoef(conguess)
