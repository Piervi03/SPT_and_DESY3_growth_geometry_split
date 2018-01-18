from __future__ import division
import numpy as np
import scipy.optimize as op
from colossus.cosmology import cosmology as DKcosmo
from colossus.halo import concentration

class ConcentrationConversion(object):

    def __init__(self, MCrelation, cosmology=None):
        self.MCrelation = MCrelation
        if isinstance(MCrelation, str):
            if MCrelation=='DK15':
                self.colossuscosmo = DKcosmo.setCosmology('mycosmo',
                    {'flat':True, 'H0':100.*cosmology['h'], 'Om0':cosmology['Omega_m'],
                    'Ob0':cosmology['Omega_b'], 'sigma8':cosmology['sigma8'], 'ns':cosmology['ns']})

            elif MCrelation!='Duffy08':
                raise ValueError('Unknown mass-concentration relation:', MCrelation)

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
