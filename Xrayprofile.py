from __future__ import division
import numpy as np
import imp
from cosmosis.datablock import option_section
import cosmo


class XrayProfile:
    def __init__(self, options):
        self.todo = {'Yx': options.get_bool(option_section, 'doYx'),
            'Mgas': options.get_bool(option_section, 'doMgas')}
        self.todo['Xobs'] = 'Mg' if self.todo['Mgas'] else 'Yx'
        self.YXPARAM = options.get_string(option_section, 'YXPARAM')
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        self.XraymPivot = options.get_double(option_section, 'XraymPivot')
        self.XrayProfileHandling = options.get_string(option_section, 'XrayProfileHandling')
        self.XraySample = SPTdata.XraySample

        ##### Reference cosmology for which Mgas is measured
        # Never ever change these!
        self.cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1}
        # Fiducial scaling relation to get r_500fiducial
        # Maybe re-adjust if very different from run output
        self.scalingRef = {'Ax':.12, 'Bx':1.2, 'Cx':0.}
        self.scalingYxRef = {'Ax':6., 'Bx':.57, 'Cx':-.4}
        self.scalingYxMunichRef = {}



    ########################################
    def setRef(self, catalog):
        """Extract an X-ray observable from the measured profiles. Find radius r at which
            data(r)==scalingRelation(r).

        Returns: ln(likelihood) to account for the fact that the observable is actually a
            function of the parameters.

        Note: Everything in this function is in units of Mpc or Msun, no h!
        """

        E06 = cosmo.Ez(.6, self.cosmologyRef)

        catalog['XrayRef'] = [None for i in range(len(catalog['SPT_ID']))]


        ##### Copy mock values when not using profiles
        if self.XrayProfileHandling=='fixed':
            for i,name in enumerate(catalog['SPT_ID']):
                if self.todo['Xobs']=='Yx':
                     if catalog['Yx_MM'][i]!=0:
                         obserr = catalog['lnYx_err'][i]
                         catalog['XrayRef'][i] = (1234., catalog['Yx_MM'][i], obserr)
                elif self.todo['Xobs']=='Mg':
                     if catalog['Mgas'][i]!=0:
                         obserr = catalog['lnMg_err'][i]
                         catalog['XrayRef'][i] = (1234., catalog['Mgas'][i], obserr)
            return


        for i,name in enumerate(catalog['SPT_ID']):
            ##### Exclude double entries
            if name in self.XraySample and (name,catalog['field'][i]) not in self.SPTdoubleCount:

                E_z = cosmo.Ez(catalog['redshift'][i], self.cosmologyRef)
                rho_c_z = cosmo.RHOCRIT * (self.cosmologyRef['h']*cosmo.Ez(catalog['redshift'][i], self.cosmologyRef))**2.

                ##### Initial M500 guess, and observational error
                if self.todo['Xobs']=='Yx':
                    Xold = 1.
                    obserr = catalog['lnYx_err'][i]
                elif self.todo['Xobs']=='Mg':
                    Xold = 1e13
                    obserr = catalog['lnMg_err'][i]

                nonzero = np.nonzero(catalog['Mg'][i][0])[0]
                for j in range(100):
                    if self.todo['Xobs'] == 'Yx':
                        if self.YXPARAM=='SPT_XVP':
                            M500 = 1e14 * self.scalingYxRef['Ax'] * self.cosmologyRef['h']**.5 * (self.cosmologyRef['h']/.72)**(2.5*self.scalingYxRef['Bx']-1.5) * (Xold/3)**self.scalingYxRef['Bx'] * E_z**self.scalingYxRef['Cx']
                        elif self.YXPARAM=='Munich':
                            M500 = self.XraymPivot * cosmology['h']**.5 * (Xold/(self.scalingYxMunichRef['Ax']*(E_z/E06)**self.scalingYxMunichRef['Cx']))**(1/self.scalingYxMunichRef['Bx'])
                    elif self.todo['Xobs'] == 'Mg':
                        M500 = self.XraymPivot * (Xold/self.XraymPivot/self.scalingRef['Ax'] / (E_z/E06)**self.scalingRef['Cx'])**(1/self.scalingRef['Bx'])

                    r500ref = 1000 * (3*M500/(4*np.pi*500*rho_c_z))**(1/3)

                    MgRef = np.interp(r500ref, catalog['Mg'][i][0,nonzero], catalog['Mg'][i][1,nonzero])

                    Xnew = MgRef
                    if self.todo['Xobs']=='Yx':
                        Xnew*= 1e-14*catalog['Tx'][i]

                    if np.absolute(Xold/Xnew-1)<1e-4: break

                    Xold = Xnew

                if j==99:
                    print name,r500ref,'no convergence after 100 steps'
                if r500ref>catalog['Mg'][i][0,nonzero][-1]:
                    print 'Warning,',catalog['SPT_ID'][i],'r500ref', r500ref, catalog['Mg'][i][0,nonzero][-1]

                catalog['XrayRef'][i] = (r500ref, Xnew, obserr)



    ########################################
    def getXray(self, catalog, cosmology, scaling):
        """Extract an X-ray observable from the measured profiles. Find radius r at which
            data(r)==scalingRelation(r).

        Returns: ln(likelihood) to account for the fact that the observable is actually a
            function of the parameters.

        Note: Everything in this function is in units of Mpc or Msun, no h!
        """

        E06 = cosmo.Ez(.6, cosmology)

        catalog['XrayRef'] = [None for i in range(len(catalog['SPT_ID']))]

        for i,name in enumerate(catalog['SPT_ID']):
            ##### Exclude double entries
            if (name,catalog['field'][i]) in self.SPTdoubleCount: continue
            if catalog['Mg'][i][0,0]!=0.:

                E_z = cosmo.Ez(catalog['redshift'][i], cosmology)
                rho_c_z = cosmo.RHOCRIT * (cosmology['h']*E_z)**2.
                dA = cosmo.AngDiamDist(catalog['redshift'][i], cosmology) / cosmology['h']

                # Angular diameter distance in reference cosmology
                dAref = cosmo.AngDiamDist(catalog['redshift'][i], self.cosmologyRef) / self.cosmologyRef['h']

                ##### Initial M500 guess, and observational error
                if self.todo['Xobs']=='Yx':
                    Xold = 1.
                    obserr = catalog['lnYx_err'][i]
                elif self.todo['Xobs']=='Mg':
                    Xold = 1e13
                    obserr = catalog['lnMg_err'][i]

                nonzero = np.nonzero(catalog['Mg'][i][0])[0]
                for j in range(100):
                    if self.todo['Xobs'] == 'Yx':
                        if self.YXPARAM=='SPT_XVP':
                            M500 = 1e14 * scaling['Ax'] * cosmology['h']**.5 * (cosmology['h']/.72)**(2.5*scaling['Bx']-1.5) * (Xold/3)**scaling['Bx'] * E_z**scaling['Cx']
                        elif self.YXPARAM=='Munich':
                            M500 = self.XraymPivot * cosmology['h']**.5 * (Xold/(scaling['Ax']*(E_z/E06)**scaling['Cx']))**(1/scaling['Bx'])
                    elif self.todo['Xobs'] == 'Mg':
                        M500 = self.XraymPivot * (Xold/self.XraymPivot/self.scalingRef['Ax'] / (E_z/E06)**self.scalingRef['Cx'])**(1/self.scalingRef['Bx'])

                    r500 = 1000 * (3*M500/(4*np.pi*500*rho_c_z))**(1/3)
                    r500ref = r500 * (dAref/dA)

                    MgRef = np.interp(r500ref, catalog['Mg'][i][0,nonzero], catalog['Mg'][i][1,nonzero])
                    Mg = MgRef * (dA/dAref)**2.5

                    Xnew = Mg
                    if self.todo['Xobs']=='Yx':
                        Xnew*= 1e-14*catalog['Tx'][i]

                    if np.absolute(Xold/Xnew-1)<1e-4: break

                    Xold = Xnew

                if j==99:
                    print name,r500ref,'no convergence after 100 steps'
                if r500ref>catalog['Mg'][i][0,nonzero][-1]:
                    print 'Warning,',catalog['SPT_ID'][i],'r500ref', r500ref

                catalog['XrayRef'][i] = (r500ref, Xnew, obserr)
