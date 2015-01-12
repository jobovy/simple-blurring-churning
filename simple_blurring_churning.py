###############################################################################
# Simple models of the effect of blurring and churning on the properties of 
# the Milky Way
###############################################################################
from functools import wraps
import numpy
from scipy import integrate
from galpy.orbit import Orbit
from galpy.df import dehnendf
from skewnormal import skewnormal
_R0= 8. #kpc
_V0= 220. #kms
def scalarDecorator(func):
    """Decorator to return scalar outputs"""
    @wraps(func)
    def scalar_wrapper(*args,**kwargs):
        if numpy.array(args[0]).shape == ():
            scalarOut= True
            newargs= ()
            for ii in range(len(args)):
                newargs= newargs+(numpy.array([args[ii]]),)
            args= newargs
        else:
            scalarOut= False
        result= func(*args,**kwargs)
        if scalarOut:
            return result[0]
        else:
            return result
    return scalar_wrapper

#
# PURE DYNAMICS
#

# Blurring p(Rg|R)
@scalarDecorator
def blurring_pRgR(Rg,R,sr=31.4,hr=3.,hs=267.):
    """
    NAME:
       blurring_pRgR
    PURPOSE:
       The distribution of guiding-center radii at a given R from blurring
    INPUT:
       Rg - Guiding center radius (/kpc), can be array
       R - Given radius (/kpc)
       sr= (31.4 km/s) velocity dispersion at R0
       hr= (3 kpc) scale length
       hs= (267 kpc) dispersion scale length
    OUTPUT:
       p(Rg|R)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    # Setup the DF
    df= dehnendf(beta=0.,profileParams=(hr/_R0,hs/_R0,sr/_V0))
    out= numpy.empty(len(Rg))
    for ii in range(len(Rg)):
        out[ii]= df(Orbit([R/8.,0.,Rg[ii]/R]))
    return out

# Churning p(final Rg | initial Rg, tau)
@scalarDecorator
def churning_pRgfRgi(Rgf,Rgi,tau,Rd=2.2):
    """
    NAME:
       churning_pRgfRgi
    PURPOSE:
       The distribution of final guiding-center radii from churning
    INPUT:
       Rgf - Guiding center radius (/kpc), can be array
       Rgi - Initial guiding-center radius (/kpc)
       tau - time (/Gyr)
       Rd= (2.2 kpc) mass scale length of the disk
    OUTPUT:
       p(Rgf|Rgi)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    return 1./numpy.sqrt(2.*numpy.pi*(0.01+tau*Rgi*numpy.exp(-Rgi/Rd/2.)))\
        *numpy.exp(-(Rgi-Rgf)**2./2./(0.01+tau*Rgi*numpy.exp(-Rgi/Rd/2.)))

# Churning p(Rg|R,tau)
@scalarDecorator
def churning_pRgRtau(Rg,R,tau,Rd=2.2,sr=31.4,hr=3.,hs=267.):
    """
    NAME:
       churning_pRgRtau
    PURPOSE:
       The distribution of guiding-center radii at a given radius and time from churning
    INPUT:
       Rg - Guiding center radius (/kpc), can be array
       R - Given radius (/kpc)
       tau - time (/Gyr)
       Rd= (2.2 kpc) mass scale length of the disk
       sr= (31.4 km/s) velocity dispersion at R0
       hr= (3 kpc) scale length
       hs= (267 kpc) dispersion scale length
    OUTPUT:
       p(Rg|R,tau)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    # Setup the DF
    df= dehnendf(beta=0.,profileParams=(hr/_R0,hs/_R0,sr/_V0))
    out= numpy.empty(len(Rg))
    for ii in range(len(Rg)):
        out[ii]= integrate.fixed_quad(lambda x: df(Orbit([R/8.,0.,x/R]))\
                                          *churning_pRgfRgi(x,Rg[ii],tau,
                                                            Rd=Rd),
                                      numpy.amax([Rg[ii]-4.,0.]),
                                      Rg[ii]+6.,n=40)[0]
    return out

# Churning p(Rg|R)
@scalarDecorator
def churning_pRgR(Rg,R,Rd=2.2,sr=31.4,hr=3.,hs=267.):
    """
    NAME:
       churning_pRgR
    PURPOSE:
       The distribution of guiding-center radii at a given radius from churning (assume constant SFH)
    INPUT:
       Rg - Guiding center radius (/kpc), can be array
       R - Given radius (/kpc)
       Rd= (2.2 kpc) mass scale length of the disk
       sr= (31.4 km/s) velocity dispersion at R0
       hr= (3 kpc) scale length
       hs= (267 kpc) dispersion scale length
    OUTPUT:
       p(Rg|R)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    # Setup the DF
    df= dehnendf(beta=0.,profileParams=(hr/_R0,hs/_R0,sr/_V0))
    out= numpy.empty(len(Rg))
    for ii in range(len(Rg)):
        out[ii]= integrate.quadrature(\
            lambda tau: integrate.fixed_quad(lambda x: \
                                                 df(Orbit([R/8.,0.,x/R]))
                                             *churning_pRgfRgi(x,Rg[ii],
                                                               tau,Rd=Rd),
                                             numpy.amax([Rg[ii]-4.,0.]),
                                             Rg[ii]+6.,n=40)[0],
            0.,10.,tol=10.**-4.,rtol=10**-3.,vec_func=False)[0]
    return out

#
# MDFs
#

# Initial MDF at different radii
def pFehRg(Feh,Rg,
           skewm=0.2,skews=0.2,skewa=-4.,
           dFehdR=-0.075):
    """
    NAME:
       pFehRg
    PURPOSE:
       The initial MDF at a given radius Rg
    INPUT:
       Feh - Metallicity
       Rg - Radius (/kpc)
       skewm= (0.2) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.075) initial metallicity gradient
    OUTPUT:
       p(Feh|Rg) at the initial time
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    return skewnormal(Feh,m=skewm+dFehdR*(Rg-4.),s=skews,a=skewa)

# The relation between age and metallicity at a given radius
def fehAgeRg(age,Rg,skewm=0.2,dFehdR=-0.075):
    """
    NAME:
       fehAgeRg
    PURPOSE:
       The metallicity corresponding to a given age at radius Rg; assuming linear increase in exp(Feh) with time from 0.05 Zsolar
    INPUT:
       age - age (/Gyr)
       Rg - guiding-center radius (/kpc)
       skewm= (0.2) mean of the initial MDF at 4 kpc
       dFehdR= (-0.075) initial metallicity gradient  
    OUTPUT:
       FeH(age,Rg)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    return numpy.log10(0.05+(10.-age)/10.*(numpy.exp(skewm+dFehdR*(Rg-8.))-0.05))

def ageFehRg(feh,Rg,skewm=0.2,dFehdR=-0.075):
    """
    NAME:
       ageFehRg
    PURPOSE:
       The age corresponding to a given metallicity at radius Rg; assuming linear increase in exp(Feh) with time from 0.05 Zsolar
    INPUT:
       feh - metallicity
       Rg - guiding-center radius (/kpc)
       skewm= (0.2) mean of the initial MDF at 4 kpc
       dFehdR= (-0.075) initial metallicity gradient  
    OUTPUT:
       age(FeH,Rg)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    return 10.-10.*(10.**feh-0.05)/((numpy.exp(skewm+dFehdR*(Rg-8.))-0.05))

# Also need derivatives for integrals

# Blurring MDF
def blurring_pFehR(feh,R,
                   skewm=0.2,skews=0.2,skewa=-4.,
                   dFehdR=-0.075,
                   sr=31.4,hr=3.,hs=267.):
    """
    NAME:
       blurring_pFehR
    PURPOSE:
       The distribution of metallicities at a given R due to blurring
    INPUT:
       feh - metallicity
       R - radius (/kpc)
       skewm= (0.2) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.075) initial metallicity gradient
       sr= (31.4 km/s) velocity dispersion at R0
       hr= (3 kpc) scale length
       hs= (267 kpc) dispersion scale length
    OUTPUT:
       p(Feh|R)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    out= numpy.empty_like(feh)
    for ii in range(len(feh)):
        out[ii]= integrate.quadrature(lambda x: pFehRg(feh[ii],x,
                                                       skewm=skewm,skews=skews,
                                                       skewa=skewa,
                                                       dFehdR=-0.075)\
                                          *blurring_pRgR(x,R,sr=sr,
                                                         hr=hr,hs=hs),
                                      numpy.amax([0.,R-4.]),R+4.,
                                      tol=10.**-4.,rtol=10.**-3.,
                                      vec_func=False)[0]
    return out
