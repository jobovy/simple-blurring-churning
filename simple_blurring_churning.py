###############################################################################
# Simple models of the effect of blurring and churning on the properties of 
# the Milky Way
###############################################################################
from functools import wraps
import numpy
from galpy.orbit import Orbit
from galpy.df import dehnendf
from scipy import integrate
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
            0.,10.,tol=10.**-4.,rtol=10**-3.,vec_func=False)[0] #hack
    return out

