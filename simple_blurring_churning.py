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

