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
_LINEARENRICHMENT= False
_TAUEQ= 2.
_ZINIT= 0.12
_FMIG= 0.2
# defaults
_SKEWM_DEFAULT= 0.4
_SKEWS_DEFAULT= 0.1
_SKEWA_DEFAULT= -4.
_DFEHDR_DEFAULT= -0.1
def scalarDecorator(func):
    """Decorator to return scalar outputs"""
    @wraps(func)
    def scalar_wrapper(*args,**kwargs):
        if numpy.array(args[0]).shape == ():
            scalarOut= True
            newargs= ()
            for ii in range(len(args)):
                if ii == 0:
                    newargs= newargs+(numpy.array([args[ii]]),)
                else:
                    newargs= newargs+(args[ii],)
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
    sig= (0.01+_FMIG*tau*Rgi*numpy.exp(-(Rgi-8.)**2./16.))
    return 1./numpy.sqrt(2.*numpy.pi)\
        *numpy.exp(-(Rgi-Rgf)**2./2./sig)

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
           skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,skewa=_SKEWA_DEFAULT,
           dFehdR=_DFEHDR_DEFAULT):
    """
    NAME:
       pFehRg
    PURPOSE:
       The initial MDF at a given radius Rg
    INPUT:
       Feh - Metallicity
       Rg - Radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient
    OUTPUT:
       p(Feh|Rg) at the initial time
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    return skewnormal(Feh,m=skewm+dFehdR*(Rg-4.),s=skews,a=skewa)\
        *0.5*(1.+numpy.tanh((Feh-numpy.log10(_ZINIT))/0.2))

def pAgeRg(age,Rg,
           skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,skewa=_SKEWA_DEFAULT,
           dFehdR=_DFEHDR_DEFAULT):
    """
    NAME:
       pAgeRg
    PURPOSE:
       The initial age DF at a given radius Rg
    INPUT:
       age - age (/Gyr)
       Rg - Radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient
    OUTPUT:
       p(age|Rg) at the initial time
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    ageFeh= fehAgeRg(age,Rg,skewm=skewm,dFehdR=dFehdR)
    return pFehRg(ageFeh,Rg,skewm=skewm,skews=skews,skewa=skewa,
                  dFehdR=dFehdR)\
                  /numpy.fabs(_dagedFehRg(ageFeh,Rg,skewm=skewm,dFehdR=dFehdR))

# The relation between age and metallicity at a given radius
def fehAgeRg(age,Rg,skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,dFehdR=_DFEHDR_DEFAULT):
    """
    NAME:
       fehAgeRg
    PURPOSE:
       The metallicity corresponding to a given age at radius Rg; assuming linear increase in exp(Feh) with time from Zinit Zsolar
    INPUT:
       age - age (/Gyr)
       Rg - guiding-center radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient
    OUTPUT:
       FeH(age,Rg)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    if _LINEARENRICHMENT:
        return numpy.log10(_ZINIT+(10.-age)/10.*(10.**(skews+skewm+dFehdR*(Rg-4.))-_ZINIT))
    else:
        eq= 10.**(skews+skewm+dFehdR*(Rg-4.))
        return numpy.log10((eq-_ZINIT)*(1.-numpy.exp(-(10.-age)/_TAUEQ))+_ZINIT)

def ageFehRg(feh,Rg,skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,dFehdR=_DFEHDR_DEFAULT):
    """
    NAME:
       ageFehRg
    PURPOSE:
       The age corresponding to a given metallicity at radius Rg; assuming linear increase in exp(Feh) with time from _ZINIT Zsolar
    INPUT:
       feh - metallicity
       Rg - guiding-center radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient  
    OUTPUT:
       age(FeH,Rg)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    if _LINEARENRICHMENT:
        return 10.-10.*(10.**feh-_ZINIT)/((10.**(skews+skewm+dFehdR*(Rg-4.))-_ZINIT))
    else:
        eq= 10.**(skews+skewm+dFehdR*(Rg-4.))
        return 10.+numpy.log(1.-(10.**feh-_ZINIT)/(eq-_ZINIT))*_TAUEQ

def RgAgeFeh(age,feh,
             skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,dFehdR=_DFEHDR_DEFAULT):
    """
    NAME:
       RgAgeFeh
    PURPOSE:
       The guiding-center radius corresponding to a given metallicity and age; assuming linear increase in exp(Feh) with time from _ZINIT Zsolar
    INPUT:
       age - age (/Gyr)
       feh - metallicity
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient  
    OUTPUT:
       Rg(age,FeH)
    HISTORY:
       2015-01-13 - Written - Bovy (IAS)
    """
    if _LINEARENRICHMENT:
        return (numpy.log10(10.*(10.**feh-_ZINIT)/(10.-age))-skewm-skews)/dFehdR+4.
    else:
        return (numpy.log10((10.**feh-_ZINIT)/(1.-numpy.exp(-(10.-age)/_TAUEQ))+_ZINIT)-skews-skewm)/dFehdR+4.

# Also need derivatives for integrals and distribution
def _dfehdAgeRg(age,Rg,skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,dFehdR=_DFEHDR_DEFAULT):
    if _LINEARENRICHMENT:
        return -1./10./numpy.log(10.)*(10.**(skews+skewm+dFehdR*(Rg-4.))-_ZINIT)\
        /(_ZINIT+(10.-age)/10.*(numpy.exp(skews+skewm+dFehdR*(Rg-4.))-_ZINIT))
    else:
        eq= 10.**(skews+skewm+dFehdR*(Rg-4.))
        return -(eq-_ZINIT)*numpy.exp(-(10.-age)/_TAUEQ)/(((eq-_ZINIT)*(1.-numpy.exp(-(10.-age)/_TAUEQ))+_ZINIT))/numpy.log(10.)/_TAUEQ

def _dagedFehRg(feh,Rg,skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,dFehdR=_DFEHDR_DEFAULT):
    if _LINEARENRICHMENT:
        return -10.*10.**feh*numpy.log(10.)\
            /((10.**(skews+skewm+dFehdR*(Rg-4.))-_ZINIT))
    else:
        eq= 10.**(skews+skewm+dFehdR*(Rg-4.))
        return -_TAUEQ*numpy.log(10.)*10.**feh/(eq-_ZINIT)/(1.-(10.**feh-_ZINIT)/(eq-_ZINIT))

def _dfehdRgAge(Rg,age,skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,dFehdR=_DFEHDR_DEFAULT):
    feh= fehAgeRg(age,Rg,skewm=skewm,skews=skews,dFehdR=dFehdR)
    if _LINEARENRICHMENT:
        return (10.-age)/10.*10.**(skews+skewm+dFehdR*(Rg-4.))*dFehdR/10.**feh
    else:
        eq= 10.**(skews+skewm+dFehdR*(Rg-4.))
        return (1.-numpy.exp(-(10.-age)/_TAUEQ))*eq*dFehdR/10.**feh

def test_dfehdAgeRg():
    ages= numpy.tile(numpy.linspace(1.,10.,101),(101,1))
    Rs= numpy.tile(numpy.linspace(2.,16.,101),(101,1)).T
    dx= 10.**-8.
    dage= _dfehdAgeRg(ages,Rs)
    dage_num= (fehAgeRg(ages+dx,Rs)-fehAgeRg(ages,Rs))/dx
    assert numpy.all(numpy.fabs(dage-dage_num) < 10.**-4.), 'dfehdAgeRg implemented incorrectly'
    return None

def test_dagedFgeRg():
    Rs= numpy.tile(numpy.linspace(2.,16.,101),(101,1)).T
    fehs= numpy.tile(numpy.linspace(-1.5,0.7,101),(101,1))
    Rs[fehs > fehAgeRg(0.,Rs)-0.03]= numpy.nan
    dx= 10.**-8.
    dfeh= _dagedFehRg(fehs,Rs)
    dfeh_num= (ageFehRg(fehs+dx,Rs)-ageFehRg(fehs,Rs))/dx
    assert numpy.all((numpy.fabs(dfeh-dfeh_num) < 10.**-4.)+numpy.isnan(dfeh)), 'dagedFehRg implemented incorrectly'
    return None

def test_dfehdRgAge():
    Rs= numpy.tile(numpy.linspace(2.,16.,101),(101,1)).T
    ages= numpy.tile(numpy.linspace(1.,9.9,101),(101,1))
    dx= 10.**-8.
    dfeh= _dfehdRgAge(Rs,ages)
    dfeh_num= (fehAgeRg(ages,Rs+dx)-fehAgeRg(ages,Rs))/dx
    assert numpy.all((numpy.fabs(dfeh-dfeh_num) < 10.**-6.)+numpy.isnan(dfeh)), 'dfehdRgAge implemented incorrectly'
    return None

# Blurring MDF
@scalarDecorator
def blurring_pFehR(feh,R,
                   skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,skewa=_SKEWA_DEFAULT,
                   dFehdR=_DFEHDR_DEFAULT,
                   sr=31.4,hr=3.,hs=267.):
    """
    NAME:
       blurring_pFehR
    PURPOSE:
       The distribution of metallicities at a given R due to blurring
    INPUT:
       feh - metallicity
       R - radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient
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
                                                       dFehdR=dFehdR)\
                                          *blurring_pRgR(x,R,sr=sr,
                                                         hr=hr,hs=hs),
                                      numpy.amax([0.,R-4.]),R+4.,
                                      tol=10.**-4.,rtol=10.**-3.,
                                      vec_func=False)[0]
    return out

# Churning age distribution
@scalarDecorator
def churning_pAgeR(age,R,
                   skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,skewa=_SKEWA_DEFAULT,
                   dFehdR=_DFEHDR_DEFAULT,Rd=2.2,
                   sr=31.4,hr=3.,hs=267.):
    """
    NAME:
       churning_pAgeR
    PURPOSE:
       The distribution of ages at a given R due to churning
    INPUT:
       age - age (/Gyr)
       R - radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient
       Rd= (2.2 kpc) mass scale length of the disk
       sr= (31.4 km/s) velocity dispersion at R0
       hr= (3 kpc) scale length
       hs= (267 kpc) dispersion scale length
    OUTPUT:
       p(age|R)
    HISTORY:
       2015-01-12 - Written - Bovy (IAS)
    """
    out= numpy.empty_like(age)
    for ii in range(len(age)):
        out[ii]= integrate.quadrature(\
            lambda x: pFehRg(fehAgeRg(age[ii],x,skewm=skewm,skews=skews,
                                      dFehdR=dFehdR),x,
                             skewm=skewm,skews=skews,
                             skewa=skewa,
                             dFehdR=dFehdR)\
                *churning_pRgR(x,R,Rd=Rd,sr=sr,
                               hr=hr,hs=hs)\
                /numpy.fabs(_dagedFehRg(fehAgeRg(age[ii],x,skewm=skewm,skews=skews,dFehdR=dFehdR),x)),
            numpy.amax([0.,R-4.]),R+6.,
            tol=10.**-4.,rtol=10.**-3.,
            vec_func=False)[0]
    return out

# Churning metallicity distribution
@scalarDecorator
def churning_pFehR(feh,R,
                   skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,
                   skewa=_SKEWA_DEFAULT,
                   dFehdR=_DFEHDR_DEFAULT,Rd=2.2,
                   sr=31.4,hr=3.,hs=267.,
                   useInitialAgeDF=True):
    """
    NAME:
       churning_pFehR
    PURPOSE:
       The distribution of metallicities at a given R due to churning
    INPUT:
       feh - metallicity
       R - radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient
       Rd= (2.2 kpc) mass scale length of the disk
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
        # shortcut for Age DF
        if useInitialAgeDF:
            ageDF= lambda a: pAgeRg(a,R,skewm=skewm,skews=skews,skewa=skewa,
                                    dFehdR=dFehdR)
        else:
            ageDF= lambda a: churning_pAgeR(a,R,skewm=skewm,skews=skews,
                                            skewa=skewa,dFehdR=dFehdR,Rd=Rd,
                                            sr=sr,hr=hr,hs=hs)
        # Short age function, so we don't have to repeat this
        ageFunc= lambda r: ageFehRg(feh[ii],r,skewm=skewm,skews=skews,
                                    dFehdR=dFehdR)
        # Integrate
        def intFunc(x):
            tage= ageFunc(x)
            if tage <= 0. or tage > 10. or numpy.isnan(tage):
                return 0.
            return ageDF(ageFunc(x))\
                                *churning_pRgRtau(x,R,tage,
                                                  Rd=Rd,sr=sr,
                                                  hr=hr,hs=hs)\
                /numpy.fabs(_dfehdAgeRg(tage,x))
        out[ii]= integrate.quad(intFunc,
                                      numpy.amax([0.,R-12.]),(feh[ii]-skewm-skews)/dFehdR+4.)[0]
    return out

# Churning metallicity distribution
@scalarDecorator
def churning_pFehAgeR(feh,age,R,
                      skewm=_SKEWM_DEFAULT,skews=_SKEWS_DEFAULT,
                      skewa=_SKEWA_DEFAULT,
                      dFehdR=_DFEHDR_DEFAULT,Rd=2.2,
                      sr=31.4,hr=3.,hs=267.,
                      useInitialAgeDF=True):
    """
    NAME:
       churning_pFehAgeR
    PURPOSE:
       The distribution of metallicities and ages at a given R due to churning
    INPUT:
       feh - metallicity (can be array)
       age - age (/Gyr)
       R - radius (/kpc)
       skewm= (0.8) mean of the initial MDF at 4 kpc
       skews= (0.2) standard dev. of the initial MDF
       skewa= (-4.) skewness parameter of the initial MDF
       dFehdR= (-0.15) initial metallicity gradient
       Rd= (2.2 kpc) mass scale length of the disk
       sr= (31.4 km/s) velocity dispersion at R0
       hr= (3 kpc) scale length
       hs= (267 kpc) dispersion scale length
    OUTPUT:
       p(Feh,age|R)
    HISTORY:
       2015-01-13 - Written - Bovy (IAS)
    """
    out= numpy.empty_like(feh)
    # p(age|R)
    if useInitialAgeDF:
        ageP= pAgeRg(age,R,skewm=skewm,skews=skews,skewa=skewa,
                     dFehdR=dFehdR)
    else:
        ageP= churning_pAgeR(age,R,skewm=skewm,skews=skews,
                             skewa=skewa,dFehdR=dFehdR,Rd=Rd,
                             sr=sr,hr=hr,hs=hs)
    for ii in range(len(feh)):
        trg= RgAgeFeh(age,feh[ii],
                      skewm=skewm,skews=skews,dFehdR=dFehdR)
        if trg <= 0. or numpy.isnan(trg) or numpy.isinf(trg) \
                or feh[ii] > (skews+skewm+dFehdR*(trg-4.)):
            out[ii]= 0.
            continue
        out[ii]= \
            churning_pRgRtau(trg,R,age,Rd=Rd,sr=sr,hr=hr,hs=hs)\
            *ageP/_dfehdRgAge(trg,age,skewm=skewm,skews=skews,dFehdR=dFehdR)
    return out

def skewness(x,mdf):
    m= numpy.nansum(x*mdf)/numpy.nansum(mdf)
    return numpy.nansum((x-m)**3.*mdf)/numpy.nansum(mdf)\
        /(numpy.nansum((x-m)**2*mdf)/numpy.nansum(mdf))**1.5
