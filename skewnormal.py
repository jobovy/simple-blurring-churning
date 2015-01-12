import numpy
from scipy.stats import norm
def skewnormal(x,m=0,s=1,a=-4.):
    """Specify in terms of the actual mean and sqrt{variance}"""
    # Calculate the actual parameters
    d= a/numpy.sqrt(1.+a**2.)
    s/= numpy.sqrt(1.-2*d**2./numpy.pi)
    m-= s*d*numpy.sqrt(2./numpy.pi)
    # Evaluate
    t= (x-m)/s
    return 2.*norm.pdf(t)*norm.cdf(a*t)/s
