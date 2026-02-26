'''Deorbitalization strategies for meta-generalized-gradient-approximation exchange-correlation functionals
   Author:  Daniel Mejia-Rodriguez, S. B. Trickey 

   PHYSICAL REVIEW A 96, 052512 (2017) equation(27)'''

from numpy import ones, array
from .s_sqr_q import get_s_sqr, get_q
from .vw import FvW

# The following works for arrays OR scalars.  Array for q and s_sqr must be of 
# equal length.

F0 = 1.0
def F2(s_sqr,q): 
    f2 = F0 + 5.0/27.0 * s_sqr + 20.0/9.0 * q
    try:
        one = ones(len(f2))
    except TypeError:
        one = 1.0
    dF2ds_sqr = 5.0/27.0 * one
    dF2dq = 20.0/9.0 * one
    
    return f2, dF2ds_sqr, dF2dq

def F2minusFvW(s_sqr,q):
    f2,f2s2,f2q = F2(s_sqr,q)
    fvw, fvws2, fvwq = FvW(s_sqr,q)
    return f2-fvw, f2s2-fvws2, f2q-fvwq

def F4(s_sqr,q):
    f2, df2ds_sqr, df2dq = F2(s_sqr,q)
    f4 = (8.0/81.0) * q**2 - (1.0/9.0) * s_sqr * q + (8.0/243.0) * s_sqr**2 
    dF4ds_sqr = -(1.0/9.0) * q + (8.0/243.0) * 2 * s_sqr
    dF4dq = 8.0/81.0 * 2 * q - (1.0/9.0) * s_sqr
    
    return f2 + f4, df2ds_sqr+dF4ds_sqr, df2dq+dF4dq

def F4minusF2(s_sqr,q):
    f2,f2s2,f2q = F2(s_sqr,q)
    f4, f4s2, f4q = F4(s_sqr,q)
    return f4-f2, f4s2-f2s2, f4q-f2q
