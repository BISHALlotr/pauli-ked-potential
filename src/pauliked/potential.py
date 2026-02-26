#!/usr/bin/env python
from .laplacian import RadialGradient,RadialDivergence,RadialLaplacian
from .ked import ked

__all__ = ["Potential","potential"]

def Potential(r,n,absgradn,lapln,functional,nn,mm,*args):
    """ Calculates KED and three parts of the Kinetic energy potential
    Input variables:   
       r = array of positions
       n = array of density
       absgradn = array of gradients of the density
       lapln = array of laplacians
       functional = function that calculates enhancement factor and derivatives
       nn = how many grid points (per side) to use in derivatives
       mm = how many grid points per side to use in binomial convolution
       *args = possible extra arguments to be used by functional
    Output 
       KED -- kinetic energy density
       v1 -- variation of tau with local density
       v2 -- variation of tau with gradient of density
       v3 -- variation of tau with laplacian
       E -- integral of this squared gives smoothness
    """
    
    KED, KED_n, KED_deln, KED_lapn = ked(n,absgradn,lapln, functional,*args)
    #print(KED)
    v1 = KED_n
    v2 = RadialDivergence(KED_deln,r,nn,mm)
    v3 = RadialLaplacian(KED_lapn,r,nn,mm)
    pseudo_electric_field = RadialGradient(KED_lapn,r,nn,mm) ##- KED_deln
    E = pseudo_electric_field
    
    return KED, v1, v2, v3, E

# --- lowercase production-style alias ---
def potential(r, n, absgradn, lapln, functional, nn, mm, *args):
    return Potential(r, n, absgradn, lapln, functional, nn, mm, *args)
