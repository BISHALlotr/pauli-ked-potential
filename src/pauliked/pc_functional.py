'''  Laplacian-level density functionals for the kinetic energy density and exchange-correlation energy
     John P. Perdew and Lucian A. Constantin
     equation 11,15'''


from numpy import exp
from .vw import FvW
from .gea import F2, F4
from .s_sqr_q import  get_s_sqr, get_q
from numpy import zeros,sqrt,exp
import numpy as np

def x_RPP(s_sqr, q):  # x
    '''optimized value of paramters'''
    c1 = 0.201352
    c2 = 0.185020
    c3 = 1.53804    
    def x4(s_sqr, q):
        bqq = 1.801019
        bpq = -1.850497
        bpp = 0.974002
        x4value = bqq* q**2 + bpq* s_sqr* q + (bqq-c3)* s_sqr**2
        return x4value
    
    Zvalue = 1- 40/27*s_sqr + 20/9* q + c3 * s_sqr**2 *exp(-abs(c3)* s_sqr)  
    + x4(s_sqr,q)*exp(-(s_sqr/c1)**2 - (q/c2)**2)
    return Zvalue

def Z_RPP(s_sqr,q):  # created to be consistent with our notation.
    x = x_RPP(s_sqr,q)
    return x-1

def alpha_RPP(x_optimized):
    
    x0 = 0.819411
    A = 20/x0**3
    B = -45/x0**4
    C = 36/x0**5
    D = -10/x0**6
    xvalues  = []
    for x in x_optimized:
        if (x<0):
            result = 0.0 
        elif (x>=0) & (x <= x0):
            result = x**4*(A + B*x + C*x**2 + D * x**3)
        else:
            result = x
        xvalues.append(result)
    
    return xvalues

def get_Zpc(s_sqr,q):  
 
    fvw, dfvwds2,dfvwdq = FvW(s_sqr,q)
    f4, f4s_sqr, f4q = F4(s_sqr,q)
    f2, f2s_sqr, f2q = F2(s_sqr,q)
    delf4 = f4-f2       
    delf2 = f2-1.0
    delf4_ds_sqr = f4s_sqr - f2s_sqr
    delf2_ds_sqr = f2s_sqr - 0.0
    delf4_dq = f4q - f2q
    delf2_dq = f2q - 0.0
    
    A = delf4/(1+fvw)
    As2 = delf4_ds_sqr/(1+fvw) - delf4*dfvwds2/(1+fvw)**2
    Aq = delf4_dq/(1+fvw)
    
    Num = 1.0 + delf2 + delf4      #delf2 = F2 only, delf4 = F4 only
    Nums2 = delf4_ds_sqr + delf2_ds_sqr
    Numq = delf4_dq + delf2_dq
    
    Deno = np.sqrt(1 + A**2)
    Denos2 = (A/Deno)*As2
    Denoq = (A/Deno)* Aq
    
    Fmge4 = Num/Deno  
    Fmge4s2 = (Nums2/Deno) - (Num/Deno**2)*Denos2
    Fmge4q = (Numq/Deno) - (Num/Deno**2)*Denoq
            
    Zpc = Fmge4 - fvw
    dZpc_dq = Fmge4q    #
    dZpc_ds_sqr = Fmge4s2 - dfvwds2                         
    return Zpc, dZpc_ds_sqr, dZpc_dq

def get_THETApc(Z):  
    '''Perdew-Constantin switch
    z is linear combination of p and q (Mejia/Trickey Eq. 30)
    '''
    a = 0.5389
    b = 3.0 
    small = a*0.01 #0.005   # edited 25 2022
   
    c = exp( a/np.maximum((a-Z),small) )
    d = exp( a/np.maximum(Z,small) )
    ddbydz = (-a*d/Z**2)
    dcbydz = (a*c/(a-Z)**2)
    
    #print ("in get_THETApc")
    #print (Z)
    #print (Z>0)
    #part1 = 0.0 * (Z<0)
    #part2 = ((1+c)/(d+c))**b * (Z>0) * (Z<a)
    #part3 = 1.0 * (Z>a)
    part1 = 0.0 * (Z<small)  # edited 25 2022
    part2 = ((1+c)/(d+c))**b * (Z>small) * (Z<(a-small)) # edited 25 2022
    part3 = 1.0 * (Z>(a-small))     # edited 25 2022
    e = (dcbydz) - ((1 + c)/(d + c)) * (ddbydz + dcbydz)
    Thetapc = part1 + part2 + part3
    #print(Thetapc)
    dTheta_dz =  b * Thetapc * 1/(1+c) * e *(Z>small)*(Z<(a-small))                                                                            
    return Thetapc, dTheta_dz 

def get_THETApcopt(Z):  
    '''Perdew-Constantin switch, optimized by Mejia/Trickey, 2017.
    z is linear combination of p and q (Mejia/Trickey Eq. 30)
    '''
    a = 1.784720
    b = 0.258304 
    small = a*0.01 #0.005   # edited 25 2022
   
    c = exp( a/np.maximum((a-Z),small) )
    d = exp( a/np.maximum(Z,small) )
    ddbydz = (-a*d/Z**2)
    dcbydz = (a*c/(a-Z)**2)
    
    part1 = 0.0 * (Z<small)  # edited 25 2022
    part2 = ((1+c)/(d+c))**b * (Z>small) * (Z<(a-small)) # edited 25 2022
    part3 = 1.0 * (Z>(a-small))     # edited 25 2022
    e = (dcbydz) - ((1 + c)/(d + c)) * (ddbydz + dcbydz)
    Thetapc = part1 + part2 + part3
    #print(Thetapc)
    dTheta_dz =  b * Thetapc * 1/(1+c) * e *(Z>small)*(Z<(a-small))                                                                            
    return Thetapc, dTheta_dz 
                 
def F_pc(s_sqr,q,*args):
     
    Zpc,dZpc_ds_sqr,dZpc_dq = get_Zpc(s_sqr,q)                                         
    THETA_pc, dTHETA_pc = get_THETApc(Zpc) 
    fvw, fvws2,fvwq = FvW(s_sqr,q)
    F = fvw + Zpc*THETA_pc
    Fs2 = fvws2 + dZpc_ds_sqr*( THETA_pc + Zpc*dTHETA_pc ) 
    Fq =  dZpc_dq*( THETA_pc + Zpc*dTHETA_pc )    
    return F, Fs2, Fq
                 
def Fpcopt(s_sqr,q,*args):
     
    Zpc,dZpc_ds_sqr,dZpc_dq = get_Zpc(s_sqr,q)                                         
    THETA_pc, dTHETA_pc = get_THETApcopt(Zpc) 
    fvw, fvws2,fvwq = FvW(s_sqr,q)
    F = fvw + Zpc*THETA_pc
    Fs2 = fvws2 + dZpc_ds_sqr*( THETA_pc + Zpc*dTHETA_pc ) 
    Fq =  dZpc_dq*( THETA_pc + Zpc*dTHETA_pc )    
    return F, Fs2, Fq

def F_RPP(s_sqr,q,*args):  
    x = x_RPP(s_sqr,q)
    #Z = Z_RPP(s_sqr,q)
    F =   FvW(s_sqr,q)[0] + alpha_RPP(x) 
    # To be filled in later. . .
    dF_dq = 0.0* np.ones(len(F))
    dF_ds_sqr = 0.0* np.ones(len(F))    
    return F, dF_ds_sqr, dF_dq

def Fpgconst(s_sqr,q,*args):
    '''General exponential functional developed by the team of Constantin, Fabinio, sala'''
    beta = 8.0/81.0
    c1 = 40.0/27.0
    e = np.exp(-c1*s_sqr)     #add von-weizsacker
    q_sqr = q*q
    fvw, dfvwds2, dfvwdq   = FvW(s_sqr,q)        # added third one

    f = e + beta*q_sqr + fvw
    dfds2 = -c1 * e + dfvwds2
    dfdq = 2.0*beta*q + dfvwdq

    return f, dfds2, dfdq

def Flkt(s_sqr,q):
    '''Luo-Trickey functional.
    A set of pseudodensities was generated for the atoms H through Ne with a typical Hamann norm-conserving nonlocal pseudopotential scheme using default radii.  Then a was found such that all post-scf Pauli potentials from those pseudodensities satisified v-theta >= 0.
    '''

    a = 1.3
    b = 5.0/3.0
    s = np.sqrt(s_sqr)
    flkt = 1/np.cosh(a*s) + b*s_sqr
    dfds2 = -a/(2*s)*1/np.cosh(a*s)*np.tanh(a*s) + b
    dfdq = np.zeros(len(flkt))
    return flkt, dfds2, dfdq

def Fvt84(s_sqr,q,*args):
    '''Valentin V. Karasiev,* Debajit Chakraborty, Olga A. Shukruto, and S. B. Trickey
    Nonempirical generalized gradient approximation free-energy functional for orbital-free simulations'''
    n = 4.0
    m = 8.0
    mu = 2.778
    alpha =  1.2965     #mu - (5.0/3.0) + (5.0/27.0)
    s = sqrt(s_sqr)

    frac = mu* s_sqr * exp(-alpha*s_sqr)/(1 + mu * s_sqr)
    sfact = (1 - exp(-alpha*s**(4.0))) * (s**(-2.0)-1)      #as it contain term of s named as s factors
    fvt84 = 1.0 - frac + sfact + (5.0/3.0) * s_sqr       # this part is for vw

    #these are the part of expression for dfds2
    #second term in KEDf derivative
    frac21 = (alpha*s_sqr-1)*mu*exp(-alpha*s_sqr)/(1+mu*s_sqr)
    frac22 = mu**2*s_sqr*exp(-alpha*s_sqr)/(1+mu*s_sqr)**2
    df2ds2 = frac21 + frac22
    #third term in KEDf derivative
    first = 2.0*alpha*(1 - s_sqr)*exp(-alpha*s_sqr**2)
    secon = -(1 - exp(-alpha*s_sqr**2))/s_sqr**2
    df3ds2 = first + secon

    dfds2 = df3ds2 + df2ds2 + (5.0/3.0)
    try:
        zero = np.zeros(len(fvt84))
    except TypeError:
        zero = 0.0
    dfdq = zero


    return fvt84, dfds2, dfdq
