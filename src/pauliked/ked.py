from .tf import get_tautf
from .s_sqr_q import get_s_sqr, get_q


def ked(n,absdeln,lapn,Functional,*args):                      #   second arg was absdeln#    in general use Functional instead of F = Fpc_or_cr, Fs2, Fq
    
    #get basics: tauTF, s2, q and derivatives
    tau_tf, dtau_by_dn = get_tautf(n)
    s_sqr, ds2_by_dn,ds2_by_deln = get_s_sqr(n,absdeln)   #absdeln
   
    q, dq_by_dn, dq_by_lapn = get_q(n,lapn)
    
    #kinetic energy
    #print("arguments are now", args)
    #print("try", *args)
    F, dF_by_ds_sqr, dF_by_dq = Functional(s_sqr,q,*args)
    KED = tau_tf * F
    
    #partial with respect to density
    term1 = F * dtau_by_dn
    #print ("F, dtau_by_dn", F, dtau_by_dn)
    #print ("term1", term1)
    term2 = dF_by_ds_sqr * ds2_by_dn  + dF_by_dq * dq_by_dn
    #print ("dF_by_dq", dF_by_dq)
    #print ("dF_by_ds_sqr", dF_by_ds_sqr)
    #print ("ds2_by_dn", ds2_by_dn)
    term3 = tau_tf * term2
    #print ("term3", term3)
   
    KED_n = term1 + term3
    #print ("KED_n", KED_n)
    
    #partial with respect to gradient of density
    term4 = dF_by_ds_sqr * ds2_by_deln       # deln = dn/dr
    term5 = tau_tf * term4
    absKED_deln = term5
    
    #partial with respect to laplacian of density
    term6 = dF_by_dq * dq_by_lapn
    term7 = tau_tf * term6
    KED_lapn = term7
    
    return KED, KED_n, absKED_deln, KED_lapn
