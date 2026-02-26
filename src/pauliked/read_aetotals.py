import numpy as np

def read_aetotals( atom, indirectory ):
    """Read essential data from aetotals -- position, density, KED and vpauli
    Input is name of atom -- "He" or "N76" in Totals directory.
    """
    infilename = "ae_totals_" + atom + ".dat"
    print("#Atom:" + atom)
    print("#File read: " + indirectory + infilename)

    infile=open(indirectory+infilename,"r")

    #calling initial variables
    rlist=[]
    ntotlist=[]
    taukslist=[]
    #These last three are needed to construct the exact Pauli potential.
    tautflist=[]
    plist=[]
    qlist=[]
    Fvwlist=[]
    exactvrlist=[]
    #reads variables creates lists
    for line in infile:
        if line.startswith('#'):
            continue
        else:
            columnsep=line.split()
            rlist +=[columnsep[0]]
            ntotlist +=[columnsep[1]]
            taukslist +=[columnsep[4]]
            tautflist +=[columnsep[5]]
            plist += [columnsep[6]]
            qlist += [columnsep[7]]
            Fvwlist +=[columnsep[10]]
            exactvrlist +=[columnsep[29]]
    #Creates arrays from lists
    r=np.array(rlist,dtype=np.float64)          # radial grid
    ntot=np.array(ntotlist,dtype=np.float64)    # electron density
    tauks=np.array(taukslist,dtype=np.float64)  # KS KED
    tautf=np.array(tautflist,dtype=np.float64)  # Thomas Fermi KED
    Fvw=np.array(Fvwlist,dtype=np.double)      # vW enhancement factor
    exactvr=np.array(exactvrlist,dtype=np.float64) # exact pauli potential
    taup=tauks-(Fvw*tautf)                    # pauli kinetic energy
    vp=taup/ntot+exactvr                      # pauli potential
    #rscale=Z[i]**(1./3.)*r
    #vpunitless=vp*ntot/tautf #unitless pauli potential

    return r, ntot, tauks, vp, exactvr   

if __name__ == "__main__":
    atom = "Rn"
    #path = '/Users/Bishal/Documents/functionalpython/functional/Totals/'
    r, ntot, tauks, vpexact, vr = read_aetotals( atom, indirectory=path )
    import matplotlib.pyplot as plt
    plt.plot(r, 4*np.pi*r**2*ntot, label='radial density')
    plt.show()
    plt.plot(r, 4*np.pi*r**2*tauks, label='KS KED')
    plt.show()
    plt.plot(np.log(r[0:-20]), vpexact[0:-20], label='V Pauli')
    plt.plot(np.log(r[0:-20]), vr[0:-20], label='V Pauli')
    plt.show()
