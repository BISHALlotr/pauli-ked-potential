"""Calculate moment m of radial function fr on logarithmic radial mesh r
 input
        moment  moment m of integral
	fnorm	normalization factor
	r[]	radial logarithmic mesh
	fr[]	radial function array
 local
	mmx	maximum grid index
	al	log(r[n+1]/r[n]) -- needed for getting dr on log mesh.
 output
       fmom = fnorm * int r^2dr { r^m fr(r) }
"""
import numpy
#	
def fmoment(moment,fnorm,r,fr):
#
    fm=fr*r**(moment+3)              #integrand: fr*r^m*r^2*dr with dr=al*r
    al = numpy.log(r[1]/r[0])
#
# integration by simpson rule
    mmax=len(r)
    if((mmax%2) == 0):
        mmax=mmax-1
        fm = fm[0:-1]
         
    imask = numpy.arange(mmax)%2
    esum = numpy.add.reduce(imask*fm)
    #esum=0.0
    #for i in range(1, mmax, 2):          #i=2,max-1,2
    #    esum+=fm[i]

    antimask = 1 - imask[1:-1]
    osum = numpy.add.reduce(antimask*fm[1:-1])
    #osum=0.0
    #for i in range(2, mmax-1, 2):        #do i=3,max-2,2
    #    osum+=fm[i]

    fmom=al*(4.0*esum+2.0*osum+fm[0]+fm[-1])/3.0
    fmom=(fmom+0.5*fm[0])/fnorm
    
    return fmom

# in house test.
if __name__ == "__main__":
    
    #scrounge up a grid -- maxs at 210, 1000 pts within 1 au.
    al = 1.005 
    r0 = 0.01
    nmax = 2000
 
    ri = r0
    rlist = []
    for i in range(nmax):
        rlist += [ri]
        ri = ri*al
    rarray = numpy.array(rlist)

    # make a normalized function -- F*r^2 exp( - A r )
    afactor = 0.80
    norm = afactor**5/24
    farray = norm*rarray**2*numpy.exp( -afactor*rarray )
    
    # integrate and compare to 1. 
    integral = fmoment(0,1.0,rarray,farray)
    answer = 1.0
    print ("<1>", integral, " Error = ", integral - answer)

    # second moment:
    integral = fmoment(2,1.0,rarray,farray)
    answer = 30/afactor**2
    print ("<r^2>", integral, " Error = ", integral - answer)

    # inverse second moment:
    integral = fmoment(-2,1.0,rarray,farray)
    answer = afactor**2/12
    print ("<r^{-2}>", integral, " Error = ", integral - answer)

