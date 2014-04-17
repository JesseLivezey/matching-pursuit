#This file will time various versions of LCA
from __future__ import division
import numpy as np
from timeit import default_timer as timer

import MPnumpy as mpn
import MPnumbaprog as mpg

def main():
    """Profiles various versions of Matching Pursuit."""

    nshort = 6
    tshort = 2
    nmed = 3
    tmed = 6
    nlong = 1
    
    #Setup variables for inference
    numDict = int(9600)
    numBatch = int(1024)
    dataSize = int(256)
    dictsIn = np.random.randn(numDict,dataSize)
    dictsIn = (np.sqrt(np.diag(1/np.diag(dictsIn.dot(dictsIn.T))))).dot(dictsIn)
    stimuli = np.random.randn(numBatch,dataSize)
    kin = 32
    minabsin = 0.
    
    #MP
    params = """Parameters:
             numDict: """+str(numDict)+"""
             numBatch: """+str(numBatch)+"""
             dataSize: """+str(dataSize)+"""
             k-sparseness: """+str(kin)+"""\n"""
    print params
             
    start = timer()
    mpn.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            mpn.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            mpn.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            mpn.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '---------------Numpy based MP---------------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt

    start = timer()
    mpg.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            mpg.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            mpg.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            mpg.mp(dictsIn,stimuli,k=kin,minabs=minabsin)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '----------NumbaPro GPU based MP---------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt
    


if __name__ == '__main__':
    main()
