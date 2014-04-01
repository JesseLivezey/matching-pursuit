from __future__ import division
import numpy as np
from numbapro import cuda
import numbapro.cudalib.cublas as cublas
from numba import *

@cuda.jit('void(f4[:,:],f4[:,:])')
def cudaABS(stim,output):
    i,j = cuda.grid(2)
    output[i,j] = abs(stim[i,j])

@cuda.jit('void(f4[:,:],i4[:,:],i4)')
def removeWinners(curCoef,winners,k):
    i = cuda.grid(1)
    curCoef[winners[k,i]] = 0.

@cuda.jit('void(f4[:,:],f4[:,:],i4[:,:],i4,i4)')
def maxCoefs(curCoefs,coefs,winners,k,maxLoc):
    i = cuda.grid(1)
    #This is not a great idea. Does cuda do inf? What is largest negative numer?
    maxVal = -10000.
    length = curCoefs.shape[1]
    for jj in xrange(length):
        if curCoefs[i,jj] >= maxVal:
            maxVal = curCoefs[i,jj]
            maxLoc = jj
    winners[k,i] = maxLoc
    coefs[i,maxLoc] = maxVal

def mp(dictionary,stimuli,k=None,minabs=None,posOnly=None):
    """
    Does matching pursuit on a batch of stimuli.

    Args:
        dictionary: Dictionary for matching pursuit. First axis should be dictionary element number.
        stimuli: Stimulus batch for matching pursuit. First axis should be stimulus number.
        k: Sparseness constraint. k dictionary elements will be used to represent stimuli.
        minabs: Minimum absolute value of the remaining signal to continue projection. If nothing is given, minabs is set to zero and k basis elements will be used.
        posOnly: If True, only positive coefficients will be used for representing signal.

    Returns:
        coeffs: List of dictionary element coefficients to be used for each stimulus.
    """
    if k is None:
        k = dictionary.shape[0]
    if minabs is None:
        minabs = 0.
    if posOnly is None:
        posOnly = False

    bs = cublas.Blas()

    numDict = dictionary.shape[0]
    numStim = stimuli.shape[0]
    dataLength = stimuli.shape[1]
    assert k <= numDict
    #Setup variables on GPU
    d_coefs = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    d_win = cuda.to_device(np.zeros(shape=(k,numStim),dtype=np.float32,order='F'))
    d_curCoef = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    d_coefABS = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    d_winners = cuda.to_device(np.zeros(shape=(k,numStim),dtype=np.int32,order='F'))
    d_delta = cuda.to_device(np.zeros_like(stimuli,dtype=np.float32,order='F'))
    d_coefsd = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    #Move args to GPU
    d_stim = cuda.to_device(np.array(stimuli,dtype=np.float32,order='F'))
    d_abs = cuda.to_device(np.zeros_like(stimuli,dtype=np.float32,order='F'))
    d_dict = cuda.to_device(np.array(dictionary,dtype=np.float32,order='F'))

    griddim1 = 32
    griddim2 = (32,32)
    assert numStim % 32 ==0 and dataLength % 32 == 0 and numDict % 32 == 0 
    blockdimmax = int(numStim/griddim1)
    blockdim2 = (int(numStim/griddim2[0]),int(dataLength/griddim2[1]))
    blockdimcoef = (int(numStim/griddim2[0]),int(numDict/griddim2[1])) 

    for ii in xrange(k):
        if minabs >= np.mean(np.absolute(d_abs.copy_to_host())):
            break
        bs.gemm('N','T',numStim,numDict,dataLength,1.,d_stim,d_dict,0.,d_curCoef)
        if ii != 0:
            for jj in xrange(ii-1):
                removeWinners[griddim1,blockdimmax](d_curCoef,d_winners)
        if posOnly:
            maxCoef[griddim1,blockdimmax](d_curCoef,d_coefs,d_winners,ii,0)
        else:
            cudaABS[griddim2,blockdimcoef](d_curCoef,d_coefABS)
            maxCoefs[griddim1,blockdimmax](d_coefABS,d_coefs,d_winners,ii,0)
        bs.gemm('N','N',numStim,dataLength,numDict,1.,d_coefsd,d_dict,0.,d_delta)
        bs.geam('N','N',numStim,dataLength,1.,d_stim,-1.,d_delta,d_stimt)
        bs.geam('N','N',numStim,dataLength,1.,d_stimt,0.,d_delta,d_stim)
    return d_coefs.copy_to_host()

        
