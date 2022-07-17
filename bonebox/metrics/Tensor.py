"""

Mean intercept length, Gradient structure tensor, Sobel structure tensor.

QC 20220713

"""

import numpy as np
import cupy as cp

def vonMisesFisherDistribution(phi, kappa):
    """

    Implementation of the von Mises-Fisher distribution (vMFd) for the generalized MIL tensor

    Moreno et al. Med Phys 2012 Generalizing the mean intercept length tensor for gray-level images

    """

    return kappa*np.exp(kappa*np.cosine(phi)) / (4*np.pi*np.sinh(kappa))

def readTensorCU():

    with open('Tensor.cu') as f:
        contents = f.read()

    return contents

def cuda_outer_function():
    """
    Returns a cuda function that can be used to compute the outer product
    https://docs.cupy.dev/en/stable/user_guide/kernel.html
    """

    TensorCU = readTensorCU()
    module = cp.RawModule(code=TensorCU)
    outer = module.get_function('outer')
    return outer

def cuda_outer(a,b,outer_func,Ngrid=None,Nblock=256):
    """
    cuda function for computing the outer product
    a (K,M)
    b (K,N)
    func - function object from cuda_outer_function
    returns out (K,M,N)
    """

    a = cp.array(a,dtype=cp.float64)
    b = cp.array(b,dtype=cp.float64)
    
    assert a.shape[0] == b.shape[0], "a and b must have the same number of vectors."
    
    K = a.shape[0]
    M = a.shape[1]
    N = b.shape[1]

    if Ngrid is None:
        Ngrid = np.ceil(K/Nblock).astype(int)

    out = cp.zeros((K,M,N),dtype=cp.float64)
    outer_func((Ngrid,),(Nblock,),(a,b,out,M,N,K))

    return out

def outerNP(Gv):
    """
    Input: Gv 
    applies outer product to each row
    """
    return np.array([np.outer(Gv[ind,:],Gv[ind,:]) for ind in range(Gv.shape[0])])

def GradientStructureTensor(I):
    """
    Gradient structure tensor described by Kersh
    """

    Inp = np.array(I,dtype=np.float64)

    Iag = np.gradient(Inp) # (dx, dy, dz)
    GvNP = np.ascontiguousarray(np.array([J.flatten() for J in Iag]).T) # (N,3)

    dyadicGvNP = outerNP(GvNP)
    GST = np.sum(dyadicGvNP,axis=0)

    return GST #(3,3)

def GradientStructureTensorCUDA(I,Ngrid=None,Nblock=256):
    """
    Gradient structure tensor described by Kersh
    """

    Icp = cp.array(I,dtype=cp.float64)

    Iag = cp.gradient(Icp) # (dx, dy, dz)
    GvCP = cp.ascontiguousarray(cp.array([J.flatten() for J in Iag]).T) # (N,3)

    # create CUDA outer function
    outer_func = cuda_outer_function()
    dyadicGvCP = cuda_outer(GvCP,GvCP,outer_func,Ngrid=None,Nblock=256)
    GST = cp.sum(dyadicGvCP,axis=0)

    GST = cp.asnumpy(GST)

    return GST #(3,3)

def svd(t):
    """
    returns a matrix of eigenvectors U and corresponding eigenvalues S
    """

    U, S, Vh = np.linalg.svd(t)

    return U, S

if __name__ == "__main__":

    # Test GradientStructureTensorCUDA
    np.random.seed(0)
    I = np.random.rand(100,100,100)

    # GST and GSTCUDA should be the same
    GST = GradientStructureTensorCUDA(I)
    GSTNP = GradientStructureTensor(I)
    print(f"cupy version: \n {GST}")
    print(f"numpy version: \n {GSTNP}")