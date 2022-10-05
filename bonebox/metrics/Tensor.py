"""

Mean intercept length, Gradient structure tensor, Sobel structure tensor.

QC 20220713

"""

import numpy as np

def vonMisesFisherDistribution(phi, kappa):
    """

    Implementation of the von Mises-Fisher distribution (vMFd) for the generalized MIL tensor

    Moreno et al. Med Phys 2012 Generalizing the mean intercept length tensor for gray-level images

    """

    return kappa*np.exp(kappa*np.cosine(phi)) / (4*np.pi*np.sinh(kappa))

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
    # GST = GradientStructureTensorCUDA(I)
    GSTNP = GradientStructureTensor(I)
    # print(f"cupy version: \n {GST}")
    print(f"numpy version: \n {GSTNP}")

    # Evaluate GSDT for an anisotropic image
    from scipy import ndimage
    np.random.seed(0)
    I = np.random.rand(200,200,200)
    Ifilt = ndimage.gaussian_filter(I, [1,2,3])
    # GST = GradientStructureTensorCUDA(Ifilt)
    # print(f"cupy version: \n {GST}")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(10, 4))
    axs[0].imshow(Ifilt[:,100,:].T,cmap="gray")
    axs[0].axis("off")
    axs[1].imshow(Ifilt[100,:,:].T,cmap="gray")
    axs[1].axis("off")