import numpy as np 

def Eigspace_correlation(A,B, rank):
    W1 = A@A.T
    W2 = B@B.T

    U1,_,_ = np.linalg.svd(W1)
    U2,_,_ = np.linalg.svd(W2)

    R = U1[:,:rank].T @ U2[:,:rank]
    G = np.linalg.det(R)

    return G

def get_eigspaces(A,B):
    W1 = A@A.T
    W2 = B@B.T

    U1,s1,_ = np.linalg.svd(W1)
    U2,s2,_ = np.linalg.svd(W2)

    return U1, s1, U2, s2
