import numpy as np
import scipy

def T(n1,n22,n3,omega,L):
    i = complex(0,1)
    c = scipy.constants.c
    n_air = 1.0
    n2 = complex(n22[0],-n22[1])
    a  = ((n2-n1)*(n2-n3))/((n2+n1)*(n2+n3))
    b  = np.exp(-2*i*n2*omega*L/c)

    k1 = (2*n2*(n1+n3))/((n2+n1)*(n2+n3))
    k2 = np.exp(-i*(n2-n_air)*omega*L/c)
    trans = k1*k2

    return trans
