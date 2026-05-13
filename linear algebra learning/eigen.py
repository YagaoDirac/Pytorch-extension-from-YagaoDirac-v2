import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, iota
#from pytorch_yagaodirac_v2.Random import random_rotation_matrix,angle_to_rotation_matrix_2d
import random
import matplotlib.pyplot as plt


eig_vals_complex:torch.Tensor

if "2 dim manually" and False:

    iota_2 = iota(2)

    A = torch.tensor([  [3., 2],
                        [1,  2]])
    eig_vals_complex, eig_col_vectors_complex = torch.linalg.eig(A)
    assert eig_vals_complex.imag.eq(0.).all()
    assert eig_col_vectors_complex.imag.eq(0.).all()

    eig_vals = eig_vals_complex.real


    for ii in range(eig_vals.nelement()):
        eig_val = eig_vals[ii]
        A_minus_eig = A.clone()
        A_minus_eig[iota_2, iota_2] -= eig_val
        #print(A_minus_eig)
        for ii_row in range(A_minus_eig.shape[0]):
            plt.scatter(A_minus_eig[ii_row,0], A_minus_eig[ii_row,1], marker='o')
            pass
        pass
    plt.scatter(0,0,  marker='x')

    plt.show()
    pass

if "2 dim random" and False:

    iota_2 = iota(2)

    A = torch.randn(size=[2,2])
    eig_vals_complex, eig_col_vectors_complex = torch.linalg.eig(A)
    assert eig_vals_complex.imag.eq(0.).all()
    assert eig_col_vectors_complex.imag.eq(0.).all()

    eig_vals = eig_vals_complex.real

    for ii in range(eig_vals.nelement()):
        eig_val = eig_vals[ii]
        A_minus_eig = A.clone()
        A_minus_eig[iota_2, iota_2] -= eig_val
        #print(A_minus_eig)
        for ii_row in range(A_minus_eig.shape[0]):
            plt.scatter(A_minus_eig[ii_row,0], A_minus_eig[ii_row,1], marker='o')
            pass
        pass
    plt.scatter(0,0,  marker='x')
    plt.scatter(1,0,  marker='x')
    plt.scatter(0,1,  marker='x')

    plt.show()
    pass






if "3 dim random,  I know the visualization covers only 2 dims." and True:
    # you see 7 dots, but in fact they are 9.
    dim = 3
    iota_of_dim = iota(dim)

    A = torch.randn(size=[dim,dim])
    eig_vals_complex, eig_col_vectors_complex = torch.linalg.eig(A)
    assert eig_vals_complex.imag.eq(0.).all()
    assert eig_col_vectors_complex.imag.eq(0.).all()

    eig_vals = eig_vals_complex.real

    for ii in range(eig_vals.nelement()):
        eig_val = eig_vals[ii]
        A_minus_eig = A.clone()
        A_minus_eig[iota_of_dim, iota_of_dim] -= eig_val
        print(A_minus_eig)
        for ii_row in range(A_minus_eig.shape[0]):
            plt.scatter(A_minus_eig[ii_row,0], A_minus_eig[ii_row,1], marker='o')
            pass
        pass
    plt.scatter(0,0,  marker='x')
    plt.scatter(1,0,  marker='x')
    plt.scatter(0,1,  marker='x')

    plt.show()
    pass

























